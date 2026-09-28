"""https://www.kaggle.com/code/anilkumar705/densenetperalpha/notebook"""
import os
import time
import json
import gc
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision import models, datasets
import torchvision.transforms.v2 as T
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# NOTE ON MATCHING: this script and sweep_efficientnetb3_matched.py are
# intentionally identical in every respect EXCEPT the backbone
# architecture and its native classifier layer shape:
#   - same optimizer recipe (AdamW, flat LR, weight decay)
#   - same warmup + cosine LR schedule
#   - same label smoothing, gradient clipping
#   - same train/eval augmentation and resolution (224x224)
#   - same classifier head shape (Dropout -> Linear)
#   - same batch size, worker count, data loading style (full arrays
#     loaded into RAM, matching the original DenseNet script's approach
#     — no memory-mapping)
#   - same crash-recovery checkpointing, resumable master-sweep logic,
#     single-GPU execution
# This isolates the architecture as the only real variable between runs.
# =====================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# 1. HARDWARE & SWEEP CONFIGURATION
# =====================================================================
mp.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Locked to single GPU
print(f"🚀 Initialized Single GPU Execution Environment: {DEVICE} (DenseNet-121, MATCHED)")
if torch.cuda.is_available():
    print(f"   └── GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   └── Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

NUM_CLASSES = 5
BATCH_SIZE = 32              # MATCHED
EPOCHS = 25                  # MATCHED
LEARNING_RATE = 3e-4         # MATCHED
WEIGHT_DECAY = 1e-4          # MATCHED
LABEL_SMOOTHING = 0.1        # MATCHED
DROPOUT_P = 0.3              # MATCHED
GRAD_CLIP_NORM = 1.0         # MATCHED
WARMUP_EPOCHS = 3            # MATCHED
NUM_WORKERS = 2              # MATCHED
LOG_INTERVAL = 25            # MATCHED
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.0]

RUNS_DIR = '/kaggle/working/runs/densenet121'
os.makedirs(RUNS_DIR, exist_ok=True)
MASTER_SUMMARY_PATH = '/kaggle/working/master_sweep_results_densenet121.json'  # Per-arch path — avoids clobbering the EfficientNet summary

# Same data root layout as the original DenseNet script (full-array RAM load, not mmap)
CACHE_DIR = '/kaggle/input/datasets/chakkilalaanilkumar/final-classification-dataset'
VAL_ROOT  = '/kaggle/input/datasets/anil701/orchid-official-validation-test/val/val'
TEST_ROOT = '/kaggle/input/datasets/anil701/orchid-official-validation-test/test/test'

# =====================================================================
# 2. DATASET & TRANSFORMS SETUP
# =====================================================================
print("\n[STEP 1/4] Loading cached NumPy image arrays fully into RAM...")
start_mem_load = time.time()
real_images  = np.load(f'{CACHE_DIR}/real_images.npy')
real_labels  = np.load(f'{CACHE_DIR}/real_labels.npy')
synth_images = np.load(f'{CACHE_DIR}/synth_images.npy')
synth_labels = np.load(f'{CACHE_DIR}/synth_labels.npy')
print(f"✅ Cache Loaded in {time.time() - start_mem_load:.2f}s | Real: {len(real_images)} | Synth: {len(synth_images)}")

class CachedArrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx].copy()).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])

def build_alpha_mix_indices(real_labels, synth_labels, alpha, num_classes, seed=SEED):
    rng = np.random.default_rng(seed)
    class_counts = np.bincount(real_labels, minlength=num_classes)
    majority = class_counts.max()

    real_idx = np.arange(len(real_labels))
    synth_chunks = []

    for c in range(num_classes):
        gap = int(majority - class_counts[c])
        n_needed = int(round(alpha * gap))
        candidates = np.where(synth_labels == c)[0]

        if n_needed > len(candidates):
            n_needed = len(candidates)

        if n_needed > 0:
            synth_chunks.append(rng.choice(candidates, size=n_needed, replace=False))

    synth_idx = np.concatenate(synth_chunks) if synth_chunks else np.array([], dtype=int)
    return real_idx, synth_idx

train_transform = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.RandomResizedCrop((224, 224), scale=(0.8, 1.0), antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize((224, 224), antialias=True),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\n[STEP 2/4] Validating and building evaluation DataLoaders...")
val_dataset  = datasets.ImageFolder(root=VAL_ROOT,  transform=eval_transform)
test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=eval_transform)
assert val_dataset.classes == test_dataset.classes, "Val/Test class mismatch!"
class_names = val_dataset.classes

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

print(f"✅ Data Validation Ready | Classes ({len(class_names)}): {class_names}")

# Load existing master summary if resuming across sessions
if os.path.exists(MASTER_SUMMARY_PATH):
    with open(MASTER_SUMMARY_PATH, 'r') as f:
        master_summary = json.load(f)
    completed_alphas = {entry["alpha"] for entry in master_summary}
    print(f"🔄 Resuming Existing Master Sweep. Fully Completed Alphas: {list(completed_alphas)}")
else:
    master_summary = []
    completed_alphas = set()

# =====================================================================
# 3. MASTER ABLATION SWEEP LOOP WITH CRASH/CANCEL PROTECTION
# =====================================================================
print("\n[STEP 3/4] Starting Ablation Experiments...")
print(f"[CONFIG] batch_size={BATCH_SIZE} | lr={LEARNING_RATE} | weight_decay={WEIGHT_DECAY} | "
      f"label_smoothing={LABEL_SMOOTHING} | dropout={DROPOUT_P} | grad_clip={GRAD_CLIP_NORM} | "
      f"warmup_epochs={WARMUP_EPOCHS}")

try:
    for current_alpha in ALPHAS:
        if current_alpha in completed_alphas:
            print(f"\n⏩ [SKIP] Alpha = {current_alpha:.2f} is already fully completed.")
            continue

        RUN_TAG = f"densenet121_alpha_{str(current_alpha).replace('.', 'p')}"
        OUT_DIR = os.path.join(RUNS_DIR, RUN_TAG)
        os.makedirs(OUT_DIR, exist_ok=True)
        CKPT_PATH = os.path.join(OUT_DIR, 'latest_checkpoint.pt')

        print("\n" + "=" * 80)
        print(f"🔥 SWEEP RUN | Arch: DenseNet-121 | Alpha: {current_alpha:.2f} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
        print("=" * 80)

        # Datasets & DataLoader setup
        real_idx, synth_idx = build_alpha_mix_indices(real_labels, synth_labels, current_alpha, NUM_CLASSES)
        real_ds  = Subset(CachedArrayDataset(real_images,  real_labels,  train_transform), real_idx)
        synth_ds = Subset(CachedArrayDataset(synth_images, synth_labels, train_transform), synth_idx)
        combined_ds = ConcatDataset([real_ds, synth_ds])

        train_loader = DataLoader(
            combined_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=True
        )

        # Initialize Architecture & Components
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P, inplace=True),
            nn.Linear(in_features, NUM_CLASSES)
        )
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])

        scaler = GradScaler('cuda')

        start_epoch = 0
        best_val_acc = 0.0
        best_state = None
        history = []

        # --- RESUME FROM IN-PROGRESS CHECKPOINT IF CRASHED/CANCELLED ---
        if os.path.exists(CKPT_PATH):
            print(f"📦 Found crash recovery checkpoint at: {CKPT_PATH}")
            checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            best_state = checkpoint['best_state']
            history = checkpoint['history']

            print(f"🔄 SUCCESSFULLY RESUMED! Continuing from Epoch {start_epoch + 1}/{EPOCHS}")
            print(f"   └── Previous Best Val Acc: {best_val_acc:.2f}%")
        else:
            print(f"📊 Dataset Composition -> Real: {len(real_idx)} | Synthetic: {len(synth_idx)} | Total: {len(combined_ds)}")

        # --- EPOCH LOOP ---
        for epoch in range(start_epoch, EPOCHS):
            epoch_start_time = time.time()
            current_lr = scheduler.get_last_lr()[0]

            print(f"\n--- Epoch {epoch+1:02d}/{EPOCHS} [LR: {current_lr:.6f}] ---")

            # Training Phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            train_start = time.time()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == len(train_loader):
                    batch_acc = 100. * correct / total
                    avg_loss = running_loss / total
                    print(f"  [Train] Step [{batch_idx+1:03d}/{len(train_loader):03d}] "
                          f"| Running Loss: {avg_loss:.4f} | Running Acc: {batch_acc:.2f}%")

            scheduler.step()
            train_loss = running_loss / total
            train_acc = 100. * correct / total
            train_duration = time.time() - train_start

            torch.cuda.empty_cache()

            # Validation Phase
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            val_start = time.time()

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= val_total
            val_acc = 100. * val_correct / val_total
            val_duration = time.time() - val_start
            total_epoch_time = time.time() - epoch_start_time

            # Best Model Tracking
            checkpoint_str = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(OUT_DIR, 'best_model.pt'))
                checkpoint_str = "⭐ BEST MODEL SAVED"

            history.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_s": total_epoch_time
            })

            print(f"📌 Epoch {epoch+1:02d} Summary [{total_epoch_time:.1f}s total (Train: {train_duration:.1f}s, Val: {val_duration:.1f}s)]")
            print(f"   └── Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   └── Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% {checkpoint_str}")

            # --- SAVE PROGRESS CHECKPOINT AFTER EVERY EPOCH ---
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_state': best_state,
                'history': history,
            }, CKPT_PATH)
            print(f"💾 Intermediary progress saved to: {CKPT_PATH}")

        # --- Final Test Evaluation Phase (Runs only after all epochs complete) ---
        print("\n" + "-" * 60)
        print(f"🧪 EVALUATING BEST CHECKPOINT ON UNSEEN TEST SET (Alpha={current_alpha})")
        print("-" * 60)

        model.load_state_dict(best_state)
        model.eval()

        test_correct, test_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with autocast('cuda'):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        test_acc = 100. * test_correct / test_total
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
        report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        cm = confusion_matrix(all_labels, all_preds).tolist()

        print(f"\n🎯 FINAL TEST ACCURACY (Alpha={current_alpha}): {test_acc:.2f}%\n")
        print(report_str)

        run_results = {
            "architecture": "densenet121",
            "alpha": current_alpha,
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
            "class_names": class_names,
            "per_class_report": report_dict,
            "confusion_matrix": cm,
            "epoch_history": history,
        }
        with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
            json.dump(run_results, f, indent=2)

        summary_entry = {
            "architecture": "densenet121",
            "alpha": current_alpha,
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "train_images": len(combined_ds),
            "real_images": len(real_idx),
            "synth_images": len(synth_idx),
            "run_directory": OUT_DIR
        }
        master_summary.append(summary_entry)

        with open(MASTER_SUMMARY_PATH, 'w') as f:
            json.dump(master_summary, f, indent=2)

        # Cleanup interim checkpoint file once run completes fully
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)

        print(f"💾 Saved run logs to: {OUT_DIR}/results.json")
        print(f"📁 Persisted master summary to: {MASTER_SUMMARY_PATH}")

        # Memory Cleanup
        del model, optimizer, scheduler, train_loader, best_state
        gc.collect()
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n🛑 TRAINING MANUALLY CANCELLED / INTERRUPTED BY USER!")
    print("💾 Progress saved up to the last completed epoch. Re-running the notebook will resume automatically.")
    sys.exit(0)

# =====================================================================
# 4. EXPERIMENT SUMMARY TABLE
# =====================================================================
print("\n" + "=" * 80)
print("[STEP 4/4] 🎉 ALL ABLATION SWEEPS FINISHED! SUMMARY TABLE (DenseNet-121):")
print("=" * 80)
print(f"{'ALPHA':<8} | {'TRAIN IMGS':<10} | {'BEST VAL ACC':<14} | {'TEST ACC':<12} | {'MACRO F1':<10}")
print("-" * 80)
for res in master_summary:
    print(f"{res['alpha']:<8.2f} | {res['train_images']:<10d} | {res['best_val_accuracy']:<13.2f}% | {res['test_accuracy']:<11.2f}% | {res['macro_f1']:<10.4f}")
print("=" * 80)