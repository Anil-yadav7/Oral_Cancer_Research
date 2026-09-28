'''https://www.kaggle.com/code/anil701/efficient25k'''

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, datasets
import torchvision.transforms.v2 as T
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix

# ========================================================
# 1. CONFIGURATION & HYPERPARAMETERS
#    NOTE: Hyperparameters below are intentionally matched with
#    train_densenet121_matched.py so that the two backbones can be
#    compared fairly (same optimizer recipe, LR schedule, weight decay,
#    label smoothing, dropout, batch size, num_workers). Only the
#    architecture itself differs. Epoch counts are deliberately left
#    as-is per run (25 total for DenseNet via resume, 20 for EfficientNet).
# ========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Locked to Single GPU
print(f"Training on device: {DEVICE} (EfficientNet-B3 - Single GPU Protocol, MATCHED hyperparams)")

mp.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

NUM_CLASSES = 5
BATCH_SIZE = 32          # MATCHED across both scripts (was 64)
EPOCHS = 25
TARGET_PER_CLASS = 5000
LOG_INTERVAL = 50        # MATCHED reporting cadence (was 5 — too noisy to compare logs against DenseNet's 50)

BACKBONE_LR = 1e-4       # MATCHED (was a single flat 1e-3 for all params)
HEAD_LR = 1e-3           # MATCHED
WEIGHT_DECAY = 1e-2      # MATCHED (was 1e-4)
LABEL_SMOOTHING = 0.1    # MATCHED (was 0.0 / none)
DROPOUT_P = 0.4          # MATCHED (was EfficientNet-B3's default ~0.3)
NUM_WORKERS = 0          # MATCHED (was 4 — avoids RAM duplication of the cached synth arrays,
                          # consistent with the DenseNet script's approach)

RUN_TAG = "efficientnetb3_pure_fake_25k_stress_test_matched"
OUT_DIR = f"/kaggle/working/runs/{RUN_TAG}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run outputs -> {OUT_DIR}")

VAL_ROOT  = '/kaggle/input/datasets/anil701/orchid-official-validation-test/val/val'
TEST_ROOT = '/kaggle/input/datasets/anil701/orchid-official-validation-test/test/test'
CACHE_DIR = '/kaggle/input/datasets/chakkilalaanilkumar/final-classification-dataset'

# ========================================================
# 2. DATASET & PURE FAKE IMPLEMENTATION
# ========================================================
print("\n[SETUP] Loading synthetic cache directly into System RAM for maximum I/O speed...")
synth_images = np.load(f'{CACHE_DIR}/synth_images.npy')
synth_labels = np.load(f'{CACHE_DIR}/synth_labels.npy')
print("[SETUP] Cache loaded successfully into RAM!")

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

def build_pure_fake_indices(synth_labels, target_size, num_classes, seed=SEED):
    rng = np.random.default_rng(seed)
    synth_chunks = []

    print(f"\nBuilding PURE FAKE dataset at target={target_size} per class:")
    for c in range(num_classes):
        candidates = np.where(synth_labels == c)[0]
        to_generate = min(target_size, len(candidates))

        print(f"  class {c}: real=0 + synth={to_generate} = {to_generate} (100.0% synthetic)")
        if to_generate < target_size:
            print(f"    WARNING: requested {target_size}, only {len(candidates)} available — using all")

        synth_chunks.append(rng.choice(candidates, size=to_generate, replace=False))

    return np.concatenate(synth_chunks) if synth_chunks else np.array([], dtype=int)

synth_idx = build_pure_fake_indices(synth_labels, TARGET_PER_CLASS, NUM_CLASSES)

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = T.Compose([
    T.Resize((300, 300), antialias=True),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset  = datasets.ImageFolder(root=VAL_ROOT,  transform=eval_transform)
test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=eval_transform)
assert val_dataset.classes == test_dataset.classes, "Val/test class-folder order mismatch!"
class_names = val_dataset.classes

print(f"\n[SETUP] Initializing DataLoaders (num_workers={NUM_WORKERS} to prevent RAM duplication)...")

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

combined = Subset(CachedArrayDataset(synth_images, synth_labels, train_transform), synth_idx)
train_loader = DataLoader(
    combined, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
print(f"\nDATASET BUILT | Real: 0 + Synth: {len(combined)} = {len(combined)} Total Images (global batch={BATCH_SIZE})")
print(f"Val: {len(val_dataset)} images | Test: {len(test_dataset)} images | Classes: {class_names}")

# ========================================================
# 3. MODEL INITIALIZATION & TRAINING
# ========================================================
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=DROPOUT_P, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params = [p for n, p in model.named_parameters() if "classifier" in n]

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': BACKBONE_LR},
    {'params': head_params,     'lr': HEAD_LR}
], weight_decay=WEIGHT_DECAY)

scaler = GradScaler('cuda')

best_val_acc = 0.0
best_state = None
history = []
start_epoch = 0

# Resume from a previous run's checkpoint if one exists in this OUT_DIR (unchanged from original)
resume_path = f'{OUT_DIR}/latest_checkpoint.pt'
if os.path.exists(resume_path):
    print(f"\nFound existing checkpoint at {resume_path} — resuming...")
    ckpt = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    best_val_acc = ckpt["best_val_acc"]
    history = ckpt["history"]
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from epoch {start_epoch} | best_val_acc so far: {best_val_acc:.2f}%")

# OneCycleLR MATCHED with the DenseNet script. Built after the possible resume so
# steps_per_epoch/epochs reflect the actual training loader/schedule length.
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[BACKBONE_LR, HEAD_LR],
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    last_epoch=(start_epoch * len(train_loader) - 1) if start_epoch > 0 else -1
)

print("\nSTARTING TRAINING LOOP...")
print(f"[CONFIG] batch_size={BATCH_SIZE} | backbone_lr={BACKBONE_LR} | head_lr={HEAD_LR} | "
      f"weight_decay={WEIGHT_DECAY} | label_smoothing={LABEL_SMOOTHING} | dropout={DROPOUT_P}")

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    print(f"\n>>> 🟢 EPOCH {epoch+1}/{EPOCHS} STARTING <<<")

    # --- TRAIN PHASE ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    log_start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == len(train_loader):
            current_loss = loss.item()
            current_acc = 100. * correct / total
            current_lr = optimizer.param_groups[1]['lr']
            elapsed = time.time() - log_start_time
            print(f"    -> Train Batch [{batch_idx+1:04d}/{len(train_loader)}] | LR: {current_lr:.6f} | Loss: {current_loss:.4f} | Acc: {current_acc:.2f}% | Time: {elapsed:.1f}s")
            log_start_time = time.time()

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == len(val_loader):
                elapsed_val = time.time() - val_start_time
                print(f"    -> Val Batch [{batch_idx+1:04d}/{len(val_loader)}] processed | Time: {elapsed_val:.1f}s")
                val_start_time = time.time()

    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    epoch_time = time.time() - epoch_start

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        state_dict_to_save = model.state_dict()
        best_state = {k: v.cpu().clone() for k, v in state_dict_to_save.items()}
        torch.save(best_state, f'{OUT_DIR}/best_model.pt')
        print(f"  [SAVE] 🏆 New best validation accuracy reached: {val_acc:.2f}%")

    history.append({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc, "time_s": epoch_time})

    current_state_dict = model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": current_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    }, f'{OUT_DIR}/latest_checkpoint.pt')

    print(f"Epoch {epoch+1:02d}/{EPOCHS} [{epoch_time:.0f}s] | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    if epoch == start_epoch:
        projected_hours = (epoch_time * (EPOCHS - start_epoch)) / 3600
        print(f"  [ESTIMATE] Epoch {epoch+1} took {epoch_time:.0f}s -> "
              f"projected total for remaining epochs: {projected_hours:.2f} hours")
        if projected_hours > 4:
            print(f"  [WARNING] Projected runtime exceeds 4 hours — consider stopping and "
                  f"checking num_workers/CACHE_DIR before continuing.")

# ========================================================
# 4. FINAL TEST EVALUATION
# ========================================================
print("\n" + "-" * 60)
print(f"FINAL TEST EVALUATION — {RUN_TAG}")
print("-" * 60)

best_model_path = f'{OUT_DIR}/best_model.pt'
if os.path.exists(best_model_path):
    loaded_best_state = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(loaded_best_state)
else:
    print(f"  [WARNING] Could not find {best_model_path}. Using current model state for testing.")

model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []
test_start_time = time.time()

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with autocast('cuda'):
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if (batch_idx + 1) % LOG_INTERVAL == 0 or (batch_idx + 1) == len(test_loader):
            elapsed_test = time.time() - test_start_time
            print(f"    -> Test Batch [{batch_idx+1:04d}/{len(test_loader)}] processed | Time: {elapsed_test:.1f}s")
            test_start_time = time.time()

test_acc = 100. * test_correct / test_total
report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
cm = confusion_matrix(all_labels, all_preds).tolist()

print(f"\nTEST ACCURACY: {test_acc:.2f}%\n")
print(report_str)

run_results = {
    "run": RUN_TAG,
    "architecture": "efficientnet_b3",
    "target_per_class": TARGET_PER_CLASS,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "backbone_lr": BACKBONE_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "dropout": DROPOUT_P,
    "test_accuracy": test_acc,
    "best_val_accuracy": best_val_acc,
    "train_images": len(combined),
    "real_images": 0,
    "synth_images": len(combined),
    "per_class_report": report_dict,
    "confusion_matrix": cm,
    "epoch_history": history,
}

with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(run_results, f, indent=2)

print(f"Results saved to {OUT_DIR}/results.json")