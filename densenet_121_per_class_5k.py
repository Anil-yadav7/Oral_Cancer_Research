"""https://www.kaggle.com/code/morugupranaychandra/121dense5kperalpha/output?scriptVersionId=343905509"""

import os
import time
import json
import gc
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

# ========================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ========================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE} (DenseNet-121)")

mp.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True  

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 20  
TARGET_PER_CLASS = 5000

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE_OUT_DIR = "/kaggle/working/runs/densenet_alpha_sweeps"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

SWEEP_STATUS_FILE = os.path.join(BASE_OUT_DIR, "sweep_status.json")

VAL_ROOT  = '/kaggle/input/datasets/anil701/orchid-official-validation-test/val/val'
TEST_ROOT = '/kaggle/input/datasets/anil701/orchid-official-validation-test/test/test'

# ========================================================
# 2. STATIC DATASET PREPARATION (Done Once)
# ========================================================
print("Loading cached numpy arrays into memory mapping...")
real_images  = np.load('/kaggle/input/datasets/chakkilalaanilkumar/final-classification-dataset/real_images.npy',  mmap_mode='r')
real_labels  = np.load('/kaggle/input/datasets/chakkilalaanilkumar/final-classification-dataset/real_labels.npy')
synth_images = np.load('/kaggle/input/datasets/chakkilalaanilkumar/final-classification-dataset/synth_images.npy', mmap_mode='r')
synth_labels = np.load('/kaggle/input/datasets/chakkilalaanilkumar/final-classification-dataset/synth_labels.npy')

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

def build_indices_by_formula(real_labels, synth_labels, target_size, alpha, num_classes, seed=SEED):
    rng = np.random.default_rng(seed)
    class_counts = np.bincount(real_labels, minlength=num_classes)
    
    real_idx = np.arange(len(real_labels))
    synth_chunks = []

    for c in range(num_classes):
        rc = class_counts[c]
        deficit = max(0, target_size - rc) 
        to_generate = int(round(alpha * deficit))
        
        if to_generate > 0:
            candidates = np.where(synth_labels == c)[0]
            if to_generate > len(candidates):
                to_generate = len(candidates)
            synth_chunks.append(rng.choice(candidates, size=to_generate, replace=False))

    synth_idx = np.concatenate(synth_chunks) if synth_chunks else np.array([], dtype=int)
    return real_idx, synth_idx

# OPTIMIZATION 1: Added ColorJitter to combat histopathology stain variance
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.1, contrast=0.1),
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
class_names = val_dataset.classes

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True, drop_last=False)

# ========================================================
# 3. ALPHA SWEEP EXECUTION LOOP WITH RESUMPTION
# ========================================================
if os.path.exists(SWEEP_STATUS_FILE):
    with open(SWEEP_STATUS_FILE, 'r') as f:
        completed_alphas = json.load(f).get("completed", [])
else:
    completed_alphas = []

for current_alpha in ALPHAS:
    if current_alpha in completed_alphas:
        print(f"\n>>> SKIPPING ALPHA = {current_alpha} (Already marked as completed) <<<")
        continue

    alpha_start_time = time.time()
    RUN_TAG = f"densenet_sweep_alpha_{str(current_alpha).replace('.', 'p')}"
    OUT_DIR = os.path.join(BASE_OUT_DIR, RUN_TAG)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"STARTING DENSENET-121 RUN FOR ALPHA = {current_alpha}")
    print("=" * 60)

    # 3.1 Build Dynamic Dataset
    real_idx, synth_idx = build_indices_by_formula(real_labels, synth_labels, TARGET_PER_CLASS, current_alpha, NUM_CLASSES)
    real_ds  = Subset(CachedArrayDataset(real_images,  real_labels,  train_transform), real_idx)
    synth_ds = Subset(CachedArrayDataset(synth_images, synth_labels, train_transform), synth_idx)
    combined = ConcatDataset([real_ds, synth_ds])

    train_loader = DataLoader(
        combined, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True, drop_last=True
    )
    
    print(f"DATASET BUILT | Total Training Images: {len(combined)}")

    # 3.2 Initialize DenseNet-121 Model & Training Components
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    
    # OPTIMIZATION 2: Replaced bare Linear layer with Dropout Regularized Head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    # OPTIMIZATION 3: Added Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # OPTIMIZATION 4: Added Cosine Annealing LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler('cuda')

    best_val_acc = 0.0
    best_state = None
    history = []
    start_epoch = 0

    # 3.2.1 Check for existing epoch checkpoints to resume
    checkpoint_path = os.path.join(OUT_DIR, "last_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"\n--> Found interrupted checkpoint! Resuming progress safely...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_val_acc = checkpoint['best_val_acc']
        best_state = checkpoint['best_state']
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f"--> Continuing from Epoch {start_epoch + 1}/{EPOCHS}\n")

    # 3.3 Training Loop
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            # OPTIMIZATION 5: Gradient Clipping to prevent unstable jumps
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total
        
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
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

        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tmp_best = f'{OUT_DIR}/best_model.pt.tmp'
            torch.save(best_state, tmp_best)
            os.replace(tmp_best, f'{OUT_DIR}/best_model.pt')

        history.append({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, 
                        "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr, "time_s": epoch_time})
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} [{epoch_time:.0f}s] | LR: {current_lr:.6f} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}%")

        tmp_ckpt = f'{checkpoint_path}.tmp'
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_state': best_state,
            'history': history
        }, tmp_ckpt)
        os.replace(tmp_ckpt, checkpoint_path)

    # 3.4 Final Test Evaluation for Current Alpha
    print(f"\nExecuting final test evaluation for Alpha = {current_alpha}...")
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
    
    print(f"TEST ACCURACY (Alpha {current_alpha}): {test_acc:.2f}%")

    alpha_total_time = time.time() - alpha_start_time

    run_results = {
        "run": RUN_TAG,
        "architecture": "densenet121",
        "alpha": current_alpha,
        "target_per_class": TARGET_PER_CLASS,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "total_time_s": alpha_total_time,
        "train_images_total": len(combined),
        "real_images_used": len(real_idx),
        "synth_images_used": len(synth_idx),
        "per_class_report": report_dict,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "epoch_history": history,
    }

    tmp_res = f'{OUT_DIR}/results.json.tmp'
    with open(tmp_res, 'w') as f:
        json.dump(run_results, f, indent=2)
    os.replace(tmp_res, f'{OUT_DIR}/results.json')
        
    completed_alphas.append(current_alpha)
    tmp_status = f'{SWEEP_STATUS_FILE}.tmp'
    with open(tmp_status, 'w') as f:
        json.dump({"completed": completed_alphas}, f)
    os.replace(tmp_status, SWEEP_STATUS_FILE)
        
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    # 3.5 Cleanup & Memory Release
    del model, optimizer, scaler, scheduler, train_loader, real_ds, synth_ds, combined
    gc.collect()
    torch.cuda.empty_cache()
    print(f"--- Completed Alpha {current_alpha} and released memory ---")


# ========================================================
# 4. FINAL AGGREGATION FOR THESIS PLOTTING
# ========================================================
print("\n" + "=" * 60)
print("AGGREGATING MASTER RESULTS DIRECTORY")
print("=" * 60)

master_summary = []
for a in ALPHAS:
    run_folder = f"densenet_sweep_alpha_{str(a).replace('.', 'p')}"
    res_path = os.path.join(BASE_OUT_DIR, run_folder, "results.json")
    
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            data = json.load(f)
            master_summary.append({
                "alpha": data["alpha"],
                "architecture": data["architecture"],
                "test_accuracy": data["test_accuracy"],
                "best_val_accuracy": data["best_val_accuracy"],
                "real_images": data["real_images_used"],
                "synth_images": data["synth_images_used"],
                "total_train_images": data["train_images_total"],
                "macro_f1": data["per_class_report"]["macro avg"]["f1-score"],
                "weighted_f1": data["per_class_report"]["weighted avg"]["f1-score"]
            })

tmp_master = f'{master_file}.tmp'
master_file = os.path.join(BASE_OUT_DIR, "densenet_master_sweep_summary.json")
with open(tmp_master, 'w') as f:
    json.dump(master_summary, f, indent=4)
os.replace(tmp_master, master_file)

print(f"Master summary perfectly compiled and saved to: {master_file}")
print("All DenseNet-121 alpha sweeps completed successfully! Ready for visualization.")