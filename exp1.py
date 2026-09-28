import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Setup and Data Definition
# ==========================================

# Create a separate folder to save the graphs
output_dir = "exp2_graphs"
os.makedirs(output_dir, exist_ok=True)

# Data for DenseNet-121[cite: 2]
densenet_data = {
    "test_accuracy": 9.170854271356784,
    "macro_f1": 0.08174548292623891 * 100, # Converting to percentage for plotting
    "epoch_history": [
        {"epoch": 1, "train_loss": 1.080, "train_acc": 62.75, "val_loss": 2.816, "val_acc": 8.67},
        {"epoch": 2, "train_loss": 0.687, "train_acc": 86.29, "val_loss": 3.191, "val_acc": 7.00},
        {"epoch": 3, "train_loss": 0.618, "train_acc": 90.23, "val_loss": 3.128, "val_acc": 7.73},
        {"epoch": 4, "train_loss": 0.580, "train_acc": 91.95, "val_loss": 3.313, "val_acc": 7.18},
        {"epoch": 5, "train_loss": 0.558, "train_acc": 92.98, "val_loss": 3.337, "val_acc": 8.32},
        {"epoch": 6, "train_loss": 0.545, "train_acc": 93.64, "val_loss": 3.213, "val_acc": 8.11},
        {"epoch": 7, "train_loss": 0.532, "train_acc": 94.35, "val_loss": 3.406, "val_acc": 9.12},
        {"epoch": 8, "train_loss": 0.516, "train_acc": 94.95, "val_loss": 3.383, "val_acc": 7.24},
        {"epoch": 9, "train_loss": 0.512, "train_acc": 95.23, "val_loss": 3.174, "val_acc": 8.56},
        {"epoch": 10, "train_loss": 0.497, "train_acc": 95.89, "val_loss": 3.216, "val_acc": 6.41},
        {"epoch": 11, "train_loss": 0.485, "train_acc": 96.51, "val_loss": 3.221, "val_acc": 7.90},
        {"epoch": 12, "train_loss": 0.478, "train_acc": 96.59, "val_loss": 3.199, "val_acc": 9.64},
        {"epoch": 13, "train_loss": 0.467, "train_acc": 97.11, "val_loss": 3.352, "val_acc": 8.11},
        {"epoch": 14, "train_loss": 0.461, "train_acc": 97.35, "val_loss": 3.376, "val_acc": 7.18},
        {"epoch": 15, "train_loss": 0.454, "train_acc": 97.73, "val_loss": 3.397, "val_acc": 8.35},
        {"epoch": 16, "train_loss": 0.444, "train_acc": 98.14, "val_loss": 3.336, "val_acc": 7.73},
        {"epoch": 17, "train_loss": 0.438, "train_acc": 98.36, "val_loss": 3.446, "val_acc": 8.73},
        {"epoch": 18, "train_loss": 0.433, "train_acc": 98.62, "val_loss": 3.438, "val_acc": 8.46},
        {"epoch": 19, "train_loss": 0.428, "train_acc": 98.78, "val_loss": 3.404, "val_acc": 9.46},
        {"epoch": 20, "train_loss": 0.425, "train_acc": 98.90, "val_loss": 3.434, "val_acc": 8.04},
        {"epoch": 21, "train_loss": 0.420, "train_acc": 99.12, "val_loss": 3.411, "val_acc": 9.15},
        {"epoch": 22, "train_loss": 0.417, "train_acc": 99.24, "val_loss": 3.429, "val_acc": 8.94},
        {"epoch": 23, "train_loss": 0.416, "train_acc": 99.22, "val_loss": 3.479, "val_acc": 8.46},
        {"epoch": 24, "train_loss": 0.416, "train_acc": 99.30, "val_loss": 3.437, "val_acc": 8.56},
        {"epoch": 25, "train_loss": 0.415, "train_acc": 99.34, "val_loss": 3.473, "val_acc": 8.01}
    ],
    "confusion_matrix": [
        [55, 31, 33, 291, 11],
        [133, 23, 0, 7, 0],
        [30, 284, 1, 10, 3],
        [42, 22, 3, 57, 123],
        [57, 40, 120, 206, 10]
    ]
}

# Data for EfficientNet-B3[cite: 3]
efficientnet_data = {
    "test_accuracy": 9.359296482412061,
    "macro_f1": 0.08739346678882016 * 100, # Converting to percentage
    "epoch_history": [
        {"epoch": 1, "train_loss": 1.179, "train_acc": 59.36, "val_loss": 2.704, "val_acc": 10.02},
        {"epoch": 2, "train_loss": 0.700, "train_acc": 85.02, "val_loss": 2.891, "val_acc": 9.19},
        {"epoch": 3, "train_loss": 0.596, "train_acc": 90.66, "val_loss": 2.914, "val_acc": 9.25},
        {"epoch": 4, "train_loss": 0.548, "train_acc": 92.66, "val_loss": 2.935, "val_acc": 8.18},
        {"epoch": 5, "train_loss": 0.521, "train_acc": 94.23, "val_loss": 2.951, "val_acc": 8.49},
        {"epoch": 6, "train_loss": 0.501, "train_acc": 95.23, "val_loss": 2.907, "val_acc": 8.28},
        {"epoch": 7, "train_loss": 0.488, "train_acc": 95.78, "val_loss": 2.911, "val_acc": 8.21},
        {"epoch": 8, "train_loss": 0.479, "train_acc": 96.15, "val_loss": 3.020, "val_acc": 7.35},
        {"epoch": 9, "train_loss": 0.463, "train_acc": 97.05, "val_loss": 2.938, "val_acc": 8.91},
        {"epoch": 10, "train_loss": 0.458, "train_acc": 97.26, "val_loss": 2.885, "val_acc": 9.57},
        {"epoch": 11, "train_loss": 0.454, "train_acc": 97.36, "val_loss": 3.032, "val_acc": 8.98},
        {"epoch": 12, "train_loss": 0.446, "train_acc": 97.74, "val_loss": 3.069, "val_acc": 7.97},
        {"epoch": 13, "train_loss": 0.440, "train_acc": 98.00, "val_loss": 2.980, "val_acc": 8.60},
        {"epoch": 14, "train_loss": 0.436, "train_acc": 98.25, "val_loss": 2.972, "val_acc": 7.31},
        {"epoch": 15, "train_loss": 0.431, "train_acc": 98.42, "val_loss": 2.939, "val_acc": 8.84},
        {"epoch": 16, "train_loss": 0.426, "train_acc": 98.73, "val_loss": 2.973, "val_acc": 8.77},
        {"epoch": 17, "train_loss": 0.423, "train_acc": 98.82, "val_loss": 2.988, "val_acc": 8.91},
        {"epoch": 18, "train_loss": 0.421, "train_acc": 98.83, "val_loss": 2.981, "val_acc": 8.63},
        {"epoch": 19, "train_loss": 0.417, "train_acc": 99.05, "val_loss": 3.015, "val_acc": 8.53},
        {"epoch": 20, "train_loss": 0.416, "train_acc": 99.08, "val_loss": 2.955, "val_acc": 9.01},
        {"epoch": 21, "train_loss": 0.413, "train_acc": 99.19, "val_loss": 2.994, "val_acc": 8.49},
        {"epoch": 22, "train_loss": 0.412, "train_acc": 99.30, "val_loss": 2.979, "val_acc": 8.53},
        {"epoch": 23, "train_loss": 0.413, "train_acc": 99.20, "val_loss": 2.950, "val_acc": 9.12},
        {"epoch": 24, "train_loss": 0.411, "train_acc": 99.37, "val_loss": 2.947, "val_acc": 8.84},
        {"epoch": 25, "train_loss": 0.412, "train_acc": 99.24, "val_loss": 2.946, "val_acc": 8.67}
    ],
    "confusion_matrix": [
        [40, 99, 62, 139, 81],
        [128, 30, 0, 1, 4],
        [14, 309, 1, 0, 4],
        [7, 36, 2, 17, 185],
        [35, 134, 138, 65, 61]
    ]
}

# Extraction
epochs = [d['epoch'] for d in densenet_data['epoch_history']]
dn_train_acc = [d['train_acc'] for d in densenet_data['epoch_history']]
dn_train_loss = [d['train_loss'] for d in densenet_data['epoch_history']]
dn_val_acc = [d['val_acc'] for d in densenet_data['epoch_history']]
dn_val_loss = [d['val_loss'] for d in densenet_data['epoch_history']]

en_train_acc = [d['train_acc'] for d in efficientnet_data['epoch_history']]
en_train_loss = [d['train_loss'] for d in efficientnet_data['epoch_history']]
en_val_acc = [d['val_acc'] for d in efficientnet_data['epoch_history']]
en_val_loss = [d['val_loss'] for d in efficientnet_data['epoch_history']]

class_labels = ['MDOSCC', 'Normal', 'OSMF', 'PDOSCC', 'WDOSCC']

# ==========================================
# 2. Graph Generation Functions
# ==========================================

# Graph 1: Training Accuracy Curves
plt.figure(figsize=(8, 6))
plt.plot(epochs, dn_train_acc, label='DenseNet-121 Train Acc', linewidth=2.5, marker='o', markersize=5)
plt.plot(epochs, en_train_acc, label='EfficientNet-B3 Train Acc', linewidth=2.5, marker='s', markersize=5)
plt.title('Synthetic Optimization: Training Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_training_accuracy_curves_2.png'), dpi=300)
plt.close()

# Graph 2: Training Loss Curves
plt.figure(figsize=(8, 6))
plt.plot(epochs, dn_train_loss, label='DenseNet-121 Train Loss', linewidth=2.5, marker='o', markersize=5, color='#d62728')
plt.plot(epochs, en_train_loss, label='EfficientNet-B3 Train Loss', linewidth=2.5, marker='s', markersize=5, color='#9467bd')
plt.title('Synthetic Optimization: Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Loss (Cross Entropy)', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '09_training_loss_curves_2.png'), dpi=300)
plt.close()

# Graph 3: Generalization Gap (Train vs Val Accuracy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(epochs, dn_train_acc, label='Train (Synthetic)', linewidth=2.5)
ax1.plot(epochs, dn_val_acc, label='Validation (Real)', linewidth=2.5, color='red')
ax1.set_title('DenseNet-121: Generalization Gap', fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy (%)')
ax1.grid(True, linestyle='--')
ax1.legend()

ax2.plot(epochs, en_train_acc, label='Train (Synthetic)', linewidth=2.5)
ax2.plot(epochs, en_val_acc, label='Validation (Real)', linewidth=2.5, color='red')
ax2.set_title('EfficientNet-B3: Generalization Gap', fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '11_generalization_gap_2.png'), dpi=300)
plt.close()

# Graph 4: Overall Performance Comparison Bar Chart
labels = ['DenseNet-121', 'EfficientNet-B3']
test_acc = [densenet_data['test_accuracy'], efficientnet_data['test_accuracy']]
macro_f1 = [densenet_data['macro_f1'], efficientnet_data['macro_f1']]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, test_acc, width, label='Real-World Test Accuracy', color='#1f77b4')
rects2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1-Score', color='#ff7f0e')

ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Pure-Synthetic Stress Test: Real-World Clinical Collapse', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0, 15) # Setting low y-limit to highlight the collapse
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.bar_label(rects1, padding=3, fmt='%.2f%%')
ax.bar_label(rects2, padding=3, fmt='%.2f%%')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, '01_overall_performance_comparison_2.png'), dpi=300)
plt.close()

# Function to plot Normalized Confusion Matrix
def plot_normalized_cm(cm, title, filename):
    cm_array = np.array(cm)
    cm_normalized = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Prediction Probability'})
    
    plt.title(title, fontweight='bold', pad=15)
    plt.ylabel('True Clinical Diagnosis', fontweight='bold')
    plt.xlabel('Model Prediction', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Graph 5 & 6: Confusion Matrices
plot_normalized_cm(densenet_data["confusion_matrix"], 
                   'DenseNet-121: Normalized Confusion Matrix', 
                   '14_normalized_densenet_confusion_matrix_2.png')

plot_normalized_cm(efficientnet_data["confusion_matrix"], 
                   'EfficientNet-B3: Normalized Confusion Matrix', 
                   '15_normalized_efficientnet_confusion_matrix_2.png')

print("All graphs successfully generated and saved in the 'exp2_graphs' folder.")