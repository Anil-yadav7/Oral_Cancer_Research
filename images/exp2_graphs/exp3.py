import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Setup and Data Definition
# ==========================================

# Create a separate folder to save the graphs
output_dir = "exp3_graphs"
os.makedirs(output_dir, exist_ok=True)

# Data for DenseNet-121 (Controlled Balancing Sweep)[cite: 4]
densenet_exp3_data = [
  {"alpha": 0.0,  "test_accuracy": 96.796, "macro_f1": 0.9733},
  {"alpha": 0.25, "test_accuracy": 95.728, "macro_f1": 0.9650},
  {"alpha": 0.5,  "test_accuracy": 95.917, "macro_f1": 0.9666},
  {"alpha": 0.75, "test_accuracy": 95.979, "macro_f1": 0.9659},
  {"alpha": 1.0,  "test_accuracy": 96.105, "macro_f1": 0.9674}
]

# Data for EfficientNet-B3 (Controlled Balancing Sweep)[cite: 5]
efficientnet_exp3_data = [
  {"alpha": 0.0,  "test_accuracy": 96.293, "macro_f1": 0.9690},
  {"alpha": 0.25, "test_accuracy": 96.105, "macro_f1": 0.9671},
  {"alpha": 0.5,  "test_accuracy": 96.356, "macro_f1": 0.9701},
  {"alpha": 0.75, "test_accuracy": 96.356, "macro_f1": 0.9690},
  {"alpha": 1.0,  "test_accuracy": 96.419, "macro_f1": 0.9694}
]

# Extraction[cite: 4, 5]
alphas = [d['alpha'] for d in densenet_exp3_data]
dn_test_acc = [d['test_accuracy'] for d in densenet_exp3_data]
en_test_acc = [d['test_accuracy'] for d in efficientnet_exp3_data]

# Convert Macro F1 to percentages for the heatmap[cite: 4, 5]
dn_macro_f1 = [d['macro_f1'] * 100 for d in densenet_exp3_data]
en_macro_f1 = [d['macro_f1'] * 100 for d in efficientnet_exp3_data]

# ==========================================
# 2. Graph Generation Functions
# ==========================================

# Graph 1: Test Accuracy vs. Alpha Line Plot
plt.figure(figsize=(8, 6))
plt.plot(alphas, dn_test_acc, label='DenseNet-121', linewidth=2.5, marker='o', markersize=8, color='#d62728')
plt.plot(alphas, en_test_acc, label='EfficientNet-B3', linewidth=2.5, marker='s', markersize=8, color='#1f77b4')

plt.title('Experiment 3: Test Accuracy vs. Alpha (Bounded Balancing)', fontsize=14, fontweight='bold')
plt.xlabel('Alpha (Fraction of Dataset Deficit Filled)', fontsize=12, fontweight='bold')
plt.ylabel('Real-World Test Accuracy (%)', fontsize=12, fontweight='bold')
plt.xticks(alphas)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_test_accuracy_vs_alpha.png'), dpi=300)
plt.close()

# Graph 2: Performance Heatmap (Macro F1 Scores)
heatmap_data = np.array([dn_macro_f1, en_macro_f1])
y_labels = ['DenseNet-121', 'EfficientNet-B3']
x_labels = [f"α={a}" for a in alphas]

plt.figure(figsize=(8, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", 
            xticklabels=x_labels, yticklabels=y_labels,
            cbar_kws={'label': 'Macro F1-Score (%)'})

plt.title('Experiment 3: Macro F1-Score Heatmap Across α-Sweeps', fontweight='bold', pad=15)
plt.xlabel('Alpha Mixing Ratio', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '13_performance_heatmap.png'), dpi=300)
plt.close()

print("All graphs for Experiment 3 successfully generated and saved in the 'exp3_graphs' folder.")