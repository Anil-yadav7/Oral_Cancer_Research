import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Setup and Data Definition
# ==========================================

# Create a separate folder to save the graphs
output_dir = "exp4_graphs"
os.makedirs(output_dir, exist_ok=True)

# Extraction of Alphas and Test Accuracies[cite: 6, 7]
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

# DenseNet-121 Test Accuracies[cite: 6]
dn_test_acc = [95.665, 94.660, 93.907, 93.027, 93.216]

# EfficientNet-B3 Test Accuracies[cite: 7]
en_test_acc = [96.984, 96.105, 96.168, 96.105, 96.042]

# Per-Class F1 Scores (Baseline vs. Max Expansion) converted to percentages[cite: 6, 7]
class_labels = ['MDOSCC', 'Normal', 'OSMF', 'PDOSCC', 'WDOSCC']

# DenseNet-121[cite: 6]
dn_f1_baseline = [93.00, 97.89, 99.08, 97.79, 93.63]
dn_f1_alpha1   = [88.56, 97.87, 98.94, 96.58, 89.62]

# EfficientNet-B3[cite: 7]
en_f1_baseline = [95.42, 98.18, 99.08, 98.99, 95.32]
en_f1_alpha1   = [93.45, 98.48, 99.08, 97.99, 94.20]

# ==========================================
# 2. Graph Generation Functions
# ==========================================

# Graph 1: Test Accuracy vs. Alpha Line Plot (The Dilution Curve)
plt.figure(figsize=(8, 6))
plt.plot(alphas, dn_test_acc, label='DenseNet-121', linewidth=2.5, marker='o', markersize=8, color='#d62728')
plt.plot(alphas, en_test_acc, label='EfficientNet-B3', linewidth=2.5, marker='s', markersize=8, color='#1f77b4')

plt.title('Experiment 4: The Synthetic Dilution Threshold', fontsize=14, fontweight='bold')
plt.xlabel('Alpha (Fraction of 5,000-Image Target Expansion)', fontsize=12, fontweight='bold')
plt.ylabel('Real-World Test Accuracy (%)', fontsize=12, fontweight='bold')
plt.xticks(alphas)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_test_accuracy_vs_alpha_2.png'), dpi=300)
plt.close()

# Graph 2: Architecture Performance Gap Bar Chart (Baseline vs Max Expansion)
labels = ['DenseNet-121', 'EfficientNet-B3']
baseline_accs = [dn_test_acc[0], en_test_acc[0]]
max_exp_accs = [dn_test_acc[-1], en_test_acc[-1]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline (α = 0.0)', color='#2ca02c')
rects2 = ax.bar(x + width/2, max_exp_accs, width, label='Max Over-Saturation (α = 1.0)', color='#d62728')

ax.set_ylabel('Real-World Test Accuracy (%)', fontweight='bold')
ax.set_title('Architectural Resilience to Generative Dilution', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(90, 100) # Zoomed in to clearly highlight the percentage drop
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.bar_label(rects1, padding=3, fmt='%.2f%%')
ax.bar_label(rects2, padding=3, fmt='%.2f%%')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, '07_architecture_performance_gap.png'), dpi=300)
plt.close()

# Graph 3: Per-Class F1 Degradation Heatmap
# Structuring a 4x5 matrix to show exactly where the models lost predictive power
heatmap_data = np.array([
    dn_f1_baseline,
    dn_f1_alpha1,
    en_f1_baseline,
    en_f1_alpha1
])

y_labels = [
    'DenseNet Baseline', 
    'DenseNet (α=1.0)', 
    'EfficientNet Baseline', 
    'EfficientNet (α=1.0)'
]

plt.figure(figsize=(10, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", 
            xticklabels=class_labels, yticklabels=y_labels,
            cbar_kws={'label': 'F1-Score (%)'}, vmin=85, vmax=100)

plt.title('Experiment 4: Per-Class F1 Degradation at Maximum Synthetic Saturation', fontweight='bold', pad=15)
plt.xlabel('Diagnostic Categories', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '17_per_class_f1_heatmap.jpg'), dpi=300)
plt.close()

print("All graphs for Experiment 4 successfully generated and saved in the 'exp4_graphs' folder.")