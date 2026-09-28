
# ============================================================
# ALPHA SWEEP COMPARISON
# DenseNet121 vs EfficientNet-B3
#
# Files:
#   1. per_alpha_densenet(1).json
#   2. per_alpha_efficientnet(1).json
#
# Graphs generated:
#   01. Test Accuracy vs Alpha
#   02. Best Validation Accuracy vs Alpha
#   03. Macro F1-score vs Alpha
#   04. Macro Precision vs Alpha
#   05. Macro Recall vs Alpha
#   06. Combined Performance Curves
#   07. Performance Difference Between Architectures
#   08. Training Dataset Size vs Alpha
#   09. Synthetic Images Added vs Alpha
#   10. Synthetic Data Percentage vs Alpha
#   11. Accuracy vs Synthetic Data Percentage
#   12. Macro F1 vs Synthetic Data Percentage
#   13. Architecture Performance Heatmap
#   14. Best Alpha Configuration Summary
#
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DENSENET_FILE = "per_alpha_densenet.json"
EFFICIENTNET_FILE = "per_alpha_efficientnet.json"

OUTPUT_DIR = Path("alpha_sweep_comparison_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# ============================================================
# LOAD RESULTS
# ============================================================

with open(DENSENET_FILE, "r") as f:
    densenet = json.load(f)

with open(EFFICIENTNET_FILE, "r") as f:
    efficientnet = json.load(f)


# ============================================================
# SORT RESULTS BY ALPHA
# ============================================================

densenet = sorted(densenet, key=lambda x: x["alpha"])
efficientnet = sorted(efficientnet, key=lambda x: x["alpha"])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_show(filename):
    """Save figure in high resolution and display."""
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def get_values(data, key, percentage=False):
    values = [item[key] for item in data]

    if percentage:
        values = [v * 100 if v <= 1 else v for v in values]

    return np.array(values)


def annotate_best(x_values, y_values, label="Best"):
    """Highlight the best point in a curve."""

    best_idx = np.argmax(y_values)

    plt.scatter(
        x_values[best_idx],
        y_values[best_idx],
        s=180,
        marker="*",
        zorder=10
    )

    plt.annotate(
        f"{label}\nα={x_values[best_idx]:.2f}\n{y_values[best_idx]:.2f}%",
        xy=(x_values[best_idx], y_values[best_idx]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold"
    )


# ============================================================
# EXTRACT DATA
# ============================================================

alphas = get_values(densenet, "alpha")

# ----------------------------
# DenseNet Metrics
# ----------------------------

d_test_acc = get_values(
    densenet,
    "test_accuracy",
    percentage=True
)

d_val_acc = get_values(
    densenet,
    "best_val_accuracy",
    percentage=True
)

d_macro_f1 = get_values(
    densenet,
    "macro_f1",
    percentage=True
)

d_macro_precision = get_values(
    densenet,
    "macro_precision",
    percentage=True
)

d_macro_recall = get_values(
    densenet,
    "macro_recall",
    percentage=True
)

d_train_images = get_values(
    densenet,
    "train_images"
)

d_real_images = get_values(
    densenet,
    "real_images"
)

d_synth_images = get_values(
    densenet,
    "synth_images"
)


# ----------------------------
# EfficientNet Metrics
# ----------------------------

e_test_acc = get_values(
    efficientnet,
    "test_accuracy",
    percentage=True
)

e_val_acc = get_values(
    efficientnet,
    "best_val_accuracy",
    percentage=True
)

e_macro_f1 = get_values(
    efficientnet,
    "macro_f1",
    percentage=True
)

e_macro_precision = get_values(
    efficientnet,
    "macro_precision",
    percentage=True
)

e_macro_recall = get_values(
    efficientnet,
    "macro_recall",
    percentage=True
)

e_train_images = get_values(
    efficientnet,
    "train_images"
)

e_real_images = get_values(
    efficientnet,
    "real_images"
)

e_synth_images = get_values(
    efficientnet,
    "synth_images"
)


# ============================================================
# 01. TEST ACCURACY vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_test_acc,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_test_acc,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

annotate_best(alphas, d_test_acc, "DenseNet Best")
annotate_best(alphas, e_test_acc, "EfficientNet Best")

plt.xlabel("Alpha (Synthetic Data Mixing Level)", fontsize=13)
plt.ylabel("Test Accuracy (%)", fontsize=13)

plt.title(
    "Test Accuracy Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("01_test_accuracy_vs_alpha.png")


# ============================================================
# 02. BEST VALIDATION ACCURACY vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_val_acc,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_val_acc,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("Best Validation Accuracy (%)", fontsize=13)

plt.title(
    "Best Validation Accuracy Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("02_validation_accuracy_vs_alpha.png")


# ============================================================
# 03. MACRO F1-SCORE vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_macro_f1,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_macro_f1,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

annotate_best(alphas, d_macro_f1, "DenseNet Best")
annotate_best(alphas, e_macro_f1, "EfficientNet Best")

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("Macro F1-score (%)", fontsize=13)

plt.title(
    "Macro F1-score Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("03_macro_f1_vs_alpha.png")


# ============================================================
# 04. MACRO PRECISION vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_macro_precision,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_macro_precision,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("Macro Precision (%)", fontsize=13)

plt.title(
    "Macro Precision Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("04_macro_precision_vs_alpha.png")


# ============================================================
# 05. MACRO RECALL vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_macro_recall,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_macro_recall,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("Macro Recall (%)", fontsize=13)

plt.title(
    "Macro Recall Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("05_macro_recall_vs_alpha.png")


# ============================================================
# 06. COMBINED PERFORMANCE CURVES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ------------------------------------------------------------
# Test Accuracy
# ------------------------------------------------------------

axes[0].plot(
    alphas,
    d_test_acc,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

axes[0].plot(
    alphas,
    e_test_acc,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

axes[0].set_title("Test Accuracy", fontsize=15, fontweight="bold")
axes[0].set_xlabel("Alpha")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_xticks(alphas)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend()


# ------------------------------------------------------------
# Macro F1
# ------------------------------------------------------------

axes[1].plot(
    alphas,
    d_macro_f1,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

axes[1].plot(
    alphas,
    e_macro_f1,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

axes[1].set_title("Macro F1-score", fontsize=15, fontweight="bold")
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("F1-score (%)")
axes[1].set_xticks(alphas)
axes[1].grid(True, linestyle="--", alpha=0.4)


# ------------------------------------------------------------
# Best Validation Accuracy
# ------------------------------------------------------------

axes[2].plot(
    alphas,
    d_val_acc,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

axes[2].plot(
    alphas,
    e_val_acc,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

axes[2].set_title(
    "Best Validation Accuracy",
    fontsize=15,
    fontweight="bold"
)

axes[2].set_xlabel("Alpha")
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_xticks(alphas)
axes[2].grid(True, linestyle="--", alpha=0.4)

fig.suptitle(
    "Combined Architecture Performance Across Alpha Values",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_combined_performance_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 07. PERFORMANCE DIFFERENCE
# EfficientNet - DenseNet
# ============================================================

accuracy_difference = e_test_acc - d_test_acc
f1_difference = e_macro_f1 - d_macro_f1

x = np.arange(len(alphas))
width = 0.35

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    accuracy_difference,
    width,
    label="Test Accuracy Difference",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    f1_difference,
    width,
    label="Macro F1 Difference",
    edgecolor="black"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    x,
    [f"{a:.2f}" for a in alphas],
    fontsize=11
)

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("EfficientNet − DenseNet (%)", fontsize=13)

plt.title(
    "Performance Difference Between Architectures",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=11)

for bars in [bars1, bars2]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.05 if value >= 0 else -0.25),
            f"{value:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("07_architecture_difference.png")


# ============================================================
# 08. TRAINING DATASET SIZE vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_train_images,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_train_images,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("Total Training Images", fontsize=13)

plt.title(
    "Training Dataset Size Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

save_show("08_training_dataset_size.png")


# ============================================================
# 09. SYNTHETIC IMAGES ADDED vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_synth_images,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_synth_images,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha", fontsize=13)
plt.ylabel("Number of Synthetic Images", fontsize=13)

plt.title(
    "Synthetic Images Added Across Alpha Values",
    fontsize=17
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

save_show("09_synthetic_images_vs_alpha.png")


# ============================================================
# 10. SYNTHETIC DATA PERCENTAGE
# ============================================================

d_synth_percentage = (
    d_synth_images / d_train_images
) * 100

e_synth_percentage = (
    e_synth_images / e_train_images
) * 100

x = np.arange(len(alphas))
width = 0.35

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    d_synth_percentage,
    width,
    label="DenseNet121",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    e_synth_percentage,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

plt.xticks(
    x,
    [f"α={a:.2f}" for a in alphas]
)

plt.ylabel(
    "Synthetic Data in Training Set (%)",
    fontsize=13
)

plt.title(
    "Synthetic Data Contribution at Each Alpha Level",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.7,
            f"{height:.1f}%",
            ha="center",
            fontsize=9
        )

save_show("10_synthetic_percentage.png")


# ============================================================
# 11. TEST ACCURACY vs SYNTHETIC DATA %
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    d_synth_percentage,
    d_test_acc,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    e_synth_percentage,
    e_test_acc,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel(
    "Synthetic Images in Training Dataset (%)",
    fontsize=13
)

plt.ylabel(
    "Test Accuracy (%)",
    fontsize=13
)

plt.title(
    "Effect of Synthetic Data Percentage on Test Accuracy",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

save_show("11_accuracy_vs_synthetic_percentage.png")


# ============================================================
# 12. MACRO F1 vs SYNTHETIC DATA %
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    d_synth_percentage,
    d_macro_f1,
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    e_synth_percentage,
    e_macro_f1,
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel(
    "Synthetic Images in Training Dataset (%)",
    fontsize=13
)

plt.ylabel(
    "Macro F1-score (%)",
    fontsize=13
)

plt.title(
    "Effect of Synthetic Data Percentage on Macro F1-score",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

save_show("12_macro_f1_vs_synthetic_percentage.png")


# ============================================================
# 13. PERFORMANCE HEATMAP
# ============================================================

# Rows:
# DenseNet alpha values
# EfficientNet alpha values

heatmap_data = np.vstack([
    d_test_acc,
    e_test_acc,
    d_macro_f1,
    e_macro_f1
])

row_labels = [
    "DenseNet Test Accuracy",
    "EfficientNet Test Accuracy",
    "DenseNet Macro F1",
    "EfficientNet Macro F1"
]

plt.figure(figsize=(12, 6))

im = plt.imshow(
    heatmap_data,
    aspect="auto"
)

plt.colorbar(
    im,
    label="Performance (%)"
)

plt.xticks(
    np.arange(len(alphas)),
    [f"α={a:.2f}" for a in alphas]
)

plt.yticks(
    np.arange(len(row_labels)),
    row_labels
)

plt.title(
    "Architecture Performance Heatmap Across Alpha Values",
    fontsize=17,
    fontweight="bold"
)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):

        value = heatmap_data[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white"
            if value < heatmap_data.mean()
            else "black"
        )

save_show("13_performance_heatmap.png")


# ============================================================
# 14. BEST ALPHA CONFIGURATION SUMMARY
# ============================================================

# Find best alpha based on Test Accuracy

d_best_idx = np.argmax(d_test_acc)
e_best_idx = np.argmax(e_test_acc)

best_labels = [
    "DenseNet121",
    "EfficientNet-B3"
]

best_accuracy = [
    d_test_acc[d_best_idx],
    e_test_acc[e_best_idx]
]

best_f1 = [
    d_macro_f1[d_best_idx],
    e_macro_f1[e_best_idx]
]

best_alpha = [
    alphas[d_best_idx],
    alphas[e_best_idx]
]

x = np.arange(2)
width = 0.35

plt.figure(figsize=(10, 7))

bars1 = plt.bar(
    x - width / 2,
    best_accuracy,
    width,
    label="Test Accuracy",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    best_f1,
    width,
    label="Macro F1-score",
    edgecolor="black"
)

plt.xticks(
    x,
    [
        f"DenseNet121\nBest α={best_alpha[0]:.2f}",
        f"EfficientNet-B3\nBest α={best_alpha[1]:.2f}"
    ]
)

plt.ylabel("Performance (%)", fontsize=13)

plt.title(
    "Best Alpha Configuration for Each Architecture",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{height:.2f}%",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

save_show("14_best_alpha_summary.png")


# ============================================================
# FINAL NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("ALPHA SWEEP COMPARISON SUMMARY")
print("=" * 85)

print(
    f"\n{'Alpha':<10}"
    f"{'DenseNet Acc':>16}"
    f"{'EffNet Acc':>16}"
    f"{'DenseNet F1':>16}"
    f"{'EffNet F1':>16}"
)

print("-" * 85)

for i, alpha in enumerate(alphas):

    print(
        f"{alpha:<10.2f}"
        f"{d_test_acc[i]:>16.2f}"
        f"{e_test_acc[i]:>16.2f}"
        f"{d_macro_f1[i]:>16.2f}"
        f"{e_macro_f1[i]:>16.2f}"
    )


# ============================================================
# BEST CONFIGURATIONS
# ============================================================

print("\n" + "=" * 85)
print("BEST CONFIGURATIONS")
print("=" * 85)

print(
    f"\nDenseNet121:"
    f"\n  Best Alpha        : {alphas[d_best_idx]:.2f}"
    f"\n  Test Accuracy     : {d_test_acc[d_best_idx]:.2f}%"
    f"\n  Macro F1-score    : {d_macro_f1[d_best_idx]:.2f}%"
)

e_best_idx = np.argmax(e_test_acc)

print(
    f"\nEfficientNet-B3:"
    f"\n  Best Alpha        : {alphas[e_best_idx]:.2f}"
    f"\n  Test Accuracy     : {e_test_acc[e_best_idx]:.2f}%"
    f"\n  Macro F1-score    : {e_macro_f1[e_best_idx]:.2f}%"
)


# ============================================================
# BEST MACRO F1 CONFIGURATIONS
# ============================================================

d_best_f1_idx = np.argmax(d_macro_f1)
e_best_f1_idx = np.argmax(e_macro_f1)

print("\n" + "=" * 85)
print("BEST MACRO F1 CONFIGURATIONS")
print("=" * 85)

print(
    f"\nDenseNet121:"
    f"\n  Best Alpha        : {alphas[d_best_f1_idx]:.2f}"
    f"\n  Macro F1-score    : {d_macro_f1[d_best_f1_idx]:.2f}%"
)

print(
    f"\nEfficientNet-B3:"
    f"\n  Best Alpha        : {alphas[e_best_f1_idx]:.2f}"
    f"\n  Macro F1-score    : {e_macro_f1[e_best_f1_idx]:.2f}%"
)


# ============================================================
# OUTPUT LOCATION
# ============================================================

print("\n" + "=" * 85)
print("ALL GRAPHS SAVED SUCCESSFULLY")
print("=" * 85)

print(f"\nOutput Folder:\n{OUTPUT_DIR.resolve()}")
print("\nGenerated 14 comparison graphs.")
print("=" * 85)

