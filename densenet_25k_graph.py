# ============================================================
# COMPREHENSIVE COMPARISON:
# DenseNet121 vs EfficientNet-B3
# Pure Synthetic Dataset (25,000 Images)
#
# Graphs generated:
# 1. Overall Performance Comparison
# 2. Macro vs Weighted Metrics
# 3. Per-Class Precision Comparison
# 4. Per-Class Recall Comparison
# 5. Per-Class F1-score Comparison
# 6. Combined Per-Class F1 Comparison
# 7. Training Accuracy Curves
# 8. Validation Accuracy Curves
# 9. Training Loss Curves
# 10. Validation Loss Curves
# 11. Generalization Gap
# 12. Confusion Matrix – DenseNet121
# 13. Confusion Matrix – EfficientNet-B3
# 14. Normalized Confusion Matrix – DenseNet121
# 15. Normalized Confusion Matrix – EfficientNet-B3
# 16. Training Time Comparison
#
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DENSENET_FILE = "densenet121_pure_synthetic.json"
EFFICIENTNET_FILE = "results_efficientnet25k_pure_synthetic.json"

OUTPUT_DIR = Path("architecture_comparison_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# ============================================================
# LOAD JSON FILES
# ============================================================

with open(DENSENET_FILE, "r") as f:
    densenet = json.load(f)

with open(EFFICIENTNET_FILE, "r") as f:
    efficientnet = json.load(f)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_show(filename):
    """Save high-resolution figure and display it."""
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def get_percentage(value):
    """
    Convert metric to percentage.
    Handles both:
    - 0.93 -> 93%
    - 93.0 -> 93%
    """
    return value * 100 if value <= 1 else value


def get_history(model, key):
    return [epoch[key] for epoch in model["epoch_history"]]


def add_bar_labels(bars, fmt="{:.2f}%"):
    """Add numerical labels above bars."""
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )


# ============================================================
# BASIC INFORMATION
# ============================================================

MODEL_NAMES = ["DenseNet121", "EfficientNet-B3"]

CLASSES = [
    "mdoscc",
    "normal",
    "osmf",
    "pdoscc",
    "wdoscc"
]

CLASS_LABELS = [
    "MDOSCC",
    "NORMAL",
    "OSMF",
    "PDOSCC",
    "WDOSCC"
]


# ============================================================
# EXTRACT OVERALL METRICS
# ============================================================

d_report = densenet["per_class_report"]
e_report = efficientnet["per_class_report"]

d_test_acc = get_percentage(densenet["test_accuracy"])
e_test_acc = get_percentage(efficientnet["test_accuracy"])

d_val_acc = get_percentage(densenet["best_val_accuracy"])
e_val_acc = get_percentage(efficientnet["best_val_accuracy"])

d_macro_precision = d_report["macro avg"]["precision"] * 100
e_macro_precision = e_report["macro avg"]["precision"] * 100

d_macro_recall = d_report["macro avg"]["recall"] * 100
e_macro_recall = e_report["macro avg"]["recall"] * 100

d_macro_f1 = d_report["macro avg"]["f1-score"] * 100
e_macro_f1 = e_report["macro avg"]["f1-score"] * 100

d_weighted_precision = d_report["weighted avg"]["precision"] * 100
e_weighted_precision = e_report["weighted avg"]["precision"] * 100

d_weighted_recall = d_report["weighted avg"]["recall"] * 100
e_weighted_recall = e_report["weighted avg"]["recall"] * 100

d_weighted_f1 = d_report["weighted avg"]["f1-score"] * 100
e_weighted_f1 = e_report["weighted avg"]["f1-score"] * 100


# ============================================================
# 1. OVERALL PERFORMANCE COMPARISON
# ============================================================

metrics = [
    "Test\nAccuracy",
    "Best Val\nAccuracy",
    "Macro\nPrecision",
    "Macro\nRecall",
    "Macro\nF1-score",
    "Weighted\nF1-score"
]

d_values = [
    d_test_acc,
    d_val_acc,
    d_macro_precision,
    d_macro_recall,
    d_macro_f1,
    d_weighted_f1
]

e_values = [
    e_test_acc,
    e_val_acc,
    e_macro_precision,
    e_macro_recall,
    e_macro_f1,
    e_weighted_f1
]

x = np.arange(len(metrics))
width = 0.36

plt.figure(figsize=(15, 7))

bars1 = plt.bar(
    x - width / 2,
    d_values,
    width,
    label="DenseNet121",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    e_values,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

plt.xticks(x, metrics, fontsize=11)
plt.ylabel("Performance (%)", fontsize=13)
plt.title(
    "Overall Performance Comparison: DenseNet121 vs EfficientNet-B3",
    fontsize=17
)

plt.ylim(0, max(max(d_values), max(e_values)) + 15)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

add_bar_labels(bars1)
add_bar_labels(bars2)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("01_overall_performance_comparison.png")


# ============================================================
# 2. MACRO vs WEIGHTED METRICS
# ============================================================

metrics = ["Precision", "Recall", "F1-score"]

d_macro = [
    d_macro_precision,
    d_macro_recall,
    d_macro_f1
]

e_macro = [
    e_macro_precision,
    e_macro_recall,
    e_macro_f1
]

d_weighted = [
    d_weighted_precision,
    d_weighted_recall,
    d_weighted_f1
]

e_weighted = [
    e_weighted_precision,
    e_weighted_recall,
    e_weighted_f1
]

x = np.arange(len(metrics))
width = 0.2

plt.figure(figsize=(14, 7))

plt.bar(
    x - 1.5 * width,
    d_macro,
    width,
    label="DenseNet Macro",
    edgecolor="black"
)

plt.bar(
    x - 0.5 * width,
    d_weighted,
    width,
    label="DenseNet Weighted",
    edgecolor="black"
)

plt.bar(
    x + 0.5 * width,
    e_macro,
    width,
    label="EfficientNet Macro",
    edgecolor="black"
)

plt.bar(
    x + 1.5 * width,
    e_weighted,
    width,
    label="EfficientNet Weighted",
    edgecolor="black"
)

plt.xticks(x, metrics, fontsize=12)
plt.ylabel("Score (%)", fontsize=13)

plt.title(
    "Macro vs Weighted Classification Metrics",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=10)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("02_macro_weighted_comparison.png")


# ============================================================
# EXTRACT PER-CLASS METRICS
# ============================================================

d_precision = [d_report[c]["precision"] * 100 for c in CLASSES]
e_precision = [e_report[c]["precision"] * 100 for c in CLASSES]

d_recall = [d_report[c]["recall"] * 100 for c in CLASSES]
e_recall = [e_report[c]["recall"] * 100 for c in CLASSES]

d_f1 = [d_report[c]["f1-score"] * 100 for c in CLASSES]
e_f1 = [e_report[c]["f1-score"] * 100 for c in CLASSES]


# ============================================================
# 3. PER-CLASS PRECISION COMPARISON
# ============================================================

x = np.arange(len(CLASSES))
width = 0.36

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    d_precision,
    width,
    label="DenseNet121",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    e_precision,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

plt.xticks(x, CLASS_LABELS, fontsize=12)
plt.ylabel("Precision (%)", fontsize=13)

plt.title(
    "Per-Class Precision Comparison",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

add_bar_labels(bars1)
add_bar_labels(bars2)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("03_per_class_precision.png")


# ============================================================
# 4. PER-CLASS RECALL COMPARISON
# ============================================================

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    d_recall,
    width,
    label="DenseNet121",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    e_recall,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

plt.xticks(x, CLASS_LABELS, fontsize=12)
plt.ylabel("Recall (%)", fontsize=13)

plt.title(
    "Per-Class Recall Comparison",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

add_bar_labels(bars1)
add_bar_labels(bars2)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("04_per_class_recall.png")


# ============================================================
# 5. PER-CLASS F1-SCORE COMPARISON
# ============================================================

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    d_f1,
    width,
    label="DenseNet121",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width / 2,
    e_f1,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

plt.xticks(x, CLASS_LABELS, fontsize=12)
plt.ylabel("F1-score (%)", fontsize=13)

plt.title(
    "Per-Class F1-score Comparison",
    fontsize=17
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

add_bar_labels(bars1)
add_bar_labels(bars2)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("05_per_class_f1_score.png")


# ============================================================
# 6. COMBINED PER-CLASS PERFORMANCE
# Precision + Recall + F1 for BOTH architectures
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Precision
axes[0].bar(
    x - width / 2,
    d_precision,
    width,
    label="DenseNet121",
    edgecolor="black"
)

axes[0].bar(
    x + width / 2,
    e_precision,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

axes[0].set_title("Precision", fontsize=15, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_LABELS, rotation=25)
axes[0].set_ylabel("Score (%)")
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].legend()


# Recall
axes[1].bar(
    x - width / 2,
    d_recall,
    width,
    label="DenseNet121",
    edgecolor="black"
)

axes[1].bar(
    x + width / 2,
    e_recall,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

axes[1].set_title("Recall", fontsize=15, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_LABELS, rotation=25)
axes[1].set_ylabel("Score (%)")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)


# F1-score
axes[2].bar(
    x - width / 2,
    d_f1,
    width,
    label="DenseNet121",
    edgecolor="black"
)

axes[2].bar(
    x + width / 2,
    e_f1,
    width,
    label="EfficientNet-B3",
    edgecolor="black"
)

axes[2].set_title("F1-score", fontsize=15, fontweight="bold")
axes[2].set_xticks(x)
axes[2].set_xticklabels(CLASS_LABELS, rotation=25)
axes[2].set_ylabel("Score (%)")
axes[2].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle(
    "Combined Per-Class Performance Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_combined_per_class_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# TRAINING HISTORY
# ============================================================

epochs_d = get_history(densenet, "epoch")
epochs_e = get_history(efficientnet, "epoch")

d_train_acc = get_history(densenet, "train_acc")
e_train_acc = get_history(efficientnet, "train_acc")

d_val_acc_curve = get_history(densenet, "val_acc")
e_val_acc_curve = get_history(efficientnet, "val_acc")

d_train_loss = get_history(densenet, "train_loss")
e_train_loss = get_history(efficientnet, "train_loss")

d_val_loss = get_history(densenet, "val_loss")
e_val_loss = get_history(efficientnet, "val_loss")


# ============================================================
# 7. TRAINING ACCURACY CURVES
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    epochs_d,
    d_train_acc,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    epochs_e,
    e_train_acc,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Training Accuracy (%)", fontsize=13)

plt.title(
    "Training Accuracy Curve Comparison",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("07_training_accuracy_curves.png")


# ============================================================
# 8. VALIDATION ACCURACY CURVES
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    epochs_d,
    d_val_acc_curve,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    epochs_e,
    e_val_acc_curve,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Validation Accuracy (%)", fontsize=13)

plt.title(
    "Validation Accuracy Curve Comparison",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("08_validation_accuracy_curves.png")


# ============================================================
# 9. TRAINING LOSS CURVES
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    epochs_d,
    d_train_loss,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    epochs_e,
    e_train_loss,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Training Loss", fontsize=13)

plt.title(
    "Training Loss Curve Comparison",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("09_training_loss_curves.png")


# ============================================================
# 10. VALIDATION LOSS CURVES
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    epochs_d,
    d_val_loss,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    epochs_e,
    e_val_loss,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Validation Loss", fontsize=13)

plt.title(
    "Validation Loss Curve Comparison",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("10_validation_loss_curves.png")


# ============================================================
# 11. GENERALIZATION GAP
# Training Accuracy - Validation Accuracy
# ============================================================

d_gap = np.array(d_train_acc) - np.array(d_val_acc_curve)
e_gap = np.array(e_train_acc) - np.array(e_val_acc_curve)

plt.figure(figsize=(12, 7))

plt.plot(
    epochs_d,
    d_gap,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    epochs_e,
    e_gap,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Training − Validation Accuracy (%)", fontsize=13)

plt.title(
    "Generalization Gap Comparison",
    fontsize=17
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("11_generalization_gap.png")


# ============================================================
# CONFUSION MATRIX FUNCTION
# ============================================================

def plot_confusion_matrix(
    cm,
    labels,
    title,
    filename,
    normalized=False
):

    cm = np.array(cm, dtype=float)

    if normalized:
        cm = cm / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        cm,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(
        im,
        fraction=0.046,
        pad=0.04
    )

    plt.title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Predicted Class",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "True Class",
        fontsize=13,
        fontweight="bold"
    )

    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        np.arange(len(labels)),
        labels
    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            text_value = (
                f"{cm[i, j]:.1f}%"
                if normalized
                else f"{int(cm[i, j])}"
            )

            plt.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white"
                if cm[i, j] > threshold
                else "black"
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 12. DENSENET CONFUSION MATRIX
# ============================================================

plot_confusion_matrix(
    densenet["confusion_matrix"],
    CLASS_LABELS,
    "Confusion Matrix – DenseNet121",
    "12_densenet_confusion_matrix.png"
)


# ============================================================
# 13. EFFICIENTNET CONFUSION MATRIX
# ============================================================

plot_confusion_matrix(
    efficientnet["confusion_matrix"],
    CLASS_LABELS,
    "Confusion Matrix – EfficientNet-B3",
    "13_efficientnet_confusion_matrix.png"
)


# ============================================================
# 14. NORMALIZED DENSENET CONFUSION MATRIX
# ============================================================

plot_confusion_matrix(
    densenet["confusion_matrix"],
    CLASS_LABELS,
    "Normalized Confusion Matrix (%) – DenseNet121",
    "14_normalized_densenet_confusion_matrix.png",
    normalized=True
)


# ============================================================
# 15. NORMALIZED EFFICIENTNET CONFUSION MATRIX
# ============================================================

plot_confusion_matrix(
    efficientnet["confusion_matrix"],
    CLASS_LABELS,
    "Normalized Confusion Matrix (%) – EfficientNet-B3",
    "15_normalized_efficientnet_confusion_matrix.png",
    normalized=True
)


# ============================================================
# 16. TRAINING TIME COMPARISON
# ============================================================

d_time = sum(get_history(densenet, "time_s")) / 60
e_time = sum(get_history(efficientnet, "time_s")) / 60

times = [d_time, e_time]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    MODEL_NAMES,
    times,
    edgecolor="black",
    linewidth=1
)

plt.ylabel(
    "Training Time (Minutes)",
    fontsize=13
)

plt.title(
    "Training Time Comparison",
    fontsize=17
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

for bar, value in zip(bars, times):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f} min",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

save_show("16_training_time_comparison.png")


# ============================================================
# FINAL NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("COMPREHENSIVE ARCHITECTURE COMPARISON")
print("=" * 75)

print(f"\n{'Metric':<30} {'DenseNet121':>18} {'EfficientNet-B3':>20}")
print("-" * 75)

summary = [
    ("Test Accuracy (%)", d_test_acc, e_test_acc),
    ("Best Validation Accuracy (%)", d_val_acc, e_val_acc),
    ("Macro Precision (%)", d_macro_precision, e_macro_precision),
    ("Macro Recall (%)", d_macro_recall, e_macro_recall),
    ("Macro F1-score (%)", d_macro_f1, e_macro_f1),
    ("Weighted Precision (%)", d_weighted_precision, e_weighted_precision),
    ("Weighted Recall (%)", d_weighted_recall, e_weighted_recall),
    ("Weighted F1-score (%)", d_weighted_f1, e_weighted_f1),
    ("Training Time (minutes)", d_time, e_time),
]

for metric, dense_value, eff_value in summary:
    print(
        f"{metric:<30} "
        f"{dense_value:>18.2f} "
        f"{eff_value:>20.2f}"
    )


# ============================================================
# BEST MODEL FOR EACH OVERALL METRIC
# ============================================================

print("\n" + "=" * 75)
print("BEST MODEL BY METRIC")
print("=" * 75)

for metric, dense_value, eff_value in summary[:-1]:

    if dense_value > eff_value:
        winner = "DenseNet121"
        difference = dense_value - eff_value
    else:
        winner = "EfficientNet-B3"
        difference = eff_value - dense_value

    print(
        f"{metric:<30} "
        f"→ {winner} "
        f"(difference: {difference:.2f})"
    )


print("\n" + "=" * 75)
print("ALL GRAPHS SAVED TO:")
print(OUTPUT_DIR.resolve())
print("=" * 75)