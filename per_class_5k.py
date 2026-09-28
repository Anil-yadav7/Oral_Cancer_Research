# ============================================================
# COMPREHENSIVE ALPHA-SWEEP VISUALIZATION
# DenseNet121 vs EfficientNet-B3
#
# Uses the uploaded experiment files containing:
# - Alpha-wise experiment results
# - Test Accuracy
# - Best Validation Accuracy
# - Macro / Weighted F1
# - Per-class Precision, Recall and F1-score
# - Training dataset composition
# - Training time
# - Confusion matrices
#
# Generates:
#
# 01. Test Accuracy vs Alpha
# 02. Validation Accuracy vs Alpha
# 03. Macro F1 vs Alpha
# 04. Weighted F1 vs Alpha
# 05. Combined Overall Performance
# 06. Precision / Recall / F1 vs Alpha
# 07. DenseNet vs EfficientNet Performance Gap
# 08. Training Dataset Size vs Alpha
# 09. Synthetic Images vs Alpha
# 10. Synthetic Percentage vs Alpha
# 11. Accuracy vs Synthetic Percentage
# 12. Macro F1 vs Synthetic Percentage
# 13. Training Time vs Alpha
# 14. Accuracy vs Training Time
# 15. Architecture Performance Heatmap
# 16. Best Configuration Comparison
# 17. Per-Class F1 Heatmap Across Alpha
# 18. Per-Class Precision Heatmap Across Alpha
# 19. Per-Class Recall Heatmap Across Alpha
# 20. Best Alpha Confusion Matrices
#
# ============================================================


import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DENSENET_FILE = "densenet_121_perclass_5k.json"
EFFICIENTNET_FILE = "efficienet_net_per_class_5k.json"

OUTPUT_DIR = Path("comprehensive_alpha_sweep_graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


# ============================================================
# LOAD FILES
# ============================================================

def load_json_objects(file_path):
    """
    Handles files containing multiple JSON objects.
    """

    with open(file_path, "r") as f:
        content = f.read().strip()

    decoder = json.JSONDecoder()
    objects = []

    index = 0

    while index < len(content):

        # Skip whitespace
        while index < len(content) and content[index].isspace():
            index += 1

        if index >= len(content):
            break

        try:
            obj, end_index = decoder.raw_decode(content, index)
            objects.append(obj)
            index = end_index

        except json.JSONDecodeError:
            index += 1

    return objects


densenet_objects = load_json_objects(DENSENET_FILE)
efficientnet_objects = load_json_objects(EFFICIENTNET_FILE)


# ============================================================
# EXTRACT EXPERIMENT RUNS
# ============================================================

def extract_runs(objects):

    runs = []

    for obj in objects:

        # Individual experiment dictionary
        if isinstance(obj, dict) and "alpha" in obj and "architecture" in obj:
            runs.append(obj)

        # Summary list
        elif isinstance(obj, list):

            for item in obj:
                if isinstance(item, dict) and "alpha" in item:
                    runs.append(item)

    # Remove duplicate alpha entries
    unique_runs = {}

    for run in runs:

        alpha = run["alpha"]

        # Prefer detailed records containing per_class_report
        if alpha not in unique_runs:
            unique_runs[alpha] = run

        elif "per_class_report" in run:
            unique_runs[alpha] = run

    return sorted(
        unique_runs.values(),
        key=lambda x: x["alpha"]
    )


d_runs = extract_runs(densenet_objects)
e_runs = extract_runs(efficientnet_objects)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percentage(value):

    if value <= 1:
        return value * 100

    return value


def get_metric(run, metric):

    # Direct metric
    if metric in run:
        return percentage(run[metric])

    # Macro metrics from report
    if metric == "macro_precision":
        return run["per_class_report"]["macro avg"]["precision"] * 100

    if metric == "macro_recall":
        return run["per_class_report"]["macro avg"]["recall"] * 100

    if metric == "macro_f1":
        return run["per_class_report"]["macro avg"]["f1-score"] * 100

    if metric == "weighted_f1":
        return run["per_class_report"]["weighted avg"]["f1-score"] * 100

    raise KeyError(f"Metric not found: {metric}")


def get_image_count(run, key):

    possible_keys = {
        "real": ["real_images_used", "real_images"],
        "synthetic": ["synth_images_used", "synth_images"],
        "total": ["train_images_total", "total_train_images", "train_images"]
    }

    for possible_key in possible_keys[key]:

        if possible_key in run:
            return run[possible_key]

    return 0


def save_show(filename):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def annotate_best(x, y, name):

    idx = np.argmax(y)

    plt.scatter(
        x[idx],
        y[idx],
        s=180,
        marker="*",
        zorder=10
    )

    plt.annotate(
        f"{name}\nα={x[idx]:.2f}\n{y[idx]:.2f}%",
        xy=(x[idx], y[idx]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold"
    )


def add_labels(bars):

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )


# ============================================================
# EXTRACT ALPHAS
# ============================================================

d_alphas = np.array([r["alpha"] for r in d_runs])
e_alphas = np.array([r["alpha"] for r in e_runs])

alphas = sorted(set(d_alphas).intersection(set(e_alphas)))
alphas = np.array(alphas)


def get_run_by_alpha(runs, alpha):

    for run in runs:
        if run["alpha"] == alpha:
            return run


# ============================================================
# EXTRACT OVERALL METRICS
# ============================================================

d_test_acc = np.array([
    percentage(get_run_by_alpha(d_runs, a)["test_accuracy"])
    for a in alphas
])

e_test_acc = np.array([
    percentage(get_run_by_alpha(e_runs, a)["test_accuracy"])
    for a in alphas
])


d_val_acc = np.array([
    percentage(get_run_by_alpha(d_runs, a)["best_val_accuracy"])
    for a in alphas
])

e_val_acc = np.array([
    percentage(get_run_by_alpha(e_runs, a)["best_val_accuracy"])
    for a in alphas
])


d_macro_f1 = np.array([
    get_metric(get_run_by_alpha(d_runs, a), "macro_f1")
    for a in alphas
])

e_macro_f1 = np.array([
    get_metric(get_run_by_alpha(e_runs, a), "macro_f1")
    for a in alphas
])


d_weighted_f1 = np.array([
    get_metric(get_run_by_alpha(d_runs, a), "weighted_f1")
    for a in alphas
])

e_weighted_f1 = np.array([
    get_metric(get_run_by_alpha(e_runs, a), "weighted_f1")
    for a in alphas
])


d_macro_precision = np.array([
    get_metric(get_run_by_alpha(d_runs, a), "macro_precision")
    for a in alphas
])

e_macro_precision = np.array([
    get_metric(get_run_by_alpha(e_runs, a), "macro_precision")
    for a in alphas
])


d_macro_recall = np.array([
    get_metric(get_run_by_alpha(d_runs, a), "macro_recall")
    for a in alphas
])

e_macro_recall = np.array([
    get_metric(get_run_by_alpha(e_runs, a), "macro_recall")
    for a in alphas
])


# ============================================================
# DATASET COMPOSITION
# ============================================================

d_real = np.array([
    get_image_count(get_run_by_alpha(d_runs, a), "real")
    for a in alphas
])

d_synth = np.array([
    get_image_count(get_run_by_alpha(d_runs, a), "synthetic")
    for a in alphas
])

d_total = np.array([
    get_image_count(get_run_by_alpha(d_runs, a), "total")
    for a in alphas
])


e_real = np.array([
    get_image_count(get_run_by_alpha(e_runs, a), "real")
    for a in alphas
])

e_synth = np.array([
    get_image_count(get_run_by_alpha(e_runs, a), "synthetic")
    for a in alphas
])

e_total = np.array([
    get_image_count(get_run_by_alpha(e_runs, a), "total")
    for a in alphas
])


d_synth_pct = (d_synth / d_total) * 100
e_synth_pct = (e_synth / e_total) * 100


# ============================================================
# 01. TEST ACCURACY vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_test_acc,
    marker="o",
    linewidth=2.5,
    markersize=8,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_test_acc,
    marker="s",
    linewidth=2.5,
    markersize=8,
    label="EfficientNet-B3"
)

annotate_best(alphas, d_test_acc, "DenseNet Best")
annotate_best(alphas, e_test_acc, "EfficientNet Best")

plt.xlabel("Alpha (α)")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Test Accuracy Across Synthetic Data Levels"
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("01_test_accuracy_vs_alpha.png")


# ============================================================
# 02. VALIDATION ACCURACY vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_val_acc,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_val_acc,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha (α)")
plt.ylabel("Best Validation Accuracy (%)")

plt.title(
    "Best Validation Accuracy Across Alpha Values"
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("02_validation_accuracy_vs_alpha.png")


# ============================================================
# 03. MACRO F1 vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_macro_f1,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_macro_f1,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

annotate_best(alphas, d_macro_f1, "DenseNet Best")
annotate_best(alphas, e_macro_f1, "EfficientNet Best")

plt.xlabel("Alpha (α)")
plt.ylabel("Macro F1-score (%)")

plt.title(
    "Macro F1-score Across Alpha Values"
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("03_macro_f1_vs_alpha.png")


# ============================================================
# 04. WEIGHTED F1 vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_weighted_f1,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_weighted_f1,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha (α)")
plt.ylabel("Weighted F1-score (%)")

plt.title(
    "Weighted F1-score Across Alpha Values"
)

plt.xticks(alphas)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("04_weighted_f1_vs_alpha.png")


# ============================================================
# 05. COMBINED OVERALL PERFORMANCE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(19, 6))


# Test Accuracy
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

axes[0].set_title("Test Accuracy")
axes[0].set_xlabel("Alpha")
axes[0].set_ylabel("Accuracy (%)")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend()


# Validation Accuracy
axes[1].plot(
    alphas,
    d_val_acc,
    marker="o",
    linewidth=2.5
)

axes[1].plot(
    alphas,
    e_val_acc,
    marker="s",
    linewidth=2.5
)

axes[1].set_title("Best Validation Accuracy")
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True, linestyle="--", alpha=0.4)


# Macro F1
axes[2].plot(
    alphas,
    d_macro_f1,
    marker="o",
    linewidth=2.5
)

axes[2].plot(
    alphas,
    e_macro_f1,
    marker="s",
    linewidth=2.5
)

axes[2].set_title("Macro F1-score")
axes[2].set_xlabel("Alpha")
axes[2].set_ylabel("F1-score (%)")
axes[2].grid(True, linestyle="--", alpha=0.4)


fig.suptitle(
    "Overall Architecture Performance Across Alpha Values",
    fontsize=18,
    fontweight="bold"
)

save_show("05_combined_overall_performance.png")


# ============================================================
# 06. PRECISION / RECALL / F1 COMPARISON
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))


metrics_data = [
    (
        d_macro_precision,
        e_macro_precision,
        "Macro Precision"
    ),
    (
        d_macro_recall,
        e_macro_recall,
        "Macro Recall"
    ),
    (
        d_macro_f1,
        e_macro_f1,
        "Macro F1-score"
    )
]


for ax, (d_values, e_values, title) in zip(
    axes,
    metrics_data
):

    ax.plot(
        alphas,
        d_values,
        marker="o",
        linewidth=2.5,
        label="DenseNet121"
    )

    ax.plot(
        alphas,
        e_values,
        marker="s",
        linewidth=2.5,
        label="EfficientNet-B3"
    )

    ax.set_title(title)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Score (%)")
    ax.grid(True, linestyle="--", alpha=0.4)


axes[0].legend()

fig.suptitle(
    "Macro Classification Metrics Across Alpha Values",
    fontsize=18,
    fontweight="bold"
)

save_show("06_precision_recall_f1.png")


# ============================================================
# 07. PERFORMANCE GAP
# EfficientNet - DenseNet
# ============================================================

accuracy_gap = e_test_acc - d_test_acc
f1_gap = e_macro_f1 - d_macro_f1

x = np.arange(len(alphas))
width = 0.35

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    accuracy_gap,
    width,
    label="Test Accuracy Gap"
)

bars2 = plt.bar(
    x + width / 2,
    f1_gap,
    width,
    label="Macro F1 Gap"
)

plt.axhline(0, linewidth=1)

plt.xticks(
    x,
    [f"α={a:.2f}" for a in alphas]
)

plt.ylabel("EfficientNet − DenseNet (%)")

plt.title(
    "Performance Advantage of EfficientNet-B3"
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()

save_show("07_architecture_performance_gap.png")


# ============================================================
# 08. TRAINING DATASET SIZE vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_total,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_total,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha (α)")
plt.ylabel("Total Training Images")

plt.title(
    "Training Dataset Size Across Alpha Values"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("08_training_dataset_size.png")


# ============================================================
# 09. SYNTHETIC IMAGES vs ALPHA
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_synth,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_synth,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha (α)")
plt.ylabel("Synthetic Images Used")

plt.title(
    "Synthetic Image Contribution Across Alpha Values"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("09_synthetic_images_vs_alpha.png")


# ============================================================
# 10. SYNTHETIC DATA PERCENTAGE
# ============================================================

x = np.arange(len(alphas))
width = 0.36

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width / 2,
    d_synth_pct,
    width,
    label="DenseNet121"
)

bars2 = plt.bar(
    x + width / 2,
    e_synth_pct,
    width,
    label="EfficientNet-B3"
)

plt.xticks(
    x,
    [f"α={a:.2f}" for a in alphas]
)

plt.ylabel("Synthetic Data (%)")

plt.title(
    "Percentage of Synthetic Images in Training Data"
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()

save_show("10_synthetic_data_percentage.png")


# ============================================================
# 11. TEST ACCURACY vs SYNTHETIC DATA %
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    d_synth_pct,
    d_test_acc,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    e_synth_pct,
    e_test_acc,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Synthetic Images in Training Data (%)")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Effect of Synthetic Data on Test Accuracy"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("11_accuracy_vs_synthetic_percentage.png")


# ============================================================
# 12. MACRO F1 vs SYNTHETIC DATA %
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    d_synth_pct,
    d_macro_f1,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    e_synth_pct,
    e_macro_f1,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Synthetic Images in Training Data (%)")
plt.ylabel("Macro F1-score (%)")

plt.title(
    "Effect of Synthetic Data on Macro F1-score"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("12_macro_f1_vs_synthetic_percentage.png")


# ============================================================
# 13. TRAINING TIME vs ALPHA
# ============================================================

def get_training_time(run):

    if "total_time_s" in run:
        return run["total_time_s"] / 60

    if "epoch_history" in run:
        return sum(
            epoch["time_s"]
            for epoch in run["epoch_history"]
        ) / 60

    return 0


d_time = np.array([
    get_training_time(get_run_by_alpha(d_runs, a))
    for a in alphas
])

e_time = np.array([
    get_training_time(get_run_by_alpha(e_runs, a))
    for a in alphas
])


plt.figure(figsize=(12, 7))

plt.plot(
    alphas,
    d_time,
    marker="o",
    linewidth=2.5,
    label="DenseNet121"
)

plt.plot(
    alphas,
    e_time,
    marker="s",
    linewidth=2.5,
    label="EfficientNet-B3"
)

plt.xlabel("Alpha (α)")
plt.ylabel("Training Time (Minutes)")

plt.title(
    "Training Time Across Alpha Values"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("13_training_time_vs_alpha.png")


# ============================================================
# 14. ACCURACY vs TRAINING TIME
# ============================================================

plt.figure(figsize=(12, 7))

plt.scatter(
    d_time,
    d_test_acc,
    s=120,
    label="DenseNet121"
)

plt.scatter(
    e_time,
    e_test_acc,
    s=120,
    marker="s",
    label="EfficientNet-B3"
)

for i, alpha in enumerate(alphas):

    plt.annotate(
        f"α={alpha:.2f}",
        (d_time[i], d_test_acc[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

    plt.annotate(
        f"α={alpha:.2f}",
        (e_time[i], e_test_acc[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


plt.xlabel("Training Time (Minutes)")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Accuracy vs Computational Training Cost"
)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

save_show("14_accuracy_vs_training_time.png")


# ============================================================
# 15. ARCHITECTURE PERFORMANCE HEATMAP
# ============================================================

heatmap_data = np.vstack([

    d_test_acc,
    e_test_acc,

    d_val_acc,
    e_val_acc,

    d_macro_f1,
    e_macro_f1

])


row_labels = [

    "DenseNet Test Accuracy",
    "EfficientNet Test Accuracy",

    "DenseNet Validation Accuracy",
    "EfficientNet Validation Accuracy",

    "DenseNet Macro F1",
    "EfficientNet Macro F1"

]


plt.figure(figsize=(13, 7))

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
    "Performance Heatmap Across Alpha Values",
    fontsize=17
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
            fontsize=9,
            fontweight="bold"
        )


save_show("15_architecture_performance_heatmap.png")


# ============================================================
# 16. BEST CONFIGURATION COMPARISON
# ============================================================

d_best_idx = np.argmax(d_test_acc)
e_best_idx = np.argmax(e_test_acc)


models = [
    f"DenseNet121\nα={alphas[d_best_idx]:.2f}",
    f"EfficientNet-B3\nα={alphas[e_best_idx]:.2f}"
]


best_accuracy = [
    d_test_acc[d_best_idx],
    e_test_acc[e_best_idx]
]


best_f1 = [
    d_macro_f1[d_best_idx],
    e_macro_f1[e_best_idx]
]


x = np.arange(2)
width = 0.35


plt.figure(figsize=(11, 7))

bars1 = plt.bar(
    x - width / 2,
    best_accuracy,
    width,
    label="Test Accuracy"
)

bars2 = plt.bar(
    x + width / 2,
    best_f1,
    width,
    label="Macro F1-score"
)

plt.xticks(x, models)

plt.ylabel("Performance (%)")

plt.title(
    "Best Configuration of Each Architecture"
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()

add_labels(bars1)
add_labels(bars2)

save_show("16_best_configuration_comparison.png")


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

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


def extract_per_class_metric(runs, alpha_values, metric):

    matrix = []

    for alpha in alpha_values:

        run = get_run_by_alpha(runs, alpha)

        row = []

        for cls in CLASSES:

            row.append(
                run["per_class_report"][cls][metric] * 100
            )

        matrix.append(row)

    return np.array(matrix)


d_precision_matrix = extract_per_class_metric(
    d_runs,
    alphas,
    "precision"
)

e_precision_matrix = extract_per_class_metric(
    e_runs,
    alphas,
    "precision"
)


d_recall_matrix = extract_per_class_metric(
    d_runs,
    alphas,
    "recall"
)

e_recall_matrix = extract_per_class_metric(
    e_runs,
    alphas,
    "recall"
)


d_f1_matrix = extract_per_class_metric(
    d_runs,
    alphas,
    "f1-score"
)

e_f1_matrix = extract_per_class_metric(
    e_runs,
    alphas,
    "f1-score"
)


# ============================================================
# HEATMAP FUNCTION
# ============================================================

def plot_metric_heatmap(
    dense_matrix,
    efficient_matrix,
    metric_name,
    filename
):

    combined = np.vstack([
        dense_matrix,
        efficient_matrix
    ])


    row_labels = (
        [f"DenseNet α={a:.2f}" for a in alphas]
        +
        [f"EfficientNet α={a:.2f}" for a in alphas]
    )


    plt.figure(figsize=(12, 10))

    im = plt.imshow(
        combined,
        aspect="auto"
    )

    plt.colorbar(
        im,
        label=f"{metric_name} (%)"
    )

    plt.xticks(
        np.arange(len(CLASS_LABELS)),
        CLASS_LABELS,
        rotation=20
    )

    plt.yticks(
        np.arange(len(row_labels)),
        row_labels
    )


    plt.title(
        f"Per-Class {metric_name} Across Architectures and Alpha Values",
        fontsize=16,
        fontweight="bold"
    )


    for i in range(combined.shape[0]):

        for j in range(combined.shape[1]):

            value = combined[i, j]

            plt.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8
            )


    save_show(filename)


# ============================================================
# 17. PER-CLASS F1 HEATMAP
# ============================================================

plot_metric_heatmap(
    d_f1_matrix,
    e_f1_matrix,
    "F1-score",
    "17_per_class_f1_heatmap.png"
)


# ============================================================
# 18. PER-CLASS PRECISION HEATMAP
# ============================================================

plot_metric_heatmap(
    d_precision_matrix,
    e_precision_matrix,
    "Precision",
    "18_per_class_precision_heatmap.png"
)


# ============================================================
# 19. PER-CLASS RECALL HEATMAP
# ============================================================

plot_metric_heatmap(
    d_recall_matrix,
    e_recall_matrix,
    "Recall",
    "19_per_class_recall_heatmap.png"
)


# ============================================================
# 20. BEST ALPHA CONFUSION MATRICES
# ============================================================

def plot_confusion_matrix(
    cm,
    title,
    filename
):

    cm = np.array(cm)

    plt.figure(figsize=(9, 8))

    im = plt.imshow(
        cm,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(im)

    plt.xticks(
        np.arange(len(CLASS_LABELS)),
        CLASS_LABELS,
        rotation=35,
        ha="right"
    )

    plt.yticks(
        np.arange(len(CLASS_LABELS)),
        CLASS_LABELS
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )


    threshold = cm.max() / 2


    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white"
                if cm[i, j] > threshold
                else "black"
            )


    save_show(filename)


# DenseNet Best Configuration
best_dense_run = get_run_by_alpha(
    d_runs,
    alphas[d_best_idx]
)

plot_confusion_matrix(
    best_dense_run["confusion_matrix"],
    f"DenseNet121 Confusion Matrix – Best α={alphas[d_best_idx]:.2f}",
    "20_best_densenet_confusion_matrix.png"
)


# EfficientNet Best Configuration
best_efficient_run = get_run_by_alpha(
    e_runs,
    alphas[e_best_idx]
)

plot_confusion_matrix(
    best_efficient_run["confusion_matrix"],
    f"EfficientNet-B3 Confusion Matrix – Best α={alphas[e_best_idx]:.2f}",
    "21_best_efficientnet_confusion_matrix.png"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("COMPREHENSIVE ALPHA SWEEP RESULTS")
print("=" * 85)

print(
    f"\n{'Alpha':<10}"
    f"{'Dense Acc':>15}"
    f"{'EffNet Acc':>15}"
    f"{'Dense F1':>15}"
    f"{'EffNet F1':>15}"
)

print("-" * 85)


for i, alpha in enumerate(alphas):

    print(
        f"{alpha:<10.2f}"
        f"{d_test_acc[i]:>15.2f}"
        f"{e_test_acc[i]:>15.2f}"
        f"{d_macro_f1[i]:>15.2f}"
        f"{e_macro_f1[i]:>15.2f}"
    )


print("\n" + "=" * 85)
print("BEST TEST ACCURACY")
print("=" * 85)

print(
    f"\nDenseNet121:"
    f"\nAlpha              : {alphas[d_best_idx]:.2f}"
    f"\nTest Accuracy      : {d_test_acc[d_best_idx]:.2f}%"
    f"\nMacro F1-score     : {d_macro_f1[d_best_idx]:.2f}%"
)

print(
    f"\nEfficientNet-B3:"
    f"\nAlpha              : {alphas[e_best_idx]:.2f}"
    f"\nTest Accuracy      : {e_test_acc[e_best_idx]:.2f}%"
    f"\nMacro F1-score     : {e_macro_f1[e_best_idx]:.2f}%"
)


print("\n" + "=" * 85)
print("GRAPHS GENERATED SUCCESSFULLY")
print("=" * 85)

print(f"\nOutput folder:\n{OUTPUT_DIR.resolve()}")

print("\nTotal graphs generated: 21")
print("=" * 85)