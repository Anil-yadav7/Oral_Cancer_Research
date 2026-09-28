# Conditional StyleGAN2-ADA for Synthetic Oral Histopathology Image Generation

> **A Pure-Synthetic Downstream Classification Evaluation on the ORCHID Dataset**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_RTX_5090-76b900?logo=nvidia)](https://www.nvidia.com/)
[![Paper](https://img.shields.io/badge/Paper-IEEE_Format-blue)](paper/IEEE_Paper.pdf)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation & Problem Statement](#-motivation--problem-statement)
- [Dataset: ORCHID](#-dataset-orchid)
- [Methodology](#-methodology)
- [Repository Structure](#-repository-structure)
- [Experiments](#-experiments)
- [Key Results](#-key-results)
- [Key Findings & Takeaways](#-key-findings--takeaways)
- [Authors](#-authors)
- [References](#-references)

---

## 🔬 Overview

This repository contains the full research code, experimental scripts, result logs, and published paper for a study that **critically evaluates the clinical utility of GAN-generated histopathology images** for oral cancer diagnosis.

We trained a **class-conditional StyleGAN2-ADA** model on the ORCHID oral histopathology benchmark and then performed rigorous, multi-stage downstream classification experiments using **DenseNet-121** and **EfficientNet-B3** to answer a fundamental medical AI question:

> *Does high visual and statistical fidelity (low FID score) in synthetic histopathology images actually translate into clinically meaningful diagnostic utility?*

**Short answer: No — not in isolation.**

Our experiments reveal a profound gap between superficial visual realism and genuine biological diagnostic value, establishing clear operational boundaries for deploying synthetic data in clinical AI pipelines.

---

## 🎯 Motivation & Problem Statement

**Oral Squamous Cell Carcinoma (OSCC)** and its premalignant precursor **Oral Submucous Fibrosis (OSF)** are leading cancer-related diseases in South Asia, strongly linked to betel quid, gutka, and smokeless tobacco. Despite the enormous disease burden:

- 🌍 **Geographic bias**: >50% of global clinical AI datasets originate from Western or East Asian centres, creating domain generalization failures when deployed in Indian clinical settings.
- ⚖️ **Class imbalance**: The ORCHID dataset reflects real clinical intake, with a ~2.67:1 imbalance ratio — minority classes (Normal tissue, PDOSCC) are severely underrepresented.
- 📏 **Evaluation blind spots**: Standard generative metrics like Fréchet Inception Distance (FID) operate on general-purpose ImageNet features, capturing macroscopic textures while remaining **blind to fine microscopic markers** required for histological grading.

Previous approaches to imbalance either discarded majority-class data, restricted scope to binary classification, or applied complex loss reweighting — all treating symptoms rather than the core biological data deficit.

This research directly addresses the data deficit using **generative modelling** and rigorously stress-tests whether synthetic images can genuinely substitute or supplement real clinical biopsies.

---

## 🗂 Dataset: ORCHID

The **ORCHID (Oral Cancer Histology Image Database)** is a multi-center, open-access benchmark:

| Property | Detail |
|---|---|
| Raw images | ~300,000 H&E-stained patches at 1000× magnification |
| Curated benchmark | **14,705** images |
| Diagnostic classes | 5 categories |
| Class imbalance ratio | ~2.67:1 |

### Diagnostic Categories

| Class | Description |
|---|---|
| `Normal` | Healthy oral mucosa |
| `OSMF` | Oral Submucous Fibrosis (premalignant) |
| `WDOSCC` | Well-Differentiated Oral Squamous Cell Carcinoma |
| `MDOSCC` | Moderately Differentiated Oral Squamous Cell Carcinoma |
| `PDOSCC` | Poorly Differentiated Oral Squamous Cell Carcinoma (aggressive) |

> WDOSCC and MDOSCC dominate the distribution, while Normal tissue and PDOSCC remain severely underrepresented — making them the most critical classes for imbalance mitigation.

---

## ⚙️ Methodology

### Phase 1 — GAN Training: Conditional StyleGAN2-ADA

| Parameter | Value |
|---|---|
| Architecture | StyleGAN2-ADA (class-conditional) |
| Pre-training (transfer) | BreCaHAD breast cancer weights (histopathological priors) |
| GPU | NVIDIA GeForce RTX 5090 |
| Training duration | 3,000 kimg |
| Dataset augmentation | Horizontal flips |
| ADA target threshold | 0.6 |
| Evaluation metric | FID-50k (50,000-sample Fréchet Inception Distance) |

**FID Progression:**

The model was tracked across 16 checkpoints (every 200 kimg). Starting from a transfer baseline FID of **131.19**, the model converged to a minimum FID of **6.98** at the **2,800 kimg** checkpoint.

| Checkpoint | FID-50k |
|---|---|
| 0 kimg (transfer init) | 131.19 |
| 400 kimg | 11.87 |
| 1,200 kimg | 8.58 |
| 2,400 kimg | 8.07 |
| **2,800 kimg (best)** | **6.98** |
| 3,000 kimg | 7.20 |

Note: An instability spike to **FID = 116.75** occurred at the 2,000 kimg checkpoint before recovery.

**Precision & Recall (Late-stage checkpoints):**

| Checkpoint | Precision (Fidelity) | Recall (Diversity) |
|---|---|---|
| 2,200 kimg | 0.6594 | 0.4249 |
| 2,400 kimg | 0.6933 | 0.4300 |
| **2,800 kimg** | **0.6825** | **0.4332** |
| 3,000 kimg | 0.6706 | 0.4414 |

The **2,800 kimg checkpoint** offered the best fidelity-diversity trade-off and was used to generate a **bank of 25,000 synthetic patches** (5,000 per class) for all downstream experiments.

---

### Phase 2 — Downstream Classification

Two architectures were benchmarked to ensure findings were independent of architectural bias:

| Backbone | Mechanism | Dropout | Head |
|---|---|---|---|
| **DenseNet-121** | Dense feature concatenation across layers | 0.4 | Linear → 5 classes |
| **EfficientNet-B3** | Adaptive compound scaling + channel attenuation | 0.4 | Linear → 5 classes |

**Shared Hyperparameters (matched for fair comparison):**

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Backbone LR | 1e-4 |
| Head LR | 1e-3 |
| Weight decay | 1e-2 (AdamW) |
| Label smoothing | 0.1 |
| Gradient clipping | max norm 1.0 |
| LR schedule | OneCycleLR (pct_start=0.1) |
| Precision | AMP (PyTorch automatic mixed precision) |

---

### The Alpha (α) Mixing Formulation

To calibrate synthetic augmentation, a mixing parameter **α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}** was defined:

```
S_c = floor( α × (T_target − N_c) )
```

Where:
- `S_c` = synthetic patches added to class *c*
- `α` = fraction of the class deficit filled by generated images
- `N_c` = original number of real images in class *c*
- `T_target` = target class size

> **Example**: Class with N_c = 1,500 real images, T_target = 5,000 → deficit = 3,500.
> At α = 0.50: 1,750 synthetic images added → total = 3,250.

---

## 📁 Repository Structure

```
ORCHID/
│
├── README.md                             ← Project documentation
│
├── 🧬 GAN Training & Evaluation
│   ├── GAN_fid_draw.py                   ← FID score + Precision/Recall plotting
│   ├── GAN FIDS_scores.json              ← Raw FID scores from training logs (16 checkpoints)
│   ├── GAN_FID_Curve.png                 ← FID curve visualization
│   └── LOGS for FID .txt                 ← Full training logs (~1MB)
│
├── 🔬 Experiment 1 (Pure-Synthetic Visualization)
│   └── exp1.py                           ← Graphs: generalization gap, confusion matrices
│
├── 🔬 Experiment 2 (Pure-Synthetic Training)
│   ├── densenet121_pure_synthetic.py     ← DenseNet training script (Kaggle GPU)
│   ├── densenet121_pure_synthetic.json   ← Results: 9.17% test accuracy
│   ├── efficient_net_pure_synthetic.py   ← EfficientNet training script
│   └── results_efficientnet25k_pure_synthetic.json ← Results: 9.36% test accuracy
│
├── 🔬 Experiment 3 (Controlled α-Balancing Sweep)
│   ├── per_alpha_effiecienet.py          ← EfficientNet-B3 alpha sweep training
│   ├── per_alpha_efficientnet.json       ← Per-alpha results (EfficientNet)
│   ├── peralpha_densenet.py              ← DenseNet-121 alpha sweep training
│   ├── per_alpha_densenet.json           ← Per-alpha results (DenseNet)
│   └── peralphagraph.py                  ← Visualization for alpha ablation
│
├── 🔬 Experiment 4 (Aggressive Dataset Expansion — 5k per class)
│   ├── efficient_net_5k.py               ← EfficientNet 5k expansion training
│   ├── densenet_121_per_class_5k.py      ← DenseNet 5k expansion training
│   ├── efficienet_net_per_class_5k.json  ← EfficientNet 5k results
│   ├── densenet_121_perclass_5k.json     ← DenseNet 5k results
│   ├── desnet_25k_graph.py               ← Graph generator for Exp 4
│   ├── exp4.py                           ← Dilution curve & F1 heatmap visualization
│   └── perclass5k.py                     ← Combined analysis script
│
├── 📊 Visualization & Utility
│   ├── mk.py                             ← FID progression & precision-recall bar charts
│   └── cachecing-of-25k-pure-synthetic-images.ipynb
│                                         ← Kaggle notebook: cache synthetic images to RAM
│
├── 🖼 Images & Graphs
│   ├── images/
│   │   ├── fid_score_progression.png
│   │   ├── precision_recall_comparison.png
│   │   ├── exp2_graphs/                  ← Pure-synthetic experiment visualizations
│   │   ├── exp3_graphs/                  ← Controlled balancing visualizations
│   │   └── exp4_graphs/                  ← Dilution threshold visualizations
│   └── images.zip
│
└── 📝 Paper
    ├── paper_full.txt                    ← Full paper in plain text
    ├── IEEE_Paper.pdf                    ← Published IEEE-format paper
    ├── IEEE_Paper.docx / IEEE_Format_Paper.docx
    ├── IEEE_Paper.html                   ← HTML version
    ├── create_ieee_docs.py               ← Script to generate IEEE-format Word/PDF
    └── generate_html_and_pdf.py          ← Script to render HTML + PDF
```

---

## 🧪 Experiments

### Experiment 1 — Pure-Synthetic Stress Test (Visualization)
**Script**: [`exp1.py`](exp1.py)

Generates all visualizations for the zero-real training protocol:
- Training accuracy and loss curves for DenseNet-121 and EfficientNet-B3
- Generalization gap plots (train vs. real validation accuracy)
- Overall performance comparison bar charts
- Normalized confusion matrices for both architectures

Outputs saved to `exp2_graphs/`.

---

### Experiment 2 — Pure-Synthetic Downstream Classification
**Scripts**: [`densenet121_pure_synthetic.py`](densenet121_pure_synthetic.py), [`efficient_net_pure_synthetic.py`](efficient_net_pure_synthetic.py)

Classifiers trained **exclusively on 25,000 synthetic patches** (5,000 per class, 100% synthetic, 0 real images). Tested on an independent holdout set of real clinical biopsies.

- Synthetic cache loaded directly into RAM via NumPy arrays for maximum I/O speed.
- Validation performed on real patient data throughout training to monitor generalization.

---

### Experiment 3 — Controlled Balancing Sweep (α-Ablation)
**Scripts**: [`per_alpha_effiecienet.py`](per_alpha_effiecienet.py), [`peralpha_densenet.py`](peralpha_densenet.py)

Minority classes are supplemented up to the majority-class ceiling. `T_target` = max real majority class size.
Dataset grows from **10,228 → 13,950** images at α = 1.0.
α values tested: `{0.0, 0.25, 0.50, 0.75, 1.0}`

---

### Experiment 4 — Aggressive Dataset Expansion (5k per class)
**Scripts**: [`efficient_net_5k.py`](efficient_net_5k.py), [`densenet_121_per_class_5k.py`](densenet_121_per_class_5k.py)

Every class is artificially expanded to a ceiling of 5,000 images. At α = 1.0, the total dataset swells to **25,000 images** (synthetic share ≈ 59%).

---

## 📊 Key Results

### Experiment 2: Pure-Synthetic Stress Test

| Metric | DenseNet-121 | EfficientNet-B3 |
|---|---|---|
| Training Accuracy (epoch 25) | **99.34%** | **99.24%** |
| Real-World Test Accuracy | **9.17%** | **9.36%** |
| Macro F1-Score | **8.17%** | **8.74%** |

> ⚠️ **~90% collapse** in real-world performance despite near-perfect training accuracy — the hallmark of synthetic manifold overfit.

**Class-level failure analysis:**
- **OSMF**: DenseNet misclassified **86.6%** and EfficientNet **94.2%** of real OSMF biopsies as Normal — the GAN captured general eosinophilic appearance but failed to encode dense bundled collagen fibrillogenesis and progressive stromal hyalinization.
- **PDOSCC**: EfficientNet misclassified **74.9%** of aggressive PDOSCC as WDOSCC — the GAN replaced discrete cellular phenotypes with smoothed malignant textures, removing nuclear-to-cytoplasmic ratio markers and keratin pearls.

---

### Experiment 3: Controlled Balancing (α-Ablation, T_target = majority class ceiling)

| α | Synth Images | DenseNet-121 Test Acc | EfficientNet-B3 Test Acc |
|---|---|---|---|
| 0.00 (baseline, real only) | 0 | 96.80% | 96.29% |
| 0.25 | 931 | 95.73% | 96.11% |
| 0.50 | 1,862 | 95.92% | 96.36% |
| 0.75 | 2,791 | 95.98% | 96.36% |
| 1.00 (full balance) | 3,722 | 96.11% | **96.42%** |

- **DenseNet-121**: Sensitive to synthetic noise due to dense concatenation; best at real-only baseline.
- **EfficientNet-B3**: Compound scaling suppresses generative artifacts; slightly **improves** from 96.29% → 96.42% at full minority balance — confirming bounded synthetic augmentation as an effective regularizer.

---

### Experiment 4: Aggressive Expansion (T_target = 5,000 per class)

| α | Synth Share | DenseNet-121 Test Acc | EfficientNet-B3 Test Acc |
|---|---|---|---|
| 0.00 (baseline) | 0% | 95.67% | 96.98% |
| 0.25 | ~15% | 94.66% | 96.11% |
| 0.50 | ~30% | 93.91% | 96.17% |
| 0.75 | ~45% | 93.03% | 96.11% |
| 1.00 (max saturation) | ~59% | 93.22% | 96.04% |

- **DenseNet-121**: Substantial drop of **−2.45 pp** (95.67% → 93.22%)
- **EfficientNet-B3**: Greater robustness but still drops **−0.94 pp** (96.98% → 96.04%)

Synthetic saturation at ~59% corrupts discriminative decision boundaries, with widespread per-class F1 degradation across all five diagnostic categories.

---

## 💡 Key Findings & Takeaways

| Finding | Implication |
|---|---|
| FID = 6.98 does **not** imply clinical utility | Low FID captures macroscopic H&E staining and stromal textures, not fine cytological markers |
| Pure-synthetic training → ~90% test collapse | Synthetic data cannot replace real biopsies for training clinical-grade classifiers |
| Bounded augmentation (α ≤ 1.0, T_target = majority ceiling) → stable or slight improvement | Synthetic data works as a **regularizer** for class imbalance when constrained to authentic class limits |
| Aggressive expansion (~59% synthetic) → performance decline | Over-saturation dilutes genuine biological signals and corrupts decision boundaries |
| EfficientNet-B3 > DenseNet-121 for synthetic robustness | Compound scaling + channel attenuation better suppresses generative artifacts than dense concatenation |

### Operational Guidelines for Clinical AI

```
✅  Use synthetic data conservatively to balance minority classes
    (supplement up to the majority class ceiling)

⚠️  Always validate synthetic data with downstream clinical stress tests
    — do NOT rely solely on FID or other visual fidelity metrics

❌  Do NOT use synthetic data as a standalone replacement for real patient biopsies

❌  Avoid aggressive synthetic expansion beyond real data volume
    — feature dilution actively impairs diagnostic accuracy
```

---

## 🚀 Setup & Running the Code

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn scikit-learn tqdm Pillow
```

### Training Scripts (Kaggle GPU environment)

The training scripts are designed to run on Kaggle with the following dataset paths:

```python
VAL_ROOT  = '/kaggle/input/datasets/anil701/orchid-official-validation-test/val/val'
TEST_ROOT = '/kaggle/input/datasets/anil701/orchid-official-validation-test/test/test'
CACHE_DIR = '/kaggle/input/notebooks/chakkilalaanilkumar/final-classification/cache'
```

### Caching Synthetic Images

Before running classification, cache the 25k synthetic images into numpy arrays for fast I/O:

```bash
# Run on Kaggle:
# cachecing-of-25k-pure-synthetic-images.ipynb
# Produces: synth_images.npy and synth_labels.npy
```

### Generating Visualizations (Local)

All visualization scripts can be run locally without GPU:

```bash
# GAN FID curve and Precision/Recall plots
python mk.py
python GAN_fid_draw.py

# Experiment 2 (Pure-Synthetic) graphs → saved in exp2_graphs/
python exp1.py

# Experiment 4 (Dilution threshold) graphs → saved in exp4_graphs/
python exp4.py

# Alpha ablation graphs
python peralphagraph.py
```

---

## 👥 Authors

| Author |
|---|
| **Chakkilala Anil Kumar** |
| **Malladi Nagasri** |
| **Bandaru Keerthana** |

📄 **Full Paper**: [`paper/IEEE_Paper.pdf`](paper/IEEE_Paper.pdf)

---

## 📚 References

1. G. Huang et al., "Densely connected convolutional networks," *CVPR*, 2017.
2. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for CNNs," *ICML*, 2019.
3. L. Oakden-Rayner et al., "Hidden stratification causes clinically meaningful failures in ML for medical imaging," *ACM CHIL*, 2020.
4. R. Chaudhary et al., "ORCHID: A multi-center open-access oral cancer histology image database," *Sci. Data*, 2024.
5. S. Chakrabarty et al., "YOLO26/CNN benchmarking on the ORCHID oral cancer dataset," Preprint, 2026.
6. R. Chaudhary et al., "OralPatho framework on whole slide images," *medRxiv*, 2023.
7. A. Kumar et al., "EfficientNet-B0 on balanced 3k ORCHID subset," *IEEE Trans. Med. Imaging*, 2026.
8. Y. Ren et al., "Computer-aided classification study for OSCC," *Front. Oncol.*, 2026.
9. A. Tasnim et al., "Histo-AdaptiveViT: Long-tailed classification for histopathology," *ISBI*, 2026.
10. I. Goodfellow et al., "Generative adversarial nets," *NeurIPS*, 2014.
11. T. Karras et al., "Training generative adversarial networks with limited data," *NeurIPS*, 2020.
12. J. Dee et al., "Domain adaptation with StyleGAN2 on thyroid histopathology," *PLOS ONE*, 2024.
13. M. Heusel et al., "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," *NeurIPS*, 2017.
14. M. Chong and D. Forsyth, "Effectively unbiased FID and inception score and where to find them," *CVPR*, 2020.
15. Y. Xue et al., "Selective synthetic augmentation with HistoGAN," *Med. Image Anal.*, 2021.
16. M. Frid-Adar et al., "GAN-based synthetic medical image augmentation for increased CNN performance," *Neurocomputing*, 2018.

---

*This research highlights that the **clinical deception of purely visual realism** is a real and measurable risk in medical AI. Synthetic data must be a cautiously bounded complement to — never a replacement for — real patient tissue.*
