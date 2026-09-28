import matplotlib.pyplot as plt
import numpy as np
import re

# ==========================================
# Data Extraction & Preparation
# ==========================================

# 1. FID Score Data[cite: 1]
# Extracted from the provided JSON training logs
fid_log_data = [
    {"fid50k_full": 131.19, "snapshot_pkl": "network-snapshot-000000.pkl"},
    {"fid50k_full": 16.41,  "snapshot_pkl": "network-snapshot-000200.pkl"},
    {"fid50k_full": 11.87,  "snapshot_pkl": "network-snapshot-000400.pkl"},
    {"fid50k_full": 13.38,  "snapshot_pkl": "network-snapshot-000600.pkl"},
    {"fid50k_full": 9.96,   "snapshot_pkl": "network-snapshot-000800.pkl"},
    {"fid50k_full": 9.60,   "snapshot_pkl": "network-snapshot-001000.pkl"},
    {"fid50k_full": 8.58,   "snapshot_pkl": "network-snapshot-001200.pkl"},
    {"fid50k_full": 10.27,  "snapshot_pkl": "network-snapshot-001400.pkl"},
    {"fid50k_full": 9.50,   "snapshot_pkl": "network-snapshot-001600.pkl"},
    {"fid50k_full": 8.34,   "snapshot_pkl": "network-snapshot-001800.pkl"},
    {"fid50k_full": 116.75, "snapshot_pkl": "network-snapshot-002000.pkl"},
    {"fid50k_full": 8.31,   "snapshot_pkl": "network-snapshot-002200.pkl"},
    {"fid50k_full": 8.07,   "snapshot_pkl": "network-snapshot-002400.pkl"},
    {"fid50k_full": 12.97,  "snapshot_pkl": "network-snapshot-002600.pkl"},
    {"fid50k_full": 6.98,   "snapshot_pkl": "network-snapshot-002800.pkl"},
    {"fid50k_full": 7.20,   "snapshot_pkl": "network-snapshot-003000.pkl"}
]

# Parse checkpoints and FID scores[cite: 1]
checkpoints = [int(re.search(r'\d+', entry["snapshot_pkl"]).group()) for entry in fid_log_data]
fid_scores = [entry["fid50k_full"] for entry in fid_log_data]

# 2. Precision & Recall Data
# Extracted from Screenshot 2026-08-27 at 8.23.11 PM.png
pr_checkpoints = ['2200', '2400', '2800', '3000']
precision = [0.6594, 0.6933, 0.6825, 0.6706]
recall = [0.4249, 0.4300, 0.4332, 0.4414]

# ==========================================
# Plot 1: FID Score Progression
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(checkpoints, fid_scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)

# Highlight the best FID score (global minimum)[cite: 1]
best_idx = np.argmin(fid_scores)
plt.plot(checkpoints[best_idx], fid_scores[best_idx], marker='*', color='darkorange', markersize=15, 
         label=f'Best FID: {fid_scores[best_idx]} at {checkpoints[best_idx]} kimg')

plt.title('FID Score Progression During GAN Training', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Training Snapshot (kimg)', fontsize=14, fontweight='bold')
plt.ylabel('FID Score ↓', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('fid_score_progression.png', dpi=300)
plt.show()

# ==========================================
# Plot 2: Precision vs. Recall Bar Chart
# ==========================================
x = np.arange(len(pr_checkpoints))
width = 0.35

plt.figure(figsize=(10, 6))
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, precision, width, label='Precision (Fidelity) ↑', color='#2ca02c')
rects2 = ax.bar(x + width/2, recall, width, label='Recall (Diversity) ↑', color='#9467bd')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Checkpoint (kimg)', fontsize=12, fontweight='bold')
ax.set_title('Generative Fidelity vs. Diversity Across Late-Stage Checkpoints', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(pr_checkpoints, fontsize=11)
ax.set_ylim(0, 0.8)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Attach text labels above bars
ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

fig.tight_layout()
plt.savefig('precision_recall_comparison.png', dpi=300)
plt.show()