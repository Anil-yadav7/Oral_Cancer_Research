import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DATA EXTRACTION
# ==========================================

# Data from JSON logs (FID Scores)
checkpoints_fid = [
    0, 200, 400, 600, 800, 1000, 1200, 1400, 
    1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000
]
fid_scores = [
    131.19, 16.41, 11.87, 13.38, 9.96, 9.60, 8.58, 10.27, 
    9.50, 8.34, 116.75, 8.31, 8.07, 12.97, 6.98, 7.20
]

# Data from the Table (Precision & Recall)
checkpoints_pr = ['2200', '2400', '2800', '3000']
precision = [0.6594, 0.6933, 0.6825, 0.6706]
recall = [0.4249, 0.4300, 0.4332, 0.4414]

# ==========================================
# 2. PLOTTING SETUP
# ==========================================

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
plt.style.use('seaborn-v0_8-whitegrid')

# --- Plot 1: FID Score Progression ---
ax1.plot(checkpoints_fid, fid_scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)

# Annotate the global minimum
min_fid_idx = np.argmin(fid_scores)
min_x = checkpoints_fid[min_fid_idx]
min_y = fid_scores[min_fid_idx]
ax1.annotate(f'Global Min\nFID: {min_y}', xy=(min_x, min_y), xytext=(min_x - 400, min_y + 25),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color='#d62728')

# Highlight the spike at 2000 for context
ax1.annotate('Instability Spike', xy=(2000, 116.75), xytext=(2000 - 400, 116.75 + 25),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color='#d62728')

# Data extracted from the JSON logs
checkpoints_fid = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
fid_scores = [131.19, 16.41, 11.87, 13.38, 9.96, 9.60, 8.58, 10.27, 9.49, 8.34, 116.75, 8.31, 8.07, 12.97, 6.98, 7.20]

# Data extracted from the performance table
checkpoints_pr = [2200, 2400, 2800, 3000]
precision = [0.6594, 0.6933, 0.6825, 0.6706]
recall = [0.4249, 0.4300, 0.4332, 0.4414]

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: FID-50k tracking
ax1.plot(checkpoints_fid, fid_scores, marker='o', color='tab:red', label='FID-50k')
ax1.set_title('FID Score vs Checkpoint')
ax1.set_xlabel('Training Checkpoint')
ax1.set_ylabel('FID (Lower is better)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Subplot 2: Precision (Fidelity) and Recall (Diversity)
ax2.plot(checkpoints_pr, precision, marker='s', color='tab:blue', label='Precision')
ax2.plot(checkpoints_pr, recall, marker='^', color='tab:green', label='Recall')
ax2.set_title('Fidelity & Diversity vs Checkpoint')
ax2.set_xlabel('Training Checkpoint')
ax2.set_ylabel('Score (Higher is better)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Display the graphs
plt.tight_layout()
plt.show()