import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
DATA = str(_ROOT / "data")
STATS = str(_ROOT / "data" / "stats")
OUT = str(_ROOT / "output")
Path(OUT).mkdir(parents=True, exist_ok=True)
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)

# Standardize clinical variables for clustering
X = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ward linkage
Z = linkage(X_scaled, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')

# Color map: SR=blue, MR=green, SF=red
group_colors = {'SR': '#2166AC', 'MR': '#4DAF4A', 'SF': '#D6604D'}
group_labels_full = {
    'SR': 'Severe-Responsive (SR)',
    'MR': 'Mild-Responsive (MR)',
    'SF': 'Severe-Refractory (SF)',
}

# Map each leaf to its group color
groups = cl['group'].values

# ============================================================
# Custom dendrogram coloring
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')

# Plot dendrogram with no labels
dn = dendrogram(Z, ax=ax, no_labels=True, above_threshold_color='#888888',
                color_threshold=0)

# Get leaf order
leaf_order = dn['leaves']
leaf_groups = [groups[i] for i in leaf_order]

# Color the bottom links by cluster membership
# We'll add colored bars at the bottom instead
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Add group color bars at bottom
bar_height = ylim[1] * 0.04
bar_y = -bar_height * 1.5

# x positions of leaves in dendrogram
n = len(leaf_order)
x_positions = np.arange(5, 10 * n, 10)  # dendrogram default spacing

for i, (xp, g) in enumerate(zip(x_positions, leaf_groups)):
    ax.bar(xp, bar_height, bottom=bar_y, width=8, color=group_colors[g],
          edgecolor='none', zorder=10, clip_on=False)

# Add group name labels below the bars
# Find contiguous runs of same group
runs = []
current_group = leaf_groups[0]
start_idx = 0
for i in range(1, len(leaf_groups)):
    if leaf_groups[i] != current_group:
        runs.append((current_group, start_idx, i - 1))
        current_group = leaf_groups[i]
        start_idx = i
runs.append((current_group, start_idx, len(leaf_groups) - 1))

for g, s, e in runs:
    mid_x = (x_positions[s] + x_positions[e]) / 2
    label = group_labels_full[g]
    ax.text(mid_x, bar_y - bar_height * 1.5, label,
           ha='center', va='top', fontsize=10, fontweight='bold',
           color=group_colors[g], clip_on=False)

# Formatting
ax.set_ylabel('Ward Distance', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_ylim(bar_y - bar_height * 4, ylim[1])

fig.suptitle('Hierarchical Clustering Dendrogram (Ward Linkage)',
            fontsize=15, fontweight='bold', color='#1F4E79', y=0.97)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(f'{OUT}/dendrogram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/dendrogram.pdf', bbox_inches='tight', facecolor='white')
print("Saved dendrogram.png/pdf")
