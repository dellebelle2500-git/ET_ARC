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
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)

# PCA from clinical variables
X = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
pc1, pc2 = pcs[:, 0], pcs[:, 1]
ev = pca.explained_variance_ratio_

groups = cl['group'].values
names = cl['name'].values

# ============================================================
# Style matching ward_k3_boxstrip colors
# ============================================================
group_config = {
    'SR': {'color': '#2166AC', 'marker': 'o', 'size': 120, 'label': 'Severe-Responsive (SR, n=6)'},
    'MR': {'color': '#4DAF4A', 'marker': 's', 'size': 100, 'label': 'Mild-Responsive (MR, n=11)'},
    'SF': {'color': '#D6604D', 'marker': 'D', 'size': 100, 'label': 'Severe-Refractory (SF, n=10)'},
}

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

# Plot patients
for i in range(len(cl)):
    g = groups[i]
    cfg = group_config[g]
    ax.scatter(pc1[i], pc2[i], c=cfg['color'], s=cfg['size'],
              marker=cfg['marker'], edgecolors='#333333', lw=0.8, zorder=5)

ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.15)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC', markersize=12,
           markeredgecolor='#333', markeredgewidth=1, label=group_config['SR']['label']),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#4DAF4A', markersize=11,
           markeredgecolor='#333', markeredgewidth=1, label=group_config['MR']['label']),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#D6604D', markersize=11,
           markeredgecolor='#333', markeredgewidth=1, label=group_config['SF']['label']),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95,
         edgecolor='#CCCCCC', title='Response Group', title_fontsize=11)

fig.suptitle('PCA of Clinical Outcomes with Hierarchical Clustering (Ward, k=3)',
            fontsize=14, fontweight='bold', color='#1F4E79', y=0.97)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT}/PCA_clustering.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/PCA_clustering.pdf', bbox_inches='tight', facecolor='white')
print("Saved PCA_clustering.png/pdf")
