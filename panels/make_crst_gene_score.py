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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)

dos = pd.read_csv(f'{DATA}/snp_dosage_matrix_final.csv', encoding='cp949')
patient_ids = [str(pid) for pid in cl['patient_id']]

# PCA
X = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
pc1, pc2 = pcs[:, 0], pcs[:, 1]
ev = pca.explained_variance_ratio_

improvement = cl['Imp_CRST'].values
groups = cl['group'].values
n = len(cl)

# Gene scores
pos_snps = [
    ('MSRB3','rs10878269'), ('OR7G3','rs10414255'), ('NEIL3-AGA','rs1395479'),
    ('MPP7','rs1937810'), ('PKD2L1','rs603424'), ('TGFA','rs3771501'), ('FAM185A','rs3972456'),
]
neg_snps = [
    ('CPLX3-ULK3','rs6495122'), ('RBFOX1','rs3095508'), ('GC','rs2282679'),
    ('EMX2OS-RAB11FIP2','rs10886142'), ('PKD1L2','rs8044334'),
    ('MPPED2-DCDC1','rs963837'), ('DEF8','rs4268748'), ('ZEB2','rs10427255'),
]

def get_dosage(gene, snp_id):
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    if len(row) > 0:
        return row[patient_ids].values.flatten().astype(float)
    return np.zeros(n)

pos_scores = np.zeros(n)
neg_scores_abs = np.zeros(n)
for gene, snp in pos_snps:
    pos_scores += get_dosage(gene, snp)
for gene, snp in neg_snps:
    neg_scores_abs += get_dosage(gene, snp)
net_scores = pos_scores - neg_scores_abs

# Group markers
group_markers = {'SR': 'o', 'MR': 's', 'SF': 'D'}
group_sizes = {'SR': 100, 'MR': 85, 'SF': 85}

# ============================================================
# Figure: 2x2
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor='white')

panels = [
    # (ax, scores, title, cmap, norm_mode, show_corr)
    (axes[0, 0], improvement, 'CRST Improvement (%)', 'viridis', 'sequential', False),
    (axes[0, 1], net_scores, 'Net Pharmacogenomic Score', 'viridis', 'diverging', True),
    (axes[1, 0], pos_scores, 'Metabolic Resilience Score (+)', 'viridis', 'sequential', True),
    (axes[1, 1], neg_scores_abs, 'Circuit Vulnerability Score (\u2212)', 'viridis_r', 'sequential_r', True),
]

for ax, scores, title, cmap_name, mode, show_corr in panels:
    cmap = plt.get_cmap(cmap_name)
    
    if mode == 'diverging':
        abs_max = max(abs(scores.min()), abs(scores.max()))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=scores.min(), vmax=scores.max())
    
    for i in range(n):
        g = groups[i]
        color = cmap(norm(scores[i]))
        ax.scatter(pc1[i], pc2[i], c=[color], s=group_sizes[g],
                  marker=group_markers[g], edgecolors='#333333', lw=0.7, zorder=5)
    
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#1F4E79', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, aspect=25, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    
    # Stats (correlation with improvement)
    if show_corr:
        rho, p = spearmanr(scores, improvement)
        if mode == 'sequential_r':
            rho = -abs(rho)
        p_str = f'{p:.1e}' if p < 0.001 else f'{p:.4f}'
        ax.text(0.97, 0.03, f'\u03c1 = {rho:+.3f}\np = {p_str}',
               transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
               color='#1F4E79', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA', markersize=11,
           markeredgecolor='#333', markeredgewidth=1, label='SR: Severe-Responsive (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#AAAAAA', markersize=10,
           markeredgecolor='#333', markeredgewidth=1, label='MR: Mild-Responsive (n=11)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#AAAAAA', markersize=10,
           markeredgecolor='#333', markeredgewidth=1, label='SF: Severe-Refractory (n=10)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
          framealpha=0.95, edgecolor='#CCC', bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Patient-Level Pharmacogenomic Score Mapping on PCA Embedding',
            fontsize=16, fontweight='bold', color='#1F4E79', y=0.99)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f'{OUT}/CRST_gene_score_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/CRST_gene_score_scatter.pdf', bbox_inches='tight', facecolor='white')
print("Saved CRST_gene_score_scatter.png/pdf")
