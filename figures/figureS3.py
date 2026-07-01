"""
Supplementary Figure S3. Per-locus SNP dosage overlaid on the clinical-outcome PCA embedding.
Top-left: CRST improvement (%) anchor; remaining 15 panels: tiered loci by non-ref allele dosage.
Nature Communications double-column width = 180 mm.
"""
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

MM = 1 / 25.4

# Font sizes
FS_TITLE = 6.5
FS_AXIS = 6.0
FS_TICK = 5.5
FS_ANNOT = 6.0
FS_LEG = 7.5

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)
dos = pd.read_csv(f'{DATA}/snp_dosage_matrix_final.csv', encoding='cp949')
patient_ids = [str(pid) for pid in cl['patient_id']]

X = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
pca = PCA(n_components=2).fit(StandardScaler().fit_transform(X))
pcs = pca.transform(StandardScaler().fit_transform(X))
pc1, pc2 = pcs[:, 0], pcs[:, 1]
ev = pca.explained_variance_ratio_

improvement = cl['Imp_CRST'].values
groups = cl['group'].values

tier_snps = [
    ('MSRB3','rs10878269','+','Tier 1'), ('OR7G3','rs10414255','+','Tier 2a'),
    ('NEIL3-AGA','rs1395479','+','Tier 2a'), ('MPP7','rs1937810','+','Tier 2b'),
    ('PKD2L1','rs603424','+','Tier 2b'), ('FAM185A','rs3972456','+','Tier 2b'),
    ('TGFA','rs3771501','+','Tier 2b'),
    ('CPLX3-ULK3','rs6495122','−','Tier 2a'), ('RBFOX1','rs3095508','−','Tier 2a'),
    ('EMX2OS-RAB11FIP2','rs10886142','−','Tier 2a'), ('GC','rs2282679','−','Tier 2a'),
    ('PKD1L2','rs8044334','−','Tier 2a'), ('DEF8','rs4268748','−','Tier 2b'),
    ('ZEB2','rs10427255','−','Tier 2b'), ('MPPED2-DCDC1','rs963837','−','Tier 2b'),
]

def get_dosage(gene, snp_id):
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    return row[patient_ids].values.flatten().astype(float) if len(row) else np.zeros(len(cl))

group_markers = {'SR': 'o', 'MR': 's', 'SF': 'D'}
group_sizes = {'SR': 22, 'MR': 18, 'SF': 18}
dose_colors_map = {0: '#440154', 1: '#21918C', 2: '#FDE725'}

# ============================================================
# Figure: 4x4
# ============================================================
fig = plt.figure(figsize=(180 * MM, 190 * MM), facecolor='white')
axes = fig.subplots(4, 4)
fig.subplots_adjust(left=0.055, right=0.965, top=0.955, bottom=0.075, hspace=0.62, wspace=0.42)
axes = axes.flatten()

# Panel 0: CRST improvement anchor
ax0 = axes[0]
imp_cmap = plt.cm.viridis
imp_norm = Normalize(vmin=-10, vmax=65)
for i in range(len(cl)):
    ax0.scatter(pc1[i], pc2[i], c=[imp_cmap(imp_norm(improvement[i]))], s=group_sizes[groups[i]],
                marker=group_markers[groups[i]], edgecolors='#333333', lw=0.4, zorder=5)
ax0.set_title('CRST Improvement (%)', fontsize=FS_TITLE, fontweight='bold', color='#1F4E79', pad=3)
sm = ScalarMappable(cmap=imp_cmap, norm=imp_norm); sm.set_array([])
cb = fig.colorbar(sm, ax=ax0, shrink=0.8, aspect=15, pad=0.03)
cb.ax.tick_params(labelsize=FS_TICK - 0.5)

# Panels 1-15: dosage overlays
for idx, (gene, snp_id, direction, tier) in enumerate(tier_snps):
    ax = axes[idx + 1]
    dosage = get_dosage(gene, snp_id).astype(int)
    for i in range(len(cl)):
        ax.scatter(pc1[i], pc2[i], c=dose_colors_map[dosage[i]], s=group_sizes[groups[i]],
                   marker=group_markers[groups[i]], edgecolors='#333333', lw=0.4, zorder=5)
    dir_sign = '(+)' if direction == '+' else '(−)'
    dir_text = 'Met. Resilience' if direction == '+' else 'Circuit Vuln.'
    ax.set_title(f'{gene}\n{dir_sign} {dir_text} | {tier}', fontsize=FS_TITLE,
                 fontweight='bold', color='#1F4E79', pad=3, linespacing=1.1)
    rho, p = spearmanr(dosage, improvement)
    ax.text(0.96, 0.04, f'\u03c1 = {rho:+.3f}', transform=ax.transAxes, fontsize=FS_ANNOT,
            ha='right', va='bottom', color='#1F4E79', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Common formatting
for idx, ax in enumerate(axes):
    if idx >= 12:
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=FS_AXIS)
    if idx % 4 == 0:
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)

# Shared legend at bottom
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999', markersize=6,
           markeredgecolor='#333', markeredgewidth=0.8, label='SR (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#999', markersize=5.5,
           markeredgecolor='#333', markeredgewidth=0.8, label='MR (n=11)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#999', markersize=5.5,
           markeredgecolor='#333', markeredgewidth=0.8, label='SF (n=10)'),
    Patch(facecolor='#440154', edgecolor='#333', label='Dosage 0'),
    Patch(facecolor='#21918C', edgecolor='#333', label='Dosage 1'),
    Patch(facecolor='#FDE725', edgecolor='#333', label='Dosage 2'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=FS_LEG,
           framealpha=0.95, edgecolor='#CCC', bbox_to_anchor=(0.5, -0.005),
           handletextpad=0.4, columnspacing=1.3)

fig.savefig(f'{OUT}/FigureS3.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/FigureS3.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved FigureS3.pdf/png")
