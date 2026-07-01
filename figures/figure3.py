"""
Figure 3. A two-axis pharmacogenomic model stratifies ARC treatment response.
  a: Composite pharmacogenomic scores on clinical-outcome PCA embedding (2x2)
  b: Two-axis pharmacogenomic landscape
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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

MM = 1 / 25.4  # mm -> inch

# ============================================================
# Data (shared)
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)
dos = pd.read_csv(f'{DATA}/snp_dosage_matrix_final.csv', encoding='cp949')
patient_ids = [str(pid) for pid in cl['patient_id']]

X = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
pcs = PCA(n_components=2).fit(StandardScaler().fit_transform(X))
pcs_t = pcs.transform(StandardScaler().fit_transform(X))
pc1, pc2 = pcs_t[:, 0], pcs_t[:, 1]
ev = pcs.explained_variance_ratio_

improvement = cl['Imp_CRST'].values
groups = cl['group'].values
n = len(cl)

pos_snps = [('MSRB3','rs10878269'),('OR7G3','rs10414255'),('NEIL3-AGA','rs1395479'),
            ('MPP7','rs1937810'),('PKD2L1','rs603424'),('TGFA','rs3771501'),('FAM185A','rs3972456')]
neg_snps = [('CPLX3-ULK3','rs6495122'),('RBFOX1','rs3095508'),('GC','rs2282679'),
            ('EMX2OS-RAB11FIP2','rs10886142'),('PKD1L2','rs8044334'),
            ('MPPED2-DCDC1','rs963837'),('DEF8','rs4268748'),('ZEB2','rs10427255')]

def get_dosage(gene, snp_id):
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    return row[patient_ids].values.flatten().astype(float) if len(row) else np.zeros(n)

pos_scores = sum(get_dosage(g, s) for g, s in pos_snps)
neg_scores_abs = sum(get_dosage(g, s) for g, s in neg_snps)
net_scores = pos_scores - neg_scores_abs

group_markers = {'SR': 'o', 'MR': 's', 'SF': 'D'}
group_sizes = {'SR': 34, 'MR': 28, 'SF': 28}

# Font sizes (pt) for print
FS_PANEL = 13     # panel letter (a, b)
FS_TITLE = 8      # sub-panel title
FS_AXIS = 7       # axis labels
FS_TICK = 6       # tick labels
FS_ANNOT = 6.5    # rho annotations
FS_LEG = 6.5      # legend

# ============================================================
# Figure canvas
# ============================================================
fig = plt.figure(figsize=(180 * MM, 215 * MM), facecolor='white')
subfigs = fig.subfigures(2, 1, height_ratios=[1.32, 1.0], hspace=0.0)

# ---------- Panel a: 2x2 gene-score PCA ----------
sfa = subfigs[0]
axes_a = sfa.subplots(2, 2)
sfa.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.09, hspace=0.42, wspace=0.42)

panels_a = [
    (axes_a[0, 0], improvement,     'CRST Improvement (%)',            'viridis',   'sequential',   False),
    (axes_a[0, 1], net_scores,      'Net Pharmacogenomic Score',       'viridis',   'diverging',    True),
    (axes_a[1, 0], pos_scores,      'Metabolic Resilience Score (+)',  'viridis',   'sequential',   True),
    (axes_a[1, 1], neg_scores_abs,  'Circuit Vulnerability Score (\u2212)', 'viridis_r', 'sequential_r', True),
]

for ax, scores, title, cmap_name, mode, show_corr in panels_a:
    cmap = plt.get_cmap(cmap_name)
    if mode == 'diverging':
        amax = max(abs(scores.min()), abs(scores.max()))
        norm = TwoSlopeNorm(vmin=-amax, vcenter=0, vmax=amax)
    else:
        norm = Normalize(vmin=scores.min(), vmax=scores.max())
    for i in range(n):
        ax.scatter(pc1[i], pc2[i], c=[cmap(norm(scores[i]))], s=group_sizes[groups[i]],
                   marker=group_markers[groups[i]], edgecolors='#333333', lw=0.5, zorder=5)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=FS_AXIS)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=FS_AXIS)
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', color='#1F4E79', pad=4)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = sfa.colorbar(sm, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.ax.tick_params(labelsize=FS_TICK - 0.5)
    if show_corr:
        rho, p = spearmanr(scores, improvement)
        if mode == 'sequential_r':
            rho = -abs(rho)
        p_str = f'{p:.1e}' if p < 0.001 else f'{p:.4f}'
        ax.text(0.97, 0.03, f'\u03c1 = {rho:+.3f}\np = {p_str}', transform=ax.transAxes,
                fontsize=FS_ANNOT, ha='right', va='bottom', color='#1F4E79', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Panel a legend (top, shared)
leg_a = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA', markersize=6,
           markeredgecolor='#333', markeredgewidth=0.8, label='SR (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#AAAAAA', markersize=5.5,
           markeredgecolor='#333', markeredgewidth=0.8, label='MR (n=11)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#AAAAAA', markersize=5.5,
           markeredgecolor='#333', markeredgewidth=0.8, label='SF (n=10)'),
]
sfa.legend(handles=leg_a, loc='upper center', ncol=3, fontsize=FS_LEG, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)

# ---------- Panel b: two-axis landscape ----------
sfb = subfigs[1]
# center the single axes at reduced width
ax_b = sfb.subplots(1, 1)
sfb.subplots_adjust(left=0.24, right=0.82, top=0.98, bottom=0.15)

imp_cmap = plt.cm.viridis
imp_norm = Normalize(vmin=-10, vmax=65)
rng = np.random.default_rng(42)
jx = rng.uniform(-0.18, 0.18, n)
jy = rng.uniform(-0.18, 0.18, n)

for i in range(n):
    ax_b.scatter(pos_scores[i] + jx[i], neg_scores_abs[i] + jy[i],
                 c=[imp_cmap(imp_norm(improvement[i]))], s=group_sizes[groups[i]] + 6,
                 marker=group_markers[groups[i]], edgecolors='#333333', lw=0.7, zorder=5)

ax_b.text(1.8, 10.5, 'ARC-\nRefractory', ha='center', va='center',
          fontsize=11, fontweight='bold', color='#7B2D26', alpha=0.3, style='italic')
ax_b.text(8.5, 3.2, 'ARC-\nResponsive', ha='center', va='center',
          fontsize=11, fontweight='bold', color='#1B7340', alpha=0.3, style='italic')

ax_b.set_xlabel('Metabolic Resilience Score (+)\n'
                '(MSRB3, OR7G3, NEIL3-AGA, MPP7, PKD2L1, TGFA, FAM185A)',
                fontsize=FS_AXIS, fontweight='bold', labelpad=6)
ax_b.set_ylabel('Circuit Vulnerability Score (\u2212)\n'
                '(CPLX3-ULK3, RBFOX1, GC, EMX2OS-RAB11FIP2,\nPKD1L2, MPPED2-DCDC1, DEF8, ZEB2)',
                fontsize=FS_AXIS, fontweight='bold', labelpad=6)
ax_b.set_xlim(0, 10)
ax_b.set_ylim(2, 12)
ax_b.tick_params(labelsize=FS_TICK)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(alpha=0.15, zorder=0)

sm_b = ScalarMappable(cmap=imp_cmap, norm=imp_norm); sm_b.set_array([])
cbar_b = sfb.colorbar(sm_b, ax=ax_b, shrink=0.7, aspect=22, pad=0.03)
cbar_b.set_label('CRST Improvement (%)', fontsize=FS_AXIS, fontweight='bold')
cbar_b.ax.tick_params(labelsize=FS_TICK)

leg_b = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA', markersize=6.5,
           markeredgecolor='#333', markeredgewidth=1, label='SR: Severe-Responsive (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#AAAAAA', markersize=6,
           markeredgecolor='#333', markeredgewidth=1, label='MR: Mild-Responsive (n=11)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#AAAAAA', markersize=6,
           markeredgecolor='#333', markeredgewidth=1, label='SF: Severe-Refractory (n=10)'),
]
ax_b.legend(handles=leg_b, loc='lower left', fontsize=FS_LEG, framealpha=0.95,
            edgecolor='#CCCCCC', title='Response Group', title_fontsize=FS_LEG)

rho_pos, p_pos = spearmanr(pos_scores, improvement)
rho_neg, p_neg = spearmanr(neg_scores_abs, improvement)
ax_b.text(0.97, 0.97,
          f'X vs Imp: \u03c1 = {rho_pos:+.3f} (p = {p_pos:.4f})\n'
          f'Y vs Imp: \u03c1 = {-abs(rho_neg):+.3f} (p = {p_neg:.1e})',
          transform=ax_b.transAxes, fontsize=FS_ANNOT, ha='right', va='top',
          color='#1F4E79', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# ---------- Panel letters ----------
sfa.text(0.005, 0.985, 'a', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sfb.text(0.005, 0.985, 'b', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')

fig.savefig(f'{OUT}/Figure3.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/Figure3.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved Figure3.pdf/png")
