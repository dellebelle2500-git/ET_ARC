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
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
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

dos = pd.read_csv(f'{DATA}/snp_dosage_matrix_final.csv', encoding='cp949')
patient_ids = [str(pid) for pid in cl['patient_id']]

# PCA from clinical variables
X = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
pc1, pc2 = pcs[:, 0], pcs[:, 1]
ev = pca.explained_variance_ratio_

improvement = cl['Imp_CRST'].values
groups = cl['group'].values

# 15 Tier SNPs (SCARB1 excluded)
# Tier 1: MSRB3
# Tier 2a (Dosage-only): CPLX3-ULK3, OR7G3, RBFOX1, GC, EMX2OS-RAB11FIP2, PKD1L2, NEIL3-AGA
# Tier 2b (Group-only): MPP7, MPPED2-DCDC1, DEF8, FAM185A, PKD2L1, TGFA, ZEB2
tier_snps = [
    # (+) Metabolic Resilience — by |ρ| descending
    ('MSRB3',        'rs10878269', '+', 'Tier 1'),    # ρ=+0.414
    ('OR7G3',        'rs10414255', '+', 'Tier 2a'),   # ρ=+0.475
    ('NEIL3-AGA',    'rs1395479',  '+', 'Tier 2a'),   # ρ=+0.400
    ('MPP7',         'rs1937810',  '+', 'Tier 2b'),   # ρ=+0.254
    ('PKD2L1',       'rs603424',   '+', 'Tier 2b'),   # ρ=+0.214
    ('FAM185A',      'rs3972456',  '+', 'Tier 2b'),   # ρ=+0.073
    ('TGFA',         'rs3771501',  '+', 'Tier 2b'),   # ρ=+0.044
    # (−) Circuit Vulnerability — by |ρ| descending
    ('CPLX3-ULK3',   'rs6495122',  '−', 'Tier 2a'),  # ρ=−0.521
    ('RBFOX1',       'rs3095508',  '−', 'Tier 2a'),   # ρ=−0.409
    ('EMX2OS-RAB11FIP2', 'rs10886142', '−', 'Tier 2a'), # ρ=−0.395
    ('GC',           'rs2282679',  '−', 'Tier 2a'),   # ρ=−0.393
    ('PKD1L2',       'rs8044334',  '−', 'Tier 2a'),   # ρ=−0.387
    ('DEF8',         'rs4268748',  '−', 'Tier 2b'),   # ρ=−0.333
    ('ZEB2',         'rs10427255', '−', 'Tier 2b'),   # ρ=−0.283
    ('MPPED2-DCDC1', 'rs963837',   '−', 'Tier 2b'),  # ρ=−0.152
]

def get_dosage(gene, snp_id):
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    if len(row) > 0:
        return row[patient_ids].values.flatten().astype(float)
    return np.zeros(len(patient_ids))

# Group markers
group_markers = {'SR': 'o', 'MR': 's', 'SF': 'D'}
group_sizes = {'SR': 60, 'MR': 50, 'SF': 50}

# Dosage colormap: viridis-based (0=dark blue, 1=teal, 2=yellow)
dose_colors_map = {0: '#440154', 1: '#21918C', 2: '#FDE725'}

# ============================================================
# Figure: 4x4 grid
# ============================================================
fig, axes = plt.subplots(4, 4, figsize=(18, 18), facecolor='white')
axes = axes.flatten()

# --- Panel 0: CRST Improvement (viridis) ---
ax = axes[0]
imp_cmap = plt.cm.viridis
imp_norm = Normalize(vmin=-10, vmax=65)

for i in range(len(cl)):
    g = groups[i]
    color = imp_cmap(imp_norm(improvement[i]))
    ax.scatter(pc1[i], pc2[i], c=[color], s=group_sizes[g],
              marker=group_markers[g], edgecolors='#333333', lw=0.6, zorder=5)

ax.set_title('CRST Improvement (%)', fontsize=10, fontweight='bold', color='#1F4E79')
sm = ScalarMappable(cmap=imp_cmap, norm=imp_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.02)
cbar.ax.tick_params(labelsize=7)

# --- Panels 1-15: SNP dosage overlays ---
for idx, (gene, snp_id, direction, tier) in enumerate(tier_snps):
    ax = axes[idx + 1]
    dosage = get_dosage(gene, snp_id).astype(int)
    
    for i in range(len(cl)):
        g = groups[i]
        d = dosage[i]
        ax.scatter(pc1[i], pc2[i], c=dose_colors_map[d], s=group_sizes[g],
                  marker=group_markers[g], edgecolors='#333333', lw=0.6, zorder=5)
    
    dir_sign = '(+)' if direction == '+' else '(−)'
    dir_text = 'Met. Resilience' if direction == '+' else 'Circuit Vuln.'
    ax.set_title(f'{gene}\n{dir_sign} {dir_text} | {tier}',
                fontsize=9, fontweight='bold', color='#1F4E79')
    
    # Spearman correlation (dosage vs CRST improvement)
    from scipy.stats import spearmanr
    rho, p = spearmanr(dosage, improvement)
    p_str = f'{p:.1e}' if p < 0.01 else f'{p:.3f}'
    ax.text(0.97, 0.03, f'ρ = {rho:+.3f}',
           transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
           color='#1F4E79', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Common formatting
for idx, ax in enumerate(axes):
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=8)
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)

# Legend for response groups
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999', markersize=8,
           markeredgecolor='#333', markeredgewidth=0.8, label='SR (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#999', markersize=7,
           markeredgecolor='#333', markeredgewidth=0.8, label='MR (n=11)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#999', markersize=7,
           markeredgecolor='#333', markeredgewidth=0.8, label='SF (n=10)'),
]
# Legend for dosage
from matplotlib.patches import Patch
dose_legend = [
    Patch(facecolor='#440154', edgecolor='#333', label='Dosage 0'),
    Patch(facecolor='#21918C', edgecolor='#333', label='Dosage 1'),
    Patch(facecolor='#FDE725', edgecolor='#333', label='Dosage 2'),
]

fig.legend(handles=legend_elements + dose_legend, loc='lower center',
          ncol=6, fontsize=10, framealpha=0.95, edgecolor='#CCC',
          bbox_to_anchor=(0.5, -0.01))

fig.suptitle('PCA Embedding with SNP Dosage Overlay (Tier 1 / 2a / 2b)',
            fontsize=16, fontweight='bold', color='#1F4E79', y=1.0)

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(f'{OUT}/PCA_SNP_overlay.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/PCA_SNP_overlay.pdf', bbox_inches='tight', facecolor='white')
print("Saved PCA_SNP_overlay.png/pdf")
