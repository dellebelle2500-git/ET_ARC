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
from scipy import stats
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
    return np.zeros(len(patient_ids))

n = len(cl)
pos_scores = np.zeros(n)
neg_scores_abs = np.zeros(n)

for gene, snp in pos_snps:
    pos_scores += get_dosage(gene, snp)
for gene, snp in neg_snps:
    neg_scores_abs += get_dosage(gene, snp)

improvement = cl['Imp_CRST'].values
groups = cl['group'].values

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')

# Viridis colormap
imp_cmap = plt.cm.viridis
imp_norm = Normalize(vmin=-10, vmax=65)

# No quadrant backgrounds — clean white

# Marker config
group_markers = {'SR': 'o', 'MR': 's', 'SF': 'D'}
group_sizes = {'SR': 120, 'MR': 100, 'SF': 100}

# Jitter
rng = np.random.default_rng(42)
jitter_x = rng.uniform(-0.18, 0.18, n)
jitter_y = rng.uniform(-0.18, 0.18, n)

# Plot each patient
for i in range(n):
    g = groups[i]
    color = imp_cmap(imp_norm(improvement[i]))
    ax.scatter(pos_scores[i] + jitter_x[i], neg_scores_abs[i] + jitter_y[i],
              c=[color], s=group_sizes[g], marker=group_markers[g],
              edgecolors='#333333', lw=1.0, zorder=5)

# Quadrant annotations (normal y: top=high vulnerability, bottom=low vulnerability)
# Top-left: low resilience + high vulnerability = refractory
ax.text(1.8, 10.5, 'ARC-\nRefractory', ha='center', va='center',
       fontsize=15, fontweight='bold', color='#7B2D26', alpha=0.3, style='italic')
# Bottom-right: high resilience + low vulnerability = responsive
ax.text(8.5, 3.2, 'ARC-\nResponsive', ha='center', va='center',
       fontsize=15, fontweight='bold', color='#1B7340', alpha=0.3, style='italic')

# Axis labels with UPDATED names
ax.set_xlabel('Metabolic Resilience Score (+)\n'
              '(MSRB3, OR7G3, NEIL3-AGA, MPP7, PKD2L1, TGFA, FAM185A)',
              fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Circuit Vulnerability Score (\u2212)\n'
              '(CPLX3-ULK3, RBFOX1, GC, EMX2OS-RAB11FIP2,\nPKD1L2, MPPED2-DCDC1, DEF8, ZEB2)',
              fontsize=11, fontweight='bold', labelpad=10)

# Axis formatting
ax.set_xlim(0, 10)
ax.set_ylim(2, 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.15, zorder=0)

# Colorbar
sm = ScalarMappable(cmap=imp_cmap, norm=imp_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.65, aspect=25, pad=0.02)
cbar.set_label('CRST Improvement (%)', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA', markersize=12,
           markeredgecolor='#333333', markeredgewidth=1.2, label='SR: Severe-Responsive (n=6)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#AAAAAA', markersize=11,
           markeredgecolor='#333333', markeredgewidth=1.2, label='MR: Mild-Responsive (n=11)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#AAAAAA', markersize=11,
           markeredgecolor='#333333', markeredgewidth=1.2, label='SF: Severe-Refractory (n=10)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9.5, framealpha=0.95,
         edgecolor='#CCCCCC', title='Response Group', title_fontsize=10)

# Correlation annotation
rho_pos, p_pos = stats.spearmanr(pos_scores, improvement)
rho_neg, p_neg = stats.spearmanr(neg_scores_abs, improvement)
ax.text(0.98, 0.02,
       f'X vs Imp: \u03c1 = {rho_pos:+.3f} (p = {p_pos:.4f})\n'
       f'Y vs Imp: \u03c1 = {-abs(rho_neg):+.3f} (p = {p_neg:.1e})',
       transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
       color='#1F4E79', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Title only
fig.suptitle('Two-Axis Pharmacogenomic Landscape of ET Patients',
            fontsize=15, fontweight='bold', color='#1F4E79', y=0.97)

plt.savefig(f'{OUT}/two_axis_patient_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/two_axis_patient_scatter.pdf', bbox_inches='tight', facecolor='white')
print("Saved")
