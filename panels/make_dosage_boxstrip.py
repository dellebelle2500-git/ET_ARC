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
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
dos = pd.read_csv(f'{DATA}/snp_dosage_matrix_final.csv', encoding='cp949')
patient_ids = [str(pid) for pid in cl['patient_id']]

improvement = cl['Imp_CRST'].values

def get_dosage(gene, snp_id):
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    if len(row) > 0:
        return row[patient_ids].values.flatten().astype(float)
    return np.zeros(len(patient_ids))

# 8 SNPs: compute p-values for sorting
snps_raw = [
    ('MSRB3', 'rs10878269', '+'),
    ('OR7G3', 'rs10414255', '+'),
    ('NEIL3-AGA', 'rs1395479', '+'),
    ('CPLX3-ULK3', 'rs6495122', '−'),
    ('RBFOX1', 'rs3095508', '−'),
    ('GC', 'rs2282679', '−'),
    ('EMX2OS-RAB11FIP2', 'rs10886142', '−'),
    ('PKD1L2', 'rs8044334', '−'),
]

snps_with_p = []
for gene, snp_id, direction in snps_raw:
    dosage = get_dosage(gene, snp_id)
    _, p = spearmanr(dosage, improvement)
    snps_with_p.append((gene, snp_id, direction, p))

pos_snps_sorted = sorted([s for s in snps_with_p if s[2] == '+'], key=lambda x: x[3])
neg_snps_sorted = sorted([s for s in snps_with_p if s[2] == '−'], key=lambda x: x[3])
snps = [(g, s, d) for g, s, d, _ in pos_snps_sorted + neg_snps_sorted]

for g, s, d, p in pos_snps_sorted + neg_snps_sorted:
    print(f"  {g:25s} {s:15s} {d}  p={p:.4f}")

# ============================================================
# Figure: 2 rows x 4 columns
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='white')
axes = axes.flatten()

colors_dosage = {0: '#4A90D9', 1: '#F5A623', 2: '#D0021B'}
box_colors = ['#4A90D9', '#F5A623', '#D0021B']

for idx, (gene, snp_id, direction) in enumerate(snps):
    ax = axes[idx]
    dosage = get_dosage(gene, snp_id).astype(int)
    
    # Group by dosage
    groups = {}
    for d in [0, 1, 2]:
        mask = dosage == d
        if mask.sum() > 0:
            groups[d] = improvement[mask]
    
    available_doses = sorted(groups.keys())
    positions = list(range(len(available_doses)))
    
    # Box plots
    bp_data = [groups[d] for d in available_doses]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#333333', linewidth=1.5),
                    whiskerprops=dict(color='#666666'),
                    capprops=dict(color='#666666'))
    
    for patch, d in zip(bp['boxes'], available_doses):
        patch.set_facecolor(colors_dosage[d])
        patch.set_alpha(0.35)
        patch.set_edgecolor(colors_dosage[d])
    
    # Strip (jittered points)
    rng = np.random.default_rng(42 + idx)
    for i, d in enumerate(available_doses):
        vals = groups[d]
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(positions[i] + jitter, vals, c=colors_dosage[d], s=40, alpha=0.75,
                  edgecolors='white', lw=0.5, zorder=5)
    
    # Spearman correlation
    rho, p = spearmanr(dosage, improvement)
    p_str = f'{p:.1e}' if p < 0.001 else f'{p:.4f}'
    
    # Title with gene name on one line + axis label below
    dir_color = '#1B7340' if direction == '+' else '#B91C1C'
    dir_text = 'Metabolic Resilience' if direction == '+' else 'Circuit Vulnerability'
    dir_sign = '(+)' if direction == '+' else '(−)'
    
    ax.set_title(f'{gene} ({snp_id})\n{dir_sign} {dir_text}',
                fontsize=11, fontweight='bold', color='#1F4E79')
    
    # Stats annotation
    ax.text(0.97, 0.03, f'ρ = {rho:+.3f}\np = {p_str}',
           transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
           color='#1F4E79', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Dosage {d}\n(n={len(groups[d])})' for d in available_doses], fontsize=9)
    ax.set_ylabel('CRST Improvement (%)', fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('CRST Improvement (%) by Non-Ref Allele Dosage',
            fontsize=17, fontweight='bold', color='#1F4E79', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT}/Dosage_8snps.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/Dosage_8snps.pdf', bbox_inches='tight', facecolor='white')
print("Saved Dosage_8snps.png/pdf")
