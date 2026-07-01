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
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, kruskal
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
groups = cl['group'].values
group_labels = ['SR', 'MR', 'SF']
group_ns = {'SR': 6, 'MR': 11, 'SF': 10}

target_snps = [
    ('MSRB3',        'rs10878269', '+'),
    ('MPP7',         'rs1937810',  '+'),
    ('FAM185A',      'rs3972456',  '+'),
    ('PKD2L1',       'rs603424',   '+'),
    ('TGFA',         'rs3771501',  '+'),
    ('MPPED2-DCDC1', 'rs963837',   '−'),
    ('DEF8',         'rs4268748',  '−'),
    ('ZEB2',         'rs10427255', '−'),
]

# Stats (from previous analysis)
stats_info = {
    'MSRB3':        {'tier': 'Tier 1',  'fisher': 0.054, 'kw': 0.044, 'gtest': 0.030,
                     'SR-MR': 0.270, 'SR-SF': 0.298, 'MR-SF': 0.015},
    'MPP7':         {'tier': 'Tier 2b', 'fisher': 0.027, 'kw': 0.009, 'gtest': 0.009,
                     'SR-MR': 0.004, 'SR-SF': 0.138, 'MR-SF': 0.080},
    'MPPED2-DCDC1': {'tier': 'Tier 2b', 'fisher': 0.038, 'kw': 0.021, 'gtest': 0.015,
                     'SR-MR': 0.005, 'SR-SF': 0.138, 'MR-SF': 0.207},
    'DEF8':         {'tier': 'Tier 2b', 'fisher': 0.049, 'kw': 0.063, 'gtest': 0.037,
                     'SR-MR': 1.000, 'SR-SF': 0.173, 'MR-SF': 0.063},
    'FAM185A':      {'tier': 'Tier 2b', 'fisher': 0.020, 'kw': 0.084, 'gtest': 0.011,
                     'SR-MR': 0.015, 'SR-SF': 0.147, 'MR-SF': 0.765},
    'PKD2L1':       {'tier': 'Tier 2b', 'fisher': 0.057, 'kw': 0.037, 'gtest': 0.016,
                     'SR-MR': 0.116, 'SR-SF': 1.000, 'MR-SF': 0.044},
    'TGFA':         {'tier': 'Tier 2b', 'fisher': 0.018, 'kw': 0.733, 'gtest': 0.009,
                     'SR-MR': 0.776, 'SR-SF': 0.519, 'MR-SF': 0.516},
    'ZEB2':         {'tier': 'Tier 2b', 'fisher': 0.001, 'kw': 0.184, 'gtest': 0.002,
                     'SR-MR': 0.218, 'SR-SF': 0.193, 'MR-SF': 0.175},
}

# Sort within each axis by min omnibus p-value
def min_omnibus_p(gene):
    info = stats_info[gene]
    return min(info['fisher'], info['kw'], info['gtest'])

pos_snps = [(g, s, d) for g, s, d in target_snps if d == '+']
neg_snps = [(g, s, d) for g, s, d in target_snps if d == '−']
pos_snps.sort(key=lambda x: min_omnibus_p(x[0]))
neg_snps.sort(key=lambda x: min_omnibus_p(x[0]))
target_snps_sorted = pos_snps + neg_snps

for g, s, d in target_snps_sorted:
    print(f"  {g:20s} {d}  min_p={min_omnibus_p(g):.4f}")

# Get dosage data
gene_data = {}
for gene, snp_id, direction in target_snps_sorted:
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    if len(row) > 0:
        dosages = row[patient_ids].values.flatten().astype(float)
        grp_dosages = {}
        for g in group_labels:
            mask = groups == g
            grp_dosages[g] = dosages[mask]
        gene_data[gene] = {'snp': snp_id, 'dosages': grp_dosages}

# Colors matching Dosage 8snps style
dose_colors = {0: '#4A90D9', 1: '#F5A623', 2: '#D0021B'}

def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''

def draw_bracket(ax, x1, x2, y, p, dh=2.5, barh=1.0, fontsize=11):
    stars = p_to_stars(p)
    if not stars:
        return y
    ax.plot([x1, x1, x2, x2], [y, y + barh, y + barh, y], color='#333333', lw=1.2, clip_on=False)
    ax.text((x1 + x2) / 2, y + barh + 0.5, stars, ha='center', va='bottom',
           fontsize=fontsize, fontweight='bold', color='#333333', clip_on=False)
    return y + barh + dh

# ============================================================
# Figure: 2x4 matching Dosage 8snps layout
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor='white')
axes = axes.flatten()

for idx, (gene, snp_id, direction) in enumerate(target_snps_sorted):
    ax = axes[idx]
    d = gene_data[gene]
    info = stats_info[gene]
    
    x_pos = np.arange(3)
    bar_width = 0.25
    
    # Compute frequencies (%)
    freq_data = {}
    for dose_val in [0, 1, 2]:
        freqs = []
        for g in group_labels:
            c = int(np.sum(d['dosages'][g] == dose_val))
            freqs.append(c / group_ns[g] * 100)
        freq_data[dose_val] = freqs
    
    # Draw bars
    for di, dose_val in enumerate([0, 1, 2]):
        offset = (di - 1) * bar_width
        ax.bar(x_pos + offset, freq_data[dose_val], bar_width * 0.85,
               color=dose_colors[dose_val], edgecolor='white', lw=0.8, zorder=3)
        for xi, pct in enumerate(freq_data[dose_val]):
            if pct > 0:
                ax.text(x_pos[xi] + offset, pct + 1.0, f'{pct:.0f}%', ha='center', va='bottom',
                       fontsize=7.5, fontweight='bold', color='#333333')
    
    # X axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\n(n={group_ns[g]})' for g in group_labels], fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, zorder=0)
    
    # Title: matching Dosage 8snps style — gene (snp) on line 1, axis on line 2
    dir_text = 'Metabolic Resilience' if direction == '+' else 'Circuit Vulnerability'
    dir_sign = '(+)' if direction == '+' else '(−)'
    ax.set_title(f'{gene} ({snp_id})\n{dir_sign} {dir_text}',
                fontsize=11, fontweight='bold', color='#1F4E79')
    
    # Significance brackets
    pairs = [
        ('SR-MR', 0, 1, info['SR-MR']),
        ('MR-SF', 1, 2, info['MR-SF']),
        ('SR-SF', 0, 2, info['SR-SF']),
    ]
    sig_pairs = [(name, x1, x2, p) for name, x1, x2, p in pairs if p < 0.05]
    sig_pairs.sort(key=lambda x: x[2] - x[1])
    
    max_bar = max(max(freq_data[dv]) for dv in [0, 1, 2])
    bracket_y = max_bar + 8
    
    for name, x1, x2, p in sig_pairs:
        bracket_y = draw_bracket(ax, x1, x2, bracket_y, p, dh=4, barh=2, fontsize=13)
    
    top_y = bracket_y + 5 if sig_pairs else max_bar + 12
    ax.set_ylim(0, max(105, top_y))
    
    # Omnibus stats — G-test only
    gtest_p = info['gtest']
    if gtest_p < 0.05:
        p_str = f'G p={gtest_p:.3f}'
        ax.text(0.97, 0.03, p_str,
               transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
               color='#1F4E79', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Legend
legend_elements = [
    Patch(facecolor=dose_colors[0], edgecolor='white', label='Dosage 0 (Ref/Ref)'),
    Patch(facecolor=dose_colors[1], edgecolor='white', label='Dosage 1 (Ref/Alt)'),
    Patch(facecolor=dose_colors[2], edgecolor='white', label='Dosage 2 (Alt/Alt)'),
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.97, 0.97),
          fontsize=10, framealpha=0.95, edgecolor='#CCCCCC', ncol=3,
          handlelength=1.5, handletextpad=0.5, columnspacing=1.5)

fig.suptitle('Non-Reference Allele Dosage Distribution by Response Group',
            fontsize=17, fontweight='bold', color='#1F4E79', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT}/group_dosage_freq_barplot.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/group_dosage_freq_barplot.pdf', bbox_inches='tight', facecolor='white')
print("Saved group_dosage_freq_barplot.png/pdf")
