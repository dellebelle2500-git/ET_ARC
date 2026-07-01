"""
Figure 2. Dual-framework pharmacogenomic discovery of treatment-response loci.
  a: CRST improvement (%) by non-ref allele dosage (8 loci, 2x4)
  b: Non-ref allele dosage distribution across response subgroups (8 loci, 2x4)
  c: Volcano plot (183 SNPs, LBL effect size vs -log10 p)
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

MM = 1 / 25.4

# Font sizes
FS_PANEL = 13
FS_TITLE = 6.0
FS_AXIS = 6.5
FS_TICK = 6.0
FS_ANNOT = 6.0
FS_LEG = 7.0
FS_STAR = 9.0

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)
dos = pd.read_csv(f'{DATA}/snp_dosage_matrix_final.csv', encoding='cp949')
patient_ids = [str(pid) for pid in cl['patient_id']]
improvement = cl['Imp_CRST'].values
groups_arr = cl['group'].values

def get_dosage(gene, snp_id):
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    return row[patient_ids].values.flatten().astype(float) if len(row) else np.zeros(len(patient_ids))

# ============================================================
# Figure canvas: 3 stacked subfigures
# ============================================================
fig = plt.figure(figsize=(180 * MM, 235 * MM), facecolor='white')
subfigs = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.03)

# ------------------------------------------------------------
# Panel a: dosage boxplots (2x4)
# ------------------------------------------------------------
sfa = subfigs[0]
axes_a = sfa.subplots(2, 4)
sfa.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.11, hspace=0.85, wspace=0.42)
axes_a = axes_a.flatten()

snps_raw = [
    ('MSRB3','rs10878269','+'), ('OR7G3','rs10414255','+'), ('NEIL3-AGA','rs1395479','+'),
    ('CPLX3-ULK3','rs6495122','−'), ('RBFOX1','rs3095508','−'), ('GC','rs2282679','−'),
    ('EMX2OS-RAB11FIP2','rs10886142','−'), ('PKD1L2','rs8044334','−'),
]
snps_with_p = [(g, s, d, spearmanr(get_dosage(g, s), improvement)[1]) for g, s, d in snps_raw]
pos_sorted = sorted([x for x in snps_with_p if x[2] == '+'], key=lambda z: z[3])
neg_sorted = sorted([x for x in snps_with_p if x[2] == '−'], key=lambda z: z[3])
snps_a = [(g, s, d) for g, s, d, _ in pos_sorted + neg_sorted]

dose_colors = {0: '#4A90D9', 1: '#F5A623', 2: '#D0021B'}

for idx, (gene, snp_id, direction) in enumerate(snps_a):
    ax = axes_a[idx]
    dosage = get_dosage(gene, snp_id).astype(int)
    grp = {d: improvement[dosage == d] for d in [0, 1, 2] if (dosage == d).sum() > 0}
    avail = sorted(grp.keys())
    positions = list(range(len(avail)))
    bp = ax.boxplot([grp[d] for d in avail], positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#333333', linewidth=1.0),
                    whiskerprops=dict(color='#666666', linewidth=0.7),
                    capprops=dict(color='#666666', linewidth=0.7),
                    boxprops=dict(linewidth=0.7))
    for patch, d in zip(bp['boxes'], avail):
        patch.set_facecolor(dose_colors[d]); patch.set_alpha(0.35); patch.set_edgecolor(dose_colors[d])
    rng = np.random.default_rng(42 + idx)
    for i, d in enumerate(avail):
        vals = grp[d]
        ax.scatter(positions[i] + rng.uniform(-0.12, 0.12, len(vals)), vals,
                   c=dose_colors[d], s=9, alpha=0.75, edgecolors='white', lw=0.3, zorder=5)
    rho, p = spearmanr(dosage, improvement)
    p_str = f'{p:.1e}' if p < 0.001 else f'{p:.3f}'
    dir_text = 'Metabolic Resilience' if direction == '+' else 'Circuit Vulnerability'
    dir_sign = '(+)' if direction == '+' else '(−)'
    ax.set_title(f'{gene} ({snp_id})\n{dir_sign} {dir_text}', fontsize=FS_TITLE,
                 fontweight='bold', color='#1F4E79', pad=3, linespacing=1.1)
    # place rho box in empty corner: + gene -> bottom-right, - gene -> top-right
    if direction == '+':
        bxy, bva = 0.04, 'bottom'
    else:
        bxy, bva = 0.96, 'top'
    ax.text(0.96, bxy, f'\u03c1 = {rho:+.3f}\np = {p_str}', transform=ax.transAxes,
            fontsize=FS_ANNOT, ha='right', va=bva, color='#1F4E79', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{d}\n(n={len(grp[d])})' for d in avail], fontsize=FS_TICK)
    if idx % 4 == 0:
        ax.set_ylabel('CRST Imp. (%)', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    _ir = improvement.max() - improvement.min()
    ax.set_ylim(improvement.min() - 0.10 * _ir, improvement.max() + 0.16 * _ir)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# dosage legend for panel a
leg_a = [Patch(facecolor=dose_colors[0], alpha=0.6, label='Dosage 0'),
         Patch(facecolor=dose_colors[1], alpha=0.6, label='Dosage 1'),
         Patch(facecolor=dose_colors[2], alpha=0.6, label='Dosage 2')]
sfa.legend(handles=leg_a, loc='upper right', ncol=3, fontsize=FS_LEG, frameon=False,
           bbox_to_anchor=(0.99, 1.0), handlelength=1.2, handletextpad=0.3, columnspacing=1.0)

# ------------------------------------------------------------
# Panel b: group barplots (2x4)
# ------------------------------------------------------------
sfb = subfigs[1]
axes_b = sfb.subplots(2, 4)
sfb.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.10, hspace=0.85, wspace=0.42)
axes_b = axes_b.flatten()

group_labels = ['SR', 'MR', 'SF']
group_ns = {'SR': 6, 'MR': 11, 'SF': 10}

target_snps = [
    ('MSRB3','rs10878269','+'), ('MPP7','rs1937810','+'), ('FAM185A','rs3972456','+'),
    ('PKD2L1','rs603424','+'), ('TGFA','rs3771501','+'),
    ('MPPED2-DCDC1','rs963837','−'), ('DEF8','rs4268748','−'), ('ZEB2','rs10427255','−'),
]
stats_info = {
    'MSRB3':{'gtest':0.030,'SR-MR':0.270,'SR-SF':0.298,'MR-SF':0.015},
    'MPP7':{'gtest':0.009,'SR-MR':0.004,'SR-SF':0.138,'MR-SF':0.080},
    'MPPED2-DCDC1':{'gtest':0.015,'SR-MR':0.005,'SR-SF':0.138,'MR-SF':0.207},
    'DEF8':{'gtest':0.037,'SR-MR':1.000,'SR-SF':0.173,'MR-SF':0.063},
    'FAM185A':{'gtest':0.011,'SR-MR':0.015,'SR-SF':0.147,'MR-SF':0.765},
    'PKD2L1':{'gtest':0.016,'SR-MR':0.116,'SR-SF':1.000,'MR-SF':0.044},
    'TGFA':{'gtest':0.009,'SR-MR':0.776,'SR-SF':0.519,'MR-SF':0.516},
    'ZEB2':{'gtest':0.002,'SR-MR':0.218,'SR-SF':0.193,'MR-SF':0.175},
}
def min_p_b(g): return stats_info[g]['gtest']
pos_b = sorted([x for x in target_snps if x[2] == '+'], key=lambda z: min_p_b(z[0]))
neg_b = sorted([x for x in target_snps if x[2] == '−'], key=lambda z: min_p_b(z[0]))
snps_b = pos_b + neg_b

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

for idx, (gene, snp_id, direction) in enumerate(snps_b):
    ax = axes_b[idx]
    row = dos[(dos['gene'] == gene) & (dos['SNP ID'] == snp_id)]
    dosages = row[patient_ids].values.flatten().astype(float)
    grp_dos = {g: dosages[groups_arr == g] for g in group_labels}
    info = stats_info[gene]
    x_pos = np.arange(3)
    bw = 0.25
    freq = {dv: [int(np.sum(grp_dos[g] == dv)) / group_ns[g] * 100 for g in group_labels] for dv in [0, 1, 2]}
    for di, dv in enumerate([0, 1, 2]):
        off = (di - 1) * bw
        ax.bar(x_pos + off, freq[dv], bw * 0.85, color=dose_colors[dv], edgecolor='white', lw=0.5, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}\n({group_ns[g]})' for g in group_labels], fontsize=FS_TICK, fontweight='bold')
    if idx % 4 == 0:
        ax.set_ylabel('Frequency (%)', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, zorder=0)
    dir_text = 'Metabolic Resilience' if direction == '+' else 'Circuit Vulnerability'
    dir_sign = '(+)' if direction == '+' else '(−)'
    ax.set_title(f'{gene} ({snp_id})\n{dir_sign} {dir_text}', fontsize=FS_TITLE,
                 fontweight='bold', color='#1F4E79', pad=3, linespacing=1.1)
    pairs = [('SR-MR', 0, 1, info['SR-MR']), ('MR-SF', 1, 2, info['MR-SF']), ('SR-SF', 0, 2, info['SR-SF'])]
    sig = sorted([(x1, x2, p) for _, x1, x2, p in pairs if p < 0.05], key=lambda z: z[1] - z[0])
    max_bar = max(max(freq[dv]) for dv in [0, 1, 2])
    by = max_bar + 8
    for x1, x2, p in sig:
        ax.plot([x1, x1, x2, x2], [by, by + 3, by + 3, by], color='#333333', lw=0.9, clip_on=False)
        ax.text((x1 + x2) / 2, by + 3.5, stars(p), ha='center', va='bottom',
                fontsize=FS_STAR, fontweight='bold', color='#333333', clip_on=False)
        by += 12
    top_b = max(118, by + 16)
    ax.set_ylim(0, top_b)
    if info['gtest'] < 0.05:
        ax.text(0.96, 0.97, f"G p={info['gtest']:.3f}", transform=ax.transAxes,
                fontsize=FS_ANNOT, ha='right', va='top', color='#1F4E79', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

leg_b = [Patch(facecolor=dose_colors[0], label='Dosage 0'),
         Patch(facecolor=dose_colors[1], label='Dosage 1'),
         Patch(facecolor=dose_colors[2], label='Dosage 2')]
sfb.legend(handles=leg_b, loc='upper right', ncol=3, fontsize=FS_LEG, frameon=False,
           bbox_to_anchor=(0.99, 1.0), handlelength=1.2, handletextpad=0.3, columnspacing=1.0)

# ------------------------------------------------------------
# Panel c: volcano
# ------------------------------------------------------------
sfc = subfigs[2]
ax_c = sfc.subplots(1, 1)
sfc.subplots_adjust(left=0.16, right=0.84, top=0.94, bottom=0.13)

dosage_df = pd.read_csv(f'{STATS}/dosage_crst_full_stats.csv')
fisher_df = pd.read_csv(f'{STATS}/group_nominal_stats_final.csv')
dosage_df['dose_sig'] = ((dosage_df['LBL_p'] < 0.05).astype(int) + (dosage_df['JT_p'] < 0.05).astype(int)
                         + (dosage_df['Spearman_p'] < 0.05).astype(int) + (dosage_df['OrdLogit_p'] < 0.05).astype(int))
fisher_df['grp_sig'] = ((fisher_df['Fisher_MC_p'] < 0.05).astype(int) + (fisher_df['KW_p'] < 0.05).astype(int)
                        + (fisher_df['Gtest_p'] < 0.05).astype(int))
m = dosage_df[['Gene','SNP_ID','LBL_r','LBL_p','dose_sig']].merge(
    fisher_df[['Gene','SNP_ID','grp_sig']], on=['Gene','SNP_ID'], how='outer')
m['dose_pass'] = m['dose_sig'].fillna(0) >= 2
m['grp_pass'] = m['grp_sig'].fillna(0) >= 2
def tier(r):
    if r['dose_pass'] and r['grp_pass']: return 'Tier 1'
    if r['dose_pass']: return 'Tier 2a'
    if r['grp_pass']: return 'Tier 2b'
    return 'NS'
m['Tier'] = m.apply(tier, axis=1)
m['neg_log10_p'] = -np.log10(m['LBL_p'].clip(lower=1e-10))
dfv = m.dropna(subset=['LBL_r', 'LBL_p']).copy()

tier_cfg = {
    'NS':      {'color': '#CCCCCC', 'size': 10, 'alpha': 0.4, 'zorder': 1, 'ec': 'none',    'lw': 0},
    'Tier 2b': {'color': '#ED7D31', 'size': 28, 'alpha': 0.85,'zorder': 3, 'ec': 'white',   'lw': 0.3},
    'Tier 2a': {'color': '#4472C4', 'size': 28, 'alpha': 0.85,'zorder': 4, 'ec': 'white',   'lw': 0.3},
    'Tier 1':  {'color': '#2F5496', 'size': 60, 'alpha': 1.0, 'zorder': 5, 'ec': '#FFB800', 'lw': 1.2},
}
for tname, c in tier_cfg.items():
    sub = dfv[dfv['Tier'] == tname]
    ax_c.scatter(sub['LBL_r'], sub['neg_log10_p'], s=c['size'], c=c['color'],
                 alpha=c['alpha'], edgecolors=c['ec'], lw=c['lw'], zorder=c['zorder'])
ax_c.axhline(y=-np.log10(0.05), color='#C00000', lw=0.8, ls='--', alpha=0.6, zorder=0)
ax_c.axvline(x=0, color='#999999', lw=0.6, ls='-', alpha=0.3, zorder=0)

labeled = dfv[dfv['Tier'] != 'NS'].copy()
for _, r in labeled.iterrows():
    fw = 'bold' if r['Tier'] in ['Tier 1', 'Tier 2a'] else 'normal'
    xo = 0.02 if r['LBL_r'] > 0 else -0.02
    ha = 'left' if r['LBL_r'] > 0 else 'right'
    ax_c.annotate(r['Gene'], xy=(r['LBL_r'], r['neg_log10_p']), xytext=(xo, 0.05),
                  textcoords='offset fontsize', fontsize=5.5, fontweight=fw,
                  color=tier_cfg[r['Tier']]['color'], ha=ha, va='bottom')

ax_c.set_xlabel('LBL Effect Size (r)', fontsize=FS_AXIS + 0.5, fontweight='bold')
ax_c.set_ylabel('\u2212log\u2081\u2080(LBL p-value)', fontsize=FS_AXIS + 0.5, fontweight='bold')
ax_c.tick_params(labelsize=FS_TICK)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(alpha=0.15, zorder=0)
xl = ax_c.get_xlim()
ax_c.set_xlim(xl[0] - 0.08, xl[1] + 0.08)
xl = ax_c.get_xlim(); yl = ax_c.get_ylim()
ax_c.text(xl[0] + 0.03, yl[1] * 0.96, 'Negative correlation\n(Circuit Vulnerability)',
          fontsize=FS_AXIS, color='#C00000', alpha=0.55, ha='left', va='top', style='italic')
ax_c.text(xl[1] - 0.03, yl[1] * 0.96, 'Positive correlation\n(Metabolic Resilience)',
          fontsize=FS_AXIS, color='#2F5496', alpha=0.55, ha='right', va='top', style='italic')
ax_c.text(xl[1] - 0.02, -np.log10(0.05) - 0.10, 'p = 0.05', fontsize=FS_ANNOT,
          color='#C00000', style='italic', ha='right', va='top')

leg_c = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2F5496', markersize=8,
           markeredgecolor='#FFB800', markeredgewidth=1.5, label='Tier 1 (cross-validated)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4472C4', markersize=6, label='Tier 2a (dosage-only)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ED7D31', markersize=6, label='Tier 2b (group-only)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC', markersize=4.5, label='NS'),
]
ax_c.legend(handles=leg_c, loc='lower left', fontsize=FS_LEG - 0.5, framealpha=0.95, edgecolor='#CCCCCC')

# ---------- Panel letters ----------
sfa.text(0.004, 0.99, 'a', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sfb.text(0.004, 0.99, 'b', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sfc.text(0.004, 0.99, 'c', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')

fig.savefig(f'{OUT}/Figure2.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/Figure2.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved Figure2.pdf/png")
