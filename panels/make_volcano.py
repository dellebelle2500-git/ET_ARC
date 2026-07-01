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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Data
# ============================================================
dosage_df = pd.read_csv(f'{STATS}/dosage_crst_full_stats.csv')
fisher_df = pd.read_csv(f'{STATS}/group_nominal_stats_final.csv')

dosage_df['dose_sig'] = (
    (dosage_df['LBL_p'] < 0.05).astype(int) +
    (dosage_df['JT_p'] < 0.05).astype(int) +
    (dosage_df['Spearman_p'] < 0.05).astype(int) +
    (dosage_df['OrdLogit_p'] < 0.05).astype(int)
)

fisher_df['grp_sig'] = (
    (fisher_df['Fisher_MC_p'] < 0.05).astype(int) +
    (fisher_df['KW_p'] < 0.05).astype(int) +
    (fisher_df['Gtest_p'] < 0.05).astype(int)
)

m = dosage_df[['Gene','SNP_ID','LBL_r','LBL_p','JT_p','Spearman_p','OrdLogit_p','dose_sig']].merge(
    fisher_df[['Gene','SNP_ID','Fisher_MC_p','KW_p','Gtest_p','grp_sig']],
    on=['Gene','SNP_ID'], how='outer')

m['total_sig'] = m['dose_sig'].fillna(0).astype(int) + m['grp_sig'].fillna(0).astype(int)
m['dose_pass'] = m['dose_sig'] >= 2
m['grp_pass'] = m['grp_sig'] >= 2

def tier(r):
    if r['dose_pass'] and r['grp_pass']: return 'Tier 1'
    elif r['dose_pass']: return 'Tier 2a'
    elif r['grp_pass']: return 'Tier 2b'
    return 'NS'
m['Tier'] = m.apply(tier, axis=1)

m['neg_log10_p'] = -np.log10(m['LBL_p'].clip(lower=1e-10))
df = m.dropna(subset=['LBL_r', 'LBL_p']).copy()

# ============================================================
# Basic Volcano Plot
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor='white')

tier_config = {
    'NS':      {'color': '#CCCCCC', 'size': 25, 'alpha': 0.4, 'zorder': 1, 'edgecolor': 'none'},
    'Tier 2b': {'color': '#ED7D31', 'size': 70, 'alpha': 0.8, 'zorder': 3, 'edgecolor': 'white'},
    'Tier 2a': {'color': '#4472C4', 'size': 70, 'alpha': 0.8, 'zorder': 4, 'edgecolor': 'white'},
    'Tier 1':  {'color': '#2F5496', 'size': 150, 'alpha': 1.0, 'zorder': 5, 'edgecolor': '#FFB800'},
}

for tier_name, cfg in tier_config.items():
    subset = df[df['Tier'] == tier_name]
    ax1.scatter(subset['LBL_r'], subset['neg_log10_p'],
               s=cfg['size'], c=cfg['color'], alpha=cfg['alpha'],
               edgecolors=cfg['edgecolor'], lw=1.5 if tier_name == 'Tier 1' else 0.5,
               zorder=cfg['zorder'])

ax1.axhline(y=-np.log10(0.05), color='#C00000', lw=1, ls='--', alpha=0.6, zorder=0)
ax1.text(ax1.get_xlim()[1] * 0.95, -np.log10(0.05) + 0.03, 'p = 0.05',
        ha='right', fontsize=9, color='#C00000', style='italic')
ax1.axvline(x=0, color='#999999', lw=0.8, ls='-', alpha=0.3, zorder=0)

# Labels for significant SNPs
labeled = df[df['Tier'] != 'NS'].copy()
for _, r in labeled.iterrows():
    tier_name = r['Tier']
    fontsize = 10 if tier_name == 'Tier 1' else 8
    fontweight = 'bold' if tier_name in ['Tier 1', 'Tier 2a'] else 'normal'
    color = tier_config[tier_name]['color']
    x_off = 0.02 if r['LBL_r'] > 0 else -0.02
    ha = 'left' if r['LBL_r'] > 0 else 'right'
    ax1.annotate(r['Gene'], xy=(r['LBL_r'], r['neg_log10_p']),
                xytext=(x_off, 0.06), textcoords='offset fontsize',
                fontsize=fontsize, fontweight=fontweight, color=color,
                ha=ha, va='bottom')

ax1.set_xlabel('LBL Effect Size (r)', fontsize=12, fontweight='bold')
ax1.set_ylabel('\u2212log\u2081\u2080(LBL p-value)', fontsize=12, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(alpha=0.15, zorder=0)

# UPDATED quadrant labels
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
# Pad x-axis
ax1.set_xlim(xlim[0] - 0.08, xlim[1] + 0.08)
xlim = ax1.get_xlim()
ax1.text(xlim[0] + 0.03, ylim[1] * 0.95, 'Negative correlation\n(Circuit Vulnerability)',
        fontsize=10, color='#C00000', alpha=0.5, ha='left', va='top', style='italic')
ax1.text(xlim[1] - 0.03, ylim[1] * 0.95, 'Positive correlation\n(Metabolic Resilience)',
        fontsize=10, color='#2F5496', alpha=0.5, ha='right', va='top', style='italic')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2F5496', markersize=14,
           markeredgecolor='#FFB800', markeredgewidth=2, label='\u2605 Tier 1 (Cross-validated)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4472C4', markersize=10,
           label='\u25cf Tier 2a (Dosage-only)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ED7D31', markersize=10,
           label='\u25c6 Tier 2b (Group-only)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC', markersize=7,
           label='NS (not significant)'),
]
ax1.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')

fig1.suptitle('Volcano Plot: SNP Dosage vs CRST Improvement',
             fontsize=15, fontweight='bold', color='#1F4E79', y=0.97)
fig1.text(0.5, 0.93, 'Linear-by-Linear Association  |  183 SNPs  |  N = 27 ET patients',
         ha='center', fontsize=10, color='#666666', style='italic')

fig1.savefig(f'{OUT}/volcano_basic.png', dpi=300, bbox_inches='tight', facecolor='white')
fig1.savefig(f'{OUT}/volcano_basic.pdf', bbox_inches='tight', facecolor='white')
print("Basic volcano saved")
