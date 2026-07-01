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
import matplotlib.patches as mpatches
from scipy.stats import kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
df['name'] = df['name'].str.strip()

# Group name mapping
group_names = {
    1: 'Mild-Responsive',    # moderate responder
    2: 'Severe-Responsive',  # high responder
    3: 'Severe-Refractory',  # low responder
}
group_short = {1: 'MR', 2: 'SR', 3: 'SF'}

clusters = sorted(df['Cluster'].unique())
display_order = [2, 1, 3]  # SR, MR, SF
cluster_n = {c: (df['Cluster'] == c).sum() for c in clusters}

colors = {1: '#4DAF4A', 2: '#2166AC', 3: '#D6604D'}  # MR=green, SR=blue, SF=red

vars_info = [
    ('Base_CRST', 'CRST Baseline'),
    ('Base_QUEST', 'QUEST Baseline'),
    ('Imp_CRST', 'CRST Improvement (%)'),
    ('Imp_QUEST', 'QUEST Improvement (%)'),
]

# Pre-compute stats
stats_results = {}
for col, display in vars_info:
    groups = {c: df[df['Cluster'] == c][col].dropna() for c in clusters}
    kw_stat, kw_p = kruskal(*[groups[c] for c in clusters])
    
    pairwise = {}
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            stat, p = mannwhitneyu(groups[c1], groups[c2], alternative='two-sided')
            pairwise[(c1, c2)] = p
    
    stats_results[col] = {'kw_p': kw_p, 'pairwise': pairwise}

def sig_str(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''

fig, axes = plt.subplots(1, 4, figsize=(18, 6), facecolor='white')

positions = [1, 2, 3]
box_width = 0.55

for idx, (col, display) in enumerate(vars_info):
    ax = axes[idx]
    groups = {c: df[df['Cluster'] == c][col].dropna() for c in clusters}
    kw_p = stats_results[col]['kw_p']
    pairwise = stats_results[col]['pairwise']
    
    # Box plots in SR, MR, SF order
    bp_data = [groups[c].values for c in display_order]
    bp = ax.boxplot(bp_data, positions=positions, widths=box_width,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#333333', linewidth=1.5),
                    whiskerprops=dict(color='#666666'),
                    capprops=dict(color='#666666'))
    
    for patch, c in zip(bp['boxes'], display_order):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.45)
        patch.set_edgecolor(colors[c])
    
    # Strip (jittered individual points)
    rng = np.random.default_rng(42)
    for i, c in enumerate(display_order):
        vals = groups[c].values
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(positions[i] + jitter, vals, c=colors[c], s=35, alpha=0.7,
                  edgecolors='white', lw=0.5, zorder=5)
    
    # Significance brackets
    all_data = pd.concat([groups[c] for c in clusters])
    y_range = all_data.max() - all_data.min()
    y_max = all_data.max()
    
    sig_pairs = [(c1, c2, p) for (c1, c2), p in pairwise.items() if p < 0.05]
    sig_pairs.sort(key=lambda x: abs(display_order.index(x[0]) - display_order.index(x[1])))
    
    for bi, (c1, c2, p) in enumerate(sig_pairs):
        x1, x2 = positions[display_order.index(c1)], positions[display_order.index(c2)]
        y_bar = y_max + y_range * (0.08 + bi * 0.12)
        h = y_range * 0.02
        
        ax.plot([x1, x1, x2, x2], [y_bar - h, y_bar, y_bar, y_bar - h],
               color='#333333', lw=1.2)
        ax.text((x1 + x2) / 2, y_bar + h * 0.5, sig_str(p),
               ha='center', va='bottom', fontsize=14, fontweight='bold', color='#333333')
    
    # Title with KW p-value
    kw_str = 'p < 0.001' if kw_p < 0.001 else f'p = {kw_p:.4f}'
    ax.set_title(f'{display}\nKruskal-Wallis {kw_str}', fontsize=12, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{group_short[c]}\n(n={cluster_n[c]})' for c in display_order], fontsize=10)
    ax.set_ylabel(display, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    n_sig = len(sig_pairs)
    top_margin = y_range * (0.15 + n_sig * 0.15)
    ax.set_ylim(all_data.min() - y_range * 0.05, y_max + top_margin)

# Legend in SR, MR, SF order
patches = [mpatches.Patch(facecolor=colors[c], edgecolor=colors[c], alpha=0.5,
            label=f'{group_short[c]}: {group_names[c]} (n={cluster_n[c]})')
           for c in display_order]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=11, frameon=True,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Ward Linkage k=3: Cluster Comparison', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(f'{OUT}/ward_k3_boxstrip.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/ward_k3_boxstrip.pdf', bbox_inches='tight', facecolor='white')
print("Saved")
