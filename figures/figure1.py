"""
Figure 1. ARC treatment efficacy and data-driven stratification of response subgroups.
  a: CRST total-score spaghetti (paired t-test brackets)
  b: QUEST total-score spaghetti
  c: PCA of clinical-outcome space (Ward k=3 clusters)
  d: Ward-linkage dendrogram
  e: CRST/QUEST time-courses by subgroup (mean +/- SEM)
  f: Baseline severity & 3-month improvement by subgroup (KW; pairwise MWU)
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
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_rel, kruskal, mannwhitneyu
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

MM = 1 / 25.4
FS_PANEL = 13
FS_TITLE = 8
FS_AXIS = 7
FS_TICK = 6
FS_LEG = 6.5
FS_STAR = 8

# Colors: SR blue, MR green, SF red
GC = {'SR': '#2166AC', 'MR': '#4DAF4A', 'SF': '#D6604D'}
GM = {'SR': 'o', 'MR': 's', 'SF': 'D'}
GLAB = {'SR': 'Severe-Responsive (SR, n=6)', 'MR': 'Mild-Responsive (MR, n=11)', 'SF': 'Severe-Refractory (SF, n=10)'}

# ============================================================
# Data
# ============================================================
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')
cl['name'] = cl['name'].str.strip()
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)
groups = cl['group'].values

def parse_series(path, val_col):
    df = pd.read_csv(path, encoding='utf-8')
    data, cur = {}, None
    for _, row in df.iterrows():
        if pd.notna(row['Unnamed: 0']):
            cur = str(row['Unnamed: 0']).strip(); data[cur] = {}
        if cur and pd.notna(row['Unnamed: 1']):
            data[cur][str(row['Unnamed: 1']).strip()] = row[val_col]
    return data

crst_s = parse_series(f'{DATA}/CRST.csv', 'total')
quest_s = parse_series(f'{DATA}/QUEST.csv', 'Questionnaire_sum')

tps = ['baseline', '1month', '2month', '3month']
tp_labels = ['Pre', '1m', '2m', '3m']
x = np.arange(4)

# PCA
Xf = cl[['Base_CRST', 'Base_QUEST', 'Imp_CRST', 'Imp_QUEST']].values
Xs = StandardScaler().fit_transform(Xf)
pca = PCA(n_components=2).fit(Xs)
pcs = pca.transform(Xs)
pc1, pc2 = pcs[:, 0], pcs[:, 1]
ev = pca.explained_variance_ratio_
Z = linkage(Xs, method='ward')

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

# ============================================================
# Canvas: 4 subfigure rows
# ============================================================
fig = plt.figure(figsize=(180 * MM, 222 * MM), facecolor='white')
sfs = fig.subfigures(4, 1, height_ratios=[1.0, 1.0, 0.95, 0.95], hspace=0.04)

# ------------------------------------------------------------
# Row 1: a (CRST spaghetti), b (QUEST spaghetti)
# ------------------------------------------------------------
sf1 = sfs[0]
ax_a, ax_b = sf1.subplots(1, 2)
sf1.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.16, wspace=0.24)

def spaghetti(ax, series, ylabel, title):
    pts = list(series.keys())
    mat = np.array([[series[p].get(tp, np.nan) for tp in tps] for p in pts])
    for row in mat:
        ax.plot(x, row, color='#BBBBBB', alpha=0.4, lw=0.5, zorder=1)
    means = np.nanmean(mat, axis=0)
    nvalid = np.sum(~np.isnan(mat), axis=0)
    sems = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(nvalid)
    ax.errorbar(x, means, yerr=sems, color='#C00000', marker='o', markersize=4,
                lw=1.8, capsize=2.5, capthick=1, zorder=5, markeredgecolor='white', markeredgewidth=0.5)
    # pairwise paired t-test brackets
    pairs = []
    for i, j in combinations(range(4), 2):
        a, b = mat[:, i], mat[:, j]
        msk = ~(np.isnan(a) | np.isnan(b))
        _, p = ttest_rel(a[msk], b[msk])
        if p < 0.05:
            pairs.append((i, j, p))
    pairs.sort(key=lambda z: (z[1] - z[0], z[0]))
    ytop = np.nanmax(mat)
    yr = np.nanmax(mat) - np.nanmin(mat)
    lvl = ytop + yr * 0.06
    step = yr * 0.11
    for i, j, p in pairs:
        ax.plot([i, i, j, j], [lvl, lvl + yr*0.02, lvl + yr*0.02, lvl], color='#333333', lw=0.8, clip_on=False)
        ax.text((i + j) / 2, lvl + yr*0.025, stars(p), ha='center', va='bottom',
                fontsize=FS_STAR, fontweight='bold', color='#333333', clip_on=False)
        lvl += step
    ax.set_xticks(x); ax.set_xticklabels(tp_labels, fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS, fontweight='bold')
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', color='#1F4E79', pad=3)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(top=lvl + yr*0.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

spaghetti(ax_a, crst_s, 'CRST Total Score', 'CRST Total Score')
spaghetti(ax_b, quest_s, 'QUEST Total Score', 'QUEST Total Score (QoL)')

# ------------------------------------------------------------
# Row 2: c (PCA), d (dendrogram)
# ------------------------------------------------------------
sf2 = sfs[1]
ax_c, ax_d = sf2.subplots(1, 2)
sf2.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.16, wspace=0.24)

# c: PCA scatter
for i in range(len(cl)):
    g = groups[i]
    ax_c.scatter(pc1[i], pc2[i], c=GC[g], s=32, marker=GM[g], edgecolors='#333333', lw=0.6, zorder=5)
ax_c.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=FS_AXIS, fontweight='bold')
ax_c.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=FS_AXIS, fontweight='bold')
ax_c.set_title('Clinical-Outcome PCA (Ward k=3)', fontsize=FS_TITLE, fontweight='bold', color='#1F4E79', pad=3)
ax_c.tick_params(labelsize=FS_TICK)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(alpha=0.15)
leg_c = [Line2D([0], [0], marker=GM[g], color='w', markerfacecolor=GC[g], markersize=5.5,
                markeredgecolor='#333', markeredgewidth=0.7, label=g) for g in ['SR', 'MR', 'SF']]
ax_c.legend(handles=leg_c, loc='best', fontsize=FS_LEG, framealpha=0.9, edgecolor='#CCC')

# d: dendrogram
dn = dendrogram(Z, ax=ax_d, no_labels=True, above_threshold_color='#888888', color_threshold=0)
leaf_order = dn['leaves']
leaf_groups = [groups[k] for k in leaf_order]
xpos = np.arange(5, 10 * len(leaf_order), 10)
ylim_d = ax_d.get_ylim()
barh = ylim_d[1] * 0.045
bar_y = -barh * 1.4
for xp, g in zip(xpos, leaf_groups):
    ax_d.bar(xp, barh, bottom=bar_y, width=8, color=GC[g], edgecolor='none', zorder=10, clip_on=False)
# group name runs
runs, cg, s0 = [], leaf_groups[0], 0
for k in range(1, len(leaf_groups)):
    if leaf_groups[k] != cg:
        runs.append((cg, s0, k - 1)); cg = leaf_groups[k]; s0 = k
runs.append((cg, s0, len(leaf_groups) - 1))
for g, s0, e0 in runs:
    ax_d.text((xpos[s0] + xpos[e0]) / 2, bar_y - barh * 1.2, g, ha='center', va='top',
              fontsize=FS_LEG, fontweight='bold', color=GC[g], clip_on=False)
ax_d.set_ylabel('Ward Distance', fontsize=FS_AXIS, fontweight='bold')
ax_d.set_title('Hierarchical Clustering (Ward)', fontsize=FS_TITLE, fontweight='bold', color='#1F4E79', pad=3)
ax_d.set_xticks([])
ax_d.tick_params(labelsize=FS_TICK)
ax_d.set_ylim(bar_y - barh * 3, ylim_d[1])
for sp in ['top', 'right', 'bottom']:
    ax_d.spines[sp].set_visible(False)

# ------------------------------------------------------------
# Row 3: e (timecourse CRST + QUEST)
# ------------------------------------------------------------
sf3 = sfs[2]
ax_e1, ax_e2 = sf3.subplots(1, 2)
sf3.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.17, wspace=0.24)

def timecourse(ax, series, ylabel, title):
    # build per-patient arrays keyed by group
    pdata = {g: [] for g in ['SR', 'MR', 'SF']}
    for _, r in cl.iterrows():
        nm, g = r['name'], r['group']
        vals = [series.get(nm, {}).get(tp, np.nan) for tp in tps]
        pdata[g].append(vals)
        ax.plot(x, vals, color=GC[g], alpha=0.12, lw=0.5, zorder=1)
    for g in ['SR', 'MR', 'SF']:
        arr = np.array(pdata[g], dtype=float)
        means = np.nanmean(arr, axis=0)
        nvalid = np.sum(~np.isnan(arr), axis=0)
        sems = np.where(nvalid > 1, np.nanstd(arr, axis=0, ddof=1) / np.sqrt(nvalid), 0)
        ax.errorbar(x, means, yerr=sems, color=GC[g], marker=GM[g], markersize=4, lw=1.6,
                    capsize=2, capthick=0.8, zorder=5, markeredgecolor='white', markeredgewidth=0.5, label=g)
        ax.fill_between(x, means - sems, means + sems, color=GC[g], alpha=0.08, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(tp_labels, fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS, fontweight='bold')
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', color='#1F4E79', pad=3)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

timecourse(ax_e1, crst_s, 'CRST Total Score', 'CRST by Response Group')
timecourse(ax_e2, quest_s, 'QUEST Total Score', 'QUEST by Response Group')
ax_e1.legend(loc='upper right', fontsize=FS_LEG, framealpha=0.9, edgecolor='#CCC')

# ------------------------------------------------------------
# Row 4: f (ward boxstrip, 4 metrics)
# ------------------------------------------------------------
sf4 = sfs[3]
axes_f = sf4.subplots(1, 4)
sf4.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.17, wspace=0.34)

clusters = sorted(cl['Cluster'].unique())
display_order = [2, 1, 3]  # SR, MR, SF
cn = {c: (cl['Cluster'] == c).sum() for c in clusters}
cshort = {1: 'MR', 2: 'SR', 3: 'SF'}
ccolor = {1: GC['MR'], 2: GC['SR'], 3: GC['SF']}
vars_info = [('Base_CRST', 'CRST Baseline'), ('Base_QUEST', 'QUEST Baseline'),
             ('Imp_CRST', 'CRST Imp. (%)'), ('Imp_QUEST', 'QUEST Imp. (%)')]
positions = [0, 1, 2]

for ai, (col, disp) in enumerate(vars_info):
    ax = axes_f[ai]
    grp = {c: cl[cl['Cluster'] == c][col].dropna() for c in clusters}
    _, kw_p = kruskal(*[grp[c] for c in clusters])
    pw = {}
    for c1, c2 in combinations(clusters, 2):
        _, p = mannwhitneyu(grp[c1], grp[c2], alternative='two-sided')
        pw[(c1, c2)] = p
    bp = ax.boxplot([grp[c].values for c in display_order], positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#333333', linewidth=1.0),
                    whiskerprops=dict(color='#666666', linewidth=0.7),
                    capprops=dict(color='#666666', linewidth=0.7), boxprops=dict(linewidth=0.7))
    for patch, c in zip(bp['boxes'], display_order):
        patch.set_facecolor(ccolor[c]); patch.set_alpha(0.45); patch.set_edgecolor(ccolor[c])
    rng = np.random.default_rng(42)
    for i, c in enumerate(display_order):
        vals = grp[c].values
        ax.scatter(positions[i] + rng.uniform(-0.15, 0.15, len(vals)), vals, c=ccolor[c],
                   s=9, alpha=0.7, edgecolors='white', lw=0.3, zorder=5)
    alldata = pd.concat([grp[c] for c in clusters])
    yr = alldata.max() - alldata.min(); ymax = alldata.max()
    sig = [(c1, c2, p) for (c1, c2), p in pw.items() if p < 0.05]
    sig.sort(key=lambda z: abs(display_order.index(z[0]) - display_order.index(z[1])))
    for bi, (c1, c2, p) in enumerate(sig):
        x1, x2 = positions[display_order.index(c1)], positions[display_order.index(c2)]
        yb = ymax + yr * (0.08 + bi * 0.14); h = yr * 0.02
        ax.plot([x1, x1, x2, x2], [yb - h, yb, yb, yb - h], color='#333333', lw=0.8, clip_on=False)
        ax.text((x1 + x2) / 2, yb, stars(p), ha='center', va='bottom', fontsize=FS_STAR,
                fontweight='bold', color='#333333', clip_on=False)
    kw_str = 'p<0.001' if kw_p < 0.001 else f'p={kw_p:.3f}'
    ax.set_title(f'{disp}\nKW {kw_str}', fontsize=FS_TITLE - 1, fontweight='bold', color='#1F4E79', pad=3, linespacing=1.1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{cshort[c]}\n({cn[c]})' for c in display_order], fontsize=FS_TICK, fontweight='bold')
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ns = len(sig)
    ax.set_ylim(alldata.min() - yr * 0.05, ymax + yr * (0.15 + ns * 0.16))

# ------------------------------------------------------------
# Panel letters
# ------------------------------------------------------------
sf1.text(0.004, 0.99, 'a', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sf1.text(0.505, 0.99, 'b', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sf2.text(0.004, 0.99, 'c', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sf2.text(0.505, 0.99, 'd', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sf3.text(0.004, 0.99, 'e', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')
sf4.text(0.004, 0.99, 'f', fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')

fig.savefig(f'{OUT}/Figure1.pdf', bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/Figure1.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved Figure1.pdf/png")
