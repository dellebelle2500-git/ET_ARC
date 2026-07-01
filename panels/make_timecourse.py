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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load data
# ============================================================
crst = pd.read_csv(f'{DATA}/CRST.csv', encoding='utf-8')
quest = pd.read_csv(f'{DATA}/QUEST.csv', encoding='utf-8')
cl = pd.read_csv(f'{DATA}/patients_clustering.csv', encoding='cp949')

# Map response groups
response_map = {'high': 'SR', 'moderate': 'MR', 'low': 'SF'}
cl['group'] = cl['response'].map(response_map)
cl['name'] = cl['name'].str.strip()

# ============================================================
# Parse CRST time series (4 timepoints per patient)
# ============================================================
crst_data = {}
current_name = None
for _, row in crst.iterrows():
    if pd.notna(row['Unnamed: 0']):
        current_name = str(row['Unnamed: 0']).strip()
        crst_data[current_name] = {}
    if current_name and pd.notna(row['Unnamed: 1']):
        tp = str(row['Unnamed: 1']).strip()
        crst_data[current_name][tp] = row['total']

# Parse QUEST time series
quest_data = {}
current_name = None
for _, row in quest.iterrows():
    if pd.notna(row['Unnamed: 0']):
        current_name = str(row['Unnamed: 0']).strip()
        quest_data[current_name] = {}
    if current_name and pd.notna(row['Unnamed: 1']):
        tp = str(row['Unnamed: 1']).strip()
        quest_data[current_name][tp] = row['Questionnaire_sum']

timepoints = ['baseline', '1month', '2month', '3month']
tp_labels = ['Baseline', '1 Month', '2 Months', '3 Months']
x = np.arange(len(timepoints))

# Build patient arrays
patients = []
for _, row in cl.iterrows():
    name = row['name']
    group = row['group']
    
    crst_vals = []
    quest_vals = []
    for tp in timepoints:
        crst_vals.append(crst_data.get(name, {}).get(tp, np.nan))
        quest_vals.append(quest_data.get(name, {}).get(tp, np.nan))
    
    patients.append({
        'name': name,
        'group': group,
        'crst': np.array(crst_vals, dtype=float),
        'quest': np.array(quest_vals, dtype=float),
    })

# ============================================================
# Group colors and styles
# ============================================================
group_config = {
    'SR': {'color': '#2166AC', 'label': 'Severe-Responsive (SR, n=6)', 'marker': 'o'},
    'MR': {'color': '#4DAF4A', 'label': 'Mild-Responsive (MR, n=11)', 'marker': 's'},
    'SF': {'color': '#D6604D', 'label': 'Severe-Refractory (SF, n=10)', 'marker': 'D'},
}

# ============================================================
# Figure: 2-panel (CRST Total + QUEST Total)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

for ax, score_key, ylabel, title in [
    (ax1, 'crst', 'CRST Total Score', 'CRST Total Score'),
    (ax2, 'quest', 'QUEST Total Score', 'QUEST Total Score'),
]:
    # Individual spaghetti lines
    for p in patients:
        g = p['group']
        vals = p[score_key]
        cc = group_config[g]['color']
        ax.plot(x, vals, color=cc, alpha=0.15, lw=0.8, zorder=1)
    
    # Group means with SEM
    for g in ['SR', 'MR', 'SF']:
        cfg = group_config[g]
        group_pts = [p for p in patients if p['group'] == g]
        vals_arr = np.array([p[score_key] for p in group_pts])
        
        means = np.nanmean(vals_arr, axis=0)
        n_valid = np.sum(~np.isnan(vals_arr), axis=0)
        sems = np.nanstd(vals_arr, axis=0, ddof=1) / np.sqrt(n_valid)
        sems = np.where(n_valid > 1, sems, 0)
        
        ax.errorbar(x, means, yerr=sems, color=cfg['color'], marker=cfg['marker'],
                    markersize=8, lw=2.5, capsize=4, capthick=1.5,
                    label=cfg['label'], zorder=5, markeredgecolor='white', markeredgewidth=0.8)
        
        # Fill between
        ax.fill_between(x, means - sems, means + sems, color=cfg['color'], alpha=0.08, zorder=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(tp_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', color='#1F4E79', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95, edgecolor='#CCCCCC')

fig.suptitle('Time Course of Clinical Scores by Response Group',
            fontsize=17, fontweight='bold', color='#1F4E79', y=0.98)
fig.text(0.5, 0.01, 'Thin lines = individual patients  |  Bold lines = group mean ± SEM  |  N = 27',
        ha='center', fontsize=10, color='#888888', style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{OUT}/Timecourse_grouped.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/Timecourse_grouped.pdf', bbox_inches='tight', facecolor='white')
print("Saved Timecourse_grouped.png/pdf")
