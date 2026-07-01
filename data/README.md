# Data

All patient-level data in this folder are **de-identified**. Original patient
names and medical record numbers have been replaced with sequential anonymous
identifiers (`P01`–`P27`); appointment dates have been removed. The mapping
between anonymous identifiers and patient records is held only by the study
team and is not distributed.

The files here are the processed inputs required to regenerate the manuscript
figures. Raw source records are available from the corresponding author on
reasonable request, subject to institutional data-sharing policies and
applicable ethics approvals.

## Files

| File | Description |
|------|-------------|
| `patients_clustering.csv` | Per-patient clinical summary and response-subgroup assignment. Columns: `name`, `patient_id` (both anonymized), `Base_CRST`, `Imp_CRST` (%), `Base_QUEST`, `Imp_QUEST` (%), `Cluster` (1/2/3), `Cluster_Score`, `response` (`high`/`moderate`/`low`). |
| `snp_dosage_matrix_final.csv` | Non-reference allele dosage matrix. Rows = 183 SNPs (`gene`, `SNP ID`); columns `P01`–`P27` = per-patient dosage (0/1/2). |
| `CRST.csv` | Longitudinal CRST sub-scores. `Unnamed: 0` = patient ID (baseline row only), `Unnamed: 1` = timepoint (`baseline`/`1month`/`2month`/`3month`), plus item scores and `total`. |
| `QUEST.csv` | Longitudinal QUEST (QoL) scores in the same layout, with `Questionnaire_sum`. |
| `stats/dosage_crst_full_stats.csv` | Per-SNP dosage-based association statistics (LBL, JT, Spearman-permutation, ordinal logistic) vs. CRST improvement. Gene-level only; no patient data. |
| `stats/group_nominal_stats_final.csv` | Per-SNP subgroup-based association statistics (Fisher exact MC, Kruskal–Wallis, G-test). Gene-level only; no patient data. |

## Response-subgroup labels

The three data-driven subgroups (Ward-linkage hierarchical clustering, k = 3)
map to the `Cluster` column as follows:

| Cluster | Label | Abbrev. | n |
|---------|-------|---------|---|
| 2 | Severe-Responsive | SR | 6 |
| 1 | Mild-Responsive | MR | 11 |
| 3 | Severe-Refractory | SF | 10 |
