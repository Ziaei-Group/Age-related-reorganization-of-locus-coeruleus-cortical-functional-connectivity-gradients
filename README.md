# Age-related reorganization of locus coeruleus–cortical functional connectivity gradients

Code, source data and stimuli for the manuscript (Ziaei Group, Kavli Institute for Systems Neuroscience, NTNU).

**Preprint:** [bioRxiv, doi:10.64898/2026.02.05.704005](https://www.biorxiv.org/content/10.64898/2026.02.05.704005v2)

## Contents

| Folder | What it holds |
|---|---|
| `Analysis Scripts/` | MATLAB live scripts: `Gradient.mlx` (LC–cortical connectivity gradients), `individual_to_group.mlx` (individual to group-level gradients), `dispersion_main_F.mlx` (gradient dispersion analysis), `splithalf and kfold.mlx` (split-half and k-fold reliability). Python: `age_dispersion_behaviour_functions.py` (group comparisons of dispersion with covariate adjustment, brain–behaviour correlations with permutation testing, FDR correction, age-moderation analysis), `surface_plot.py` (projects gradient maps onto the fsaverage surface). |
| `Source Data/Disperion_source_data/` | `Source_data_LC_Cortex_Gradient.xlsx` – values underlying the main figures. |
| `Source Data/Sensitivity_Analyses_source_data/` | `LOO_LC_Cortex_Gradient.xlsx` (leave-one-out), plus `inter_individual_var/`, `k_fold/` and `Split_half/` with `.mat` results per condition (negative/neutral) and LC hemisphere (left/right). |
| `Movie stimuli/` | The two naturalistic movie clips used during fMRI (stored with Git LFS; install [Git LFS](https://git-lfs.com) before cloning). |
