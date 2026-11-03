# Identifiability of human thermal-adaptation memory

**Reproducible analysis code and results** accompanying the manuscript *"Identifiability of human thermal-adaptation memory: a multi-dataset continuous-time analysis of fast and slow time constants and the running-mean window"* (manuscript BAE-D-26-08763, under revision).

This repository contains the analysis scripts, the generated results files underlying every figure and table, the derived daily outdoor-temperature series used to drive the injection experiments, and the scripts for all statistics added during the revision. Raw participant-level datasets are **not** redistributed here; they are public and can be re-downloaded from their original sources with the scripts in `src/download_*.py` (accessions are listed in the manuscript's Data Availability statement).

## Contents

```
src/                         analysis + figure scripts (the paper's pipeline)
results/
  identifiability/           synthetic information ceiling + injection experiment outputs
  standards/                 Chinese dataset window scans, ASHRAE DB II, counterfactual
  eda/                       DORN fast-constant event fits and heterogeneity
  revision/                  analyses added during revision (censoring, fold uncertainty,
                             sensitivity, 1800-epoch re-check)
derived_series/              the two derived daily outdoor series used as injection drivers
                             (In-Gauge Melbourne, 169 d; DORN Singapore, 278 d)
revision_stats.py            recomputes every statistic added during the revision
```

## Where each number comes from

| Manuscript result | Script | Output |
|---|---|---|
| Fig. 2 / Fig. S1 identifiability ceiling | `src/identifiability_map.py`, `src/profile_ident.py` | `results/identifiability/ident_map.csv`, `profile_nll.csv` |
| Fig. 3f, Table 2 (injection on real spectra) | `src/ingauge_inject.py` | `results/identifiability/realband_injection.csv` |
| Fig. 3e (fit to observed DORN votes) | `src/dorn_fit.py` | `results/eda/dorn_fit_results.json` |
| Fig. 4 fast constant (1,550 step events) | `src/dorn_event_tau.py` | `results/eda/dorn_event_tau.csv` |
| Fig. S4 trait heterogeneity (n = 20) | `src/dorn_tau_heterogeneity.py` | `results/eda/dorn_tau_heterogeneity.csv` |
| Fig. 5, S5 window scans (Chinese dataset) | `src/ctcd_window.py` | `results/standards/ctcd_window_scales.csv`, `ctcd_living_years.csv` |
| Fig. 5d–f (ASHRAE DB II) | `src/db2_window.py` | `results/standards/db2_window.csv`, `db2_adaptive_slope.csv` |
| Fig. 6, S6 counterfactual | `src/ingauge_counterfactual.py` | `results/standards/ingauge_counterfactual.csv` |
| All revision statistics (censoring, KM median, clustered bootstrap, step-magnitude split, exact p-values, paired fold contrasts, repeated city-blocked partitions, likelihood-ratio tests, τs sensitivity) | `revision_stats.py` + `results/revision/*.py` | `results/revision/revision_stats.json` and the CSVs beside it |

## Reproducing the results

Environment: Python 3.10, PyTorch (CUDA optional; the analyses were run on a consumer GPU), numpy, pandas, scipy, scikit-learn, matplotlib.

```bash
pip install -r requirements.txt
# 1) download the raw public datasets (accessions in the manuscript):
python src/download_ingauge.py   # PhysioNet in-gauge-and-en-gage
python src/download_data.py      # DORN, CTCD, ASHRAE DB II
# 2) rerun any analysis, e.g.:
python src/ingauge_inject.py     # Table 2 / Fig. 3f
python src/ctcd_window.py        # Fig. 5 window scans
python revision_stats.py         # every revision statistic (no GPU needed)
```

### Training schedule (epochs)

The five-model injection comparison in Table 2 was trained with **600 epochs** (all arms identically). The profile-likelihood and sensitivity analyses use **1800 epochs**, the DORN five-model fit **1500 epochs**, and the profile-grid cells **250–500 epochs**; cosine annealing throughout. Re-running the Table 2 hierarchy at the full 1800-epoch schedule reproduces every NLL within 0.003 and leaves all conclusions unchanged (`results/revision/table2_1800ep.csv`).

## Data provenance and licensing

- Code in this repository: MIT (see `LICENSE`).
- `derived_series/` (daily outdoor-temperature series derived from the public In-Gauge and DORN records): CC-BY 4.0.
- Raw datasets belong to their original authors and are used under their respective public access terms; obtain them from the sources cited in the manuscript. In-Gauge contains no individual thermal-sensation votes; its companion En-Gage study does (four weeks, 488 student + 22 teacher responses).
