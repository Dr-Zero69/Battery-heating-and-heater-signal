# Project source map

Authoritative files for every number quoted in `README.md` and `docs/report.md`.
Update this file whenever a new authoritative output is designated.

## Study scope and data version

- **Dataset**: "Battery and Heating Data in Real Driving Cycles", Technical University of Munich (Institute of Automotive Technology), IEEE DataPort, 2020. DOI: 10.21227/6jr9-5235. Mirror: Kaggle `atechnohazard/battery-and-heating-data-in-real-driving-cycles`.
- **Raw files used**: `data/raw/TripA01..A32.csv` (32 summer trips, June to September 2019), `data/raw/TripB01..B38.csv` (38 winter trips, December 2019 to January 2020), `data/raw/Overview.csv` (trip metadata). Raw files are not versioned in git; download from the DOI above and place them in `data/raw/`.
- **Analysis population**: the 50 trips in which heater current was logged (TripA01, A02, A23 to A32 and all 38 winter trips). The 20 summer trips without heater instrumentation are described but not modelled.
- **Analysis version**: 1 Hz analysis table built by `python -m battery_heating prepare` (package version 1.0.0). Random seed 42, 5 leave-trips-out folds stratified by season. Three feature sets: exogenous (15 signals), exogenous without battery voltage and current, and exogenous plus four heating-loop temperatures; see `config.FEATURE_SETS`.

## Targets

- `heater_on`: heater current > 0 A for more than half of each 1 s bin (binary).
- `heating_power_w`: `Heating Power LIN [W]` averaged over the 1 s bin, readings above 8000 W masked as sensor faults (continuous).
- The original `Heater Signal` column is constant within every trip and is described at trip level only (`reports/tables/heater_signal_flag_by_trip.csv`); it is not a usable per-sample target.

## Authoritative outputs

| Quantity | File |
| --- | --- |
| All quoted numbers (registry) | `reports/results.json` |
| Cross-validation, per fold | `reports/tables/cv_fold_metrics.csv` |
| Cross-validation, mean ± SD | `reports/tables/cv_summary.csv`, `reports/tables/cv_summary.md` |
| Per-trip held-out performance | `reports/tables/per_trip_heater_on.csv`, `reports/tables/per_trip_heating_power.csv` |
| Permutation importance | `reports/tables/permutation_importance_*.csv` |
| SHAP importance | `reports/tables/shap_importance_*.csv` |
| Partial dependence | `reports/tables/partial_dependence_*.csv` |
| Linear coefficients | `reports/tables/coefficients_*.csv` |
| Trip-level descriptives | `reports/tables/trip_summary.csv`, `reports/tables/trip_overview_clean.csv` |
| Feature descriptives | `reports/tables/feature_descriptives.csv` |
| Figures | `reports/figures/fig01..fig14*.png` (+ `.pdf`) |
| Analysis table | `data/processed/analysis_table_1hz.parquet` (not versioned; rebuilt by `prepare`) |
| Fitted models | `data/processed/final_models_gradient_boosting.joblib`, `data/processed/cv_results.joblib` (not versioned) |

## Supporting scripts

- `src/battery_heating/config.py`: paths, column names, feature sets, model settings.
- `src/battery_heating/data.py`: loading, cleaning, resampling.
- `src/battery_heating/features.py`: derived features, feature matrices, trip summaries.
- `src/battery_heating/models.py`: model factories and fixed hyper-parameters.
- `src/battery_heating/evaluation.py`: grouped cross-validation and metrics.
- `src/battery_heating/interpret.py`: permutation importance, SHAP, partial dependence, coefficients.
- `src/battery_heating/plotting.py`: all figures.
- `src/battery_heating/pipeline.py`: stage orchestration and the results registry.
- `tests/`: unit tests (`python -m pytest`).

## Archived / superseded

- `notebooks/archive/*.ipynb`: the original exploratory notebooks (October 2024 to July 2025). They treated the constant per-trip `Heater Signal` flag as a per-sample target with a random row-level split and SMOTE, which leaks trip identity; their results are superseded and must not be quoted.
- `data/raw/readin.m`: MATLAB loader shipped with the dataset, kept for reference.

## Target venue and reporting

- Not yet designated. Default numerical precision follows the global manuscript rules: metrics to 3 decimals, percentages to 1 decimal, counts as integers, RMSE and MAE in watts to 0 decimals.

## Reference management

- Zotero access was not available in this session. References are listed in `docs/report.md` with DOIs; no Zotero collection has been created.
