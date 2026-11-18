# Predicting cabin-heater activity and heating-power demand in a battery electric vehicle from real driving data

Technical report, version 1.0 (18 November 2026). All numbers are taken from `reports/results.json` and the tables in `reports/tables/`; see `PROJECT_SOURCES.md` for the source map.

## Abstract

Cabin heating is one of the largest auxiliary loads of a battery electric vehicle in winter and directly reduces range. We asked whether the on/off state and the instantaneous electrical power of a high-voltage cabin heater can be predicted on unseen trips from signals that do not measure the heater itself, and which signals matter. We used 70 trips of a BMW i3 recorded at 10 Hz by the Technical University of Munich, of which 50 (12 summer, 38 winter; 21.7 h) logged heater current. The original per-trip `Heater Signal` flag was found to be constant within every trip and was therefore replaced by two sample-level targets derived from the heater current and the LIN-bus heating power at 1 Hz. Four model families (regularised linear model, depth-3 decision tree, random forest, histogram gradient boosting) were evaluated with five-fold leave-trips-out cross-validation stratified by season. Random forest classified heater on/off on held-out trips with a ROC AUC of 0.973 ± 0.043 and a balanced accuracy of 0.953 ± 0.058; gradient boosting predicted heating power with R² 0.720 ± 0.213 and RMSE 622 ± 257 W. Ambient temperature was the dominant driver of activation, with a sharp transition between 20 °C and 22 °C in the partial dependence, while heating power was governed mainly by cabin temperature, battery current and motor torque. Removing battery voltage and current, which respond to the heater's own draw, reduced balanced accuracy to 0.891 and R² to 0.671. Performance varied substantially between trips, with the largest errors in the summer trips that used the heater briefly and in the coldest winter trip. The results show that heater activation is largely a function of the thermal environment and can be anticipated from exogenous signals, whereas instantaneous power depends on the warm-up dynamics of the cabin that are only partly captured by the available sensors.

## 1. Introduction

Heating the cabin of a battery electric vehicle in cold weather can consume power comparable to propulsion in urban driving, and the resulting range loss is a well-known barrier to adoption. A model that anticipates heater activity from the driving situation and the thermal environment is useful in two ways: as a component of energy-consumption and range prediction, and as a description of the control behaviour of the heater that can be compared with the vehicle's thermal-management strategy.

The dataset used here was published by the Institute of Automotive Technology of the Technical University of Munich (Steinstraeter, Heinrich and Lienkamp, 2020; DOI 10.21227/6jr9-5235). It contains 70 real driving trips of a BMW i3 (60 Ah) around Munich with vehicle, battery and heating signals. The accompanying study used the data to validate a thermal model of the vehicle. This report takes a complementary, data-driven view and asks:

1. Can the heater's on/off state at a given second be predicted on trips the model has never seen?
2. Can its instantaneous electrical power be predicted?
3. Which measured signals carry the information, and do the answers survive the removal of signals that are partly caused by the heater?

An earlier analysis in this repository (now in `notebooks/archive/`) modelled the dataset's `Heater Signal` column with logistic regression after a random row-level train/test split and SMOTE oversampling. Section 2.2 explains why that target and that validation scheme could not answer the questions above; the present work replaces both.

## 2. Data and methods

### 2.1 Dataset

The 32 summer trips (June to September 2019, prefix A) and 38 winter trips (December 2019 to January 2020, prefix B) are stored as semicolon-separated CSV files at 10 Hz together with a per-trip overview. Winter files contain 48 signals; summer files contain between 22 and 28 because the measurement system was incomplete. Column headers required normalisation: trailing spaces, a duplicated `Temperature Vent right` column in every winter file, a closing-bracket typo in `max. SoC [%)` in every file, and a mis-typed velocity header in one file.

### 2.2 The `Heater Signal` flag

Across all 1 094 793 raw rows the `Heater Signal` column takes a single value within every trip (Figure 1): 1 in 65 trips and 0 in five summer trips (A07, A18, A19, A20, A21). Its unit of variation is therefore the trip, and with 70 trips, 5 of them negative, it supports at most a descriptive statement that the flag is off only in warm weather. Any sample-level classifier trained on this flag with a random row split learns to recognise the trip from the slowly varying signals (state of charge, elevation, ambient temperature) rather than anything about heating, and its test accuracy is not an estimate of generalisation to new trips. The flag was therefore not used as a target.

### 2.3 Analysis population and targets

Heater current, heater voltage and LIN-bus heating power were logged in 50 trips (A01, A02, A23 to A32 and all 38 winter trips). These form the analysis population. Readings of heating power above 8 000 W, which exceed the 7 kW rating of the heater, occurred in 1 580 rows of TripB02 and were masked together with the simultaneous heater current and voltage.

Each trip was averaged into 1 s bins, labelled by seconds since the first sample, yielding 78 282 rows (21.7 h). Two targets were defined:

- `heater_on`: 1 if the heater drew current for more than half of the bin, 0 otherwise. The heater was on in 77.7 % of all seconds, 11.5 % of summer seconds and 94.4 % of winter seconds. In 30 of the 50 trips the state changed at least once; of the remaining 20, the heater was on throughout 14 winter trips and never on in 6 summer trips.
- `heating_power_w`: mean LIN-bus heating power in the bin (0 W when off). Mean 1 320 W overall and 1 697 W while on; maximum 7 980 W. The distribution while on has a main mode near 1 kW and a secondary mode near 5.6 kW (Figure 4a), corresponding to steady-state operation and the full-power warm-up phase.

Heater current and the LIN power agree on the on/off state in 99.99 % of raw rows, and the product of heater voltage and current correlates with the LIN power at r = 0.9995, so the two targets are internally consistent.

### 2.4 Features

Three feature sets were evaluated (Table 1). The **exogenous** set contains fifteen signals that are not measurements of the heater or of the heating loop it controls: time since trip start, velocity, elevation, throttle, motor torque, longitudinal acceleration, regenerative-braking flag, battery voltage, battery current, battery temperature, state of charge, air-conditioning power, ambient temperature, cabin temperature and the target cabin temperature from the overview. Derived quantities such as battery power (= V × I) and the differences between the target and measured temperatures were computed for descriptive use but excluded from the models because they are exact functions of listed features and would make the linear coefficients unidentifiable (Figure 5 shows the rank correlations).

Battery voltage and current are exogenous from the perspective of the heater controller but respond to the heater's own draw (about 18 A at 7 kW). The **without battery V, I** set removes them to quantify how much of the predictive signal is genuinely environmental. The **+ thermal loop** set adds heat-exchanger temperature, heater-core coolant temperature, coolant inlet temperature and requested coolant temperature, which are downstream of the heater and are reported separately for that reason. Heater signal, heater voltage and current, and the three heating-power channels are never used as predictors; a guard in `features.check_no_leakage` enforces this.

### 2.5 Models

Four model families of increasing flexibility were compared, each with hyper-parameters fixed a priori at conservative values rather than tuned, because the aim was a fair comparison of interpretable model classes and the sample per trip is large:

- logistic regression (classification) or ridge regression (regression) on standardised features, C = 1 or alpha = 1, class weights balanced;
- decision tree of depth 3 with at least 200 samples per leaf;
- random forest of 200 trees, at least 20 samples per leaf, square-root feature subsampling, balanced subsample class weights;
- histogram gradient boosting with learning rate 0.05, 400 iterations, 31 leaves, at least 100 samples per leaf, L2 regularisation 1.

### 2.6 Validation

Samples within a trip are strongly autocorrelated. All performance estimates therefore come from five-fold cross-validation in which whole trips are held out (`StratifiedGroupKFold`, folds balanced by season, seed 42). Classification is reported as ROC AUC, average precision, balanced accuracy, F1, Matthews correlation coefficient and Brier score; regression as R², RMSE and MAE. Out-of-fold predictions were also summarised per held-out trip. Permutation importance was computed on the held-out fold of each split and averaged. SHAP values (TreeExplainer) and one-dimensional partial dependence were computed for gradient-boosting models refitted on all 50 trips, the former on a random sample of 3 000 rows. Standardised coefficients of the linear models are reported with their fold-to-fold standard deviation.

Software: Python 3.11.4, pandas 3.0.6, NumPy 2.0.2, scikit-learn 1.9.1, SHAP 0.51.0, matplotlib 3.9.0. The pipeline is reproducible with `python -m battery_heating all`.

**Table 1.** Feature sets.

| Set | Signals | Purpose |
|---|---|---|
| exogenous | 15 driving, battery, environment and set-point signals | main analysis |
| without battery V, I | exogenous minus battery voltage and current | sensitivity to the heater's own electrical draw |
| + thermal loop | exogenous plus 4 heating-loop temperatures | upper bound with downstream signals |

## 3. Results

### 3.1 Descriptive

Heater use is almost entirely a matter of season (Figure 2). Every winter trip ran the heater for at least 44 % of its duration and 36 of 38 for more than 90 %; trip-mean heating power ranged from 0.7 kW to 3.4 kW and declined with ambient temperature. Among the twelve summer trips with heater instrumentation, six never used the heater, three used it briefly, and three (A23, A24, A25) used it for more than a third of the trip, all at trip-mean ambient temperatures between 18 °C and 21 °C. The example trips in Figure 3 show the characteristic pattern: a full-power burst of 5 to 7 kW in the first minutes while the cabin is cold, followed by steady operation at 1 to 2 kW or switch-off once the cabin reaches its set-point.

### 3.2 Heater on/off

**Table 2.** Leave-trips-out cross-validation for heater on/off (mean ± SD over five folds). Full metrics in `reports/tables/cv_summary.md`.

| Model | Features | ROC AUC | Balanced accuracy | Brier |
|---|---|---|---|---|
| Logistic regression | exogenous | 0.933 ± 0.090 | 0.923 ± 0.071 | 0.084 ± 0.084 |
| Decision tree (depth 3) | exogenous | 0.912 ± 0.087 | 0.904 ± 0.088 | 0.064 ± 0.060 |
| Random forest | exogenous | 0.973 ± 0.043 | 0.953 ± 0.058 | 0.042 ± 0.046 |
| Gradient boosting | exogenous | 0.968 ± 0.052 | 0.951 ± 0.061 | 0.049 ± 0.080 |
| Random forest | without battery V, I | 0.967 ± 0.045 | 0.891 ± 0.098 | 0.052 ± 0.046 |
| Random forest | + thermal loop | 0.978 ± 0.041 | 0.931 ± 0.094 | 0.036 ± 0.039 |

The two ensembles outperform the linear model and the shallow tree on every metric, and random forest is the best exogenous model (Table 2, Figures 6 and 7). Removing battery voltage and current costs little ROC AUC but reduces balanced accuracy by 0.062, indicating that the electrical signals mainly sharpen the decision threshold rather than the ranking. Adding the thermal-loop temperatures improves the Brier score slightly but not balanced accuracy, and the differences are well within one fold-to-fold standard deviation.

At trip level (Figure 9a) the median held-out accuracy of the random forest is 0.999 in both seasons. The errors concentrate in a few trips: A25 (accuracy 0.066; the heater ran for 93 % of a warm trip and the model predicted it off), A24 (0.484), A23 (0.621) and B09 (0.687). Among the 30 trips in which the heater changed state, the median per-trip ROC AUC was 0.948.

### 3.3 Heating power

**Table 3.** Leave-trips-out cross-validation for heating power.

| Model | Features | R² | RMSE (W) | MAE (W) |
|---|---|---|---|---|
| Ridge regression | exogenous | 0.597 ± 0.192 | 785 ± 188 | 487 ± 133 |
| Decision tree (depth 3) | exogenous | 0.551 ± 0.172 | 840 ± 218 | 510 ± 138 |
| Random forest | exogenous | 0.716 ± 0.165 | 650 ± 204 | 413 ± 97 |
| Gradient boosting | exogenous | 0.720 ± 0.213 | 622 ± 257 | 383 ± 137 |
| Gradient boosting | without battery V, I | 0.639 ± 0.221 | 732 ± 269 | 432 ± 142 |
| Gradient boosting | + thermal loop | 0.691 ± 0.248 | 655 ± 279 | 393 ± 161 |

Gradient boosting and random forest explain about 72 % of the variance of heating power on held-out trips, against 55 % to 60 % for the two simpler models. The hexbin of out-of-fold predictions (Figure 8a) shows good agreement in the steady-state band around 1 to 2 kW and under-prediction of the 5 to 7 kW warm-up bursts, which appear as a horizontal band of observed high power predicted at 1 kW. Residuals are centred near zero in winter and skewed positive in summer (Figure 8b), where the model predicts heating that did not occur. Per-trip RMSE (Figure 9b) has a median of 385 W in winter and 617 W in summer; the worst trips are B38 (1 891 W; at -0.9 °C one of the two coldest trips, with a long warm-up that the model timed wrongly), A24 (1 792 W) and A23 (1 637 W). Removing battery voltage and current lowers R² by 0.08; the thermal-loop temperatures do not help.

### 3.4 What drives the predictions

**Activation.** Ambient temperature dominates every interpretation method. Its permutation importance on held-out folds (drop in ROC AUC when shuffled) is 0.156 ± 0.052, three times that of the next feature, battery current (0.048 ± 0.041), followed by battery temperature (0.017 ± 0.022) (Figure 10a). The SHAP ranking is the same (mean |SHAP| 3.19 log-odds for ambient temperature, 1.53 for battery temperature, 1.26 for battery current; Figure 11a). The partial dependence of P(heater on) on ambient temperature (Figure 12, top left) is 0.94 at 1.5 °C, 0.88 at 9 °C, flat near 0.77 between 12 °C and 19 °C, and falls to 0.15 at 22 °C. The depth-3 tree (Figure 14a) reproduces this with a root split at 10.0 °C: below it the heater is on in 90 % of weighted samples, above it off in 90 %, with battery temperature (13.5 °C) and elevation refining the warm branch. The logistic coefficient of ambient temperature is -3.11 ± 0.68 log-odds per standard deviation, by far the largest (Figure 13a). Motor torque has a negative coefficient (-2.15 ± 0.61) and longitudinal acceleration a positive one (1.74 ± 0.54); the two are correlated at 0.90, so these should be read jointly as a weak association between propulsion demand and heater state rather than separately.

**Power.** For heating power the picture is more distributed. Permutation importance (drop in R²) is led by battery current (0.680 ± 0.339) and cabin temperature (0.604 ± 0.136), then motor torque (0.292 ± 0.252), ambient temperature (0.138 ± 0.099) and time since trip start (0.117 ± 0.036) (Figure 10b). The SHAP ranking places cabin temperature first (mean |SHAP| 511 W), then ambient temperature (223 W), battery temperature (219 W) and battery current (207 W) (Figure 11b). Partial dependence (Figure 12, bottom row) shows heating power falling from about 3.5 kW at a cabin temperature of 11 °C to about 1 kW above 24 °C, and a step from about 2.2 kW to 0.8 kW as battery current crosses from discharge to near zero; the latter is the signature of the heater's own 18 A draw and is the main reason battery current ranks highly. The ridge coefficients (Figure 13b) are consistent and physically sensible: -839 ± 46 W per standard deviation of cabin temperature, -659 ± 76 W for ambient temperature and +311 ± 48 W for the target cabin temperature. The depth-3 regression tree (Figure 14b) splits first on cabin temperature at 17.2 °C (mean 4.0 kW below, 1.0 kW above) and then on time since trip start and battery temperature in the cold branch and on ambient temperature in the warm branch.

## 4. Discussion

The heater's on/off state is predictable on unseen trips with high accuracy because it is governed mainly by the thermal environment. The partial dependence and the single tree agree on a threshold near 10 °C ambient temperature below which the heater is essentially always on, and a second transition around 20 to 22 °C above which it is off. The flat region between them, where P(heater on) sits near 0.77, corresponds to the mild summer trips in which the heater is used intermittently for a few minutes, and these trips (A23 to A25) account for most of the classification errors. A sample-level model without memory cannot know whether the driver has requested heat in a mild trip; the per-trip accuracy of 0.066 in A25 shows the cost.

Heating power is harder. The dominant source of error is the full-power warm-up burst, whose timing and duration depend on how cold the cabin is at the start and on how quickly it warms, neither of which is fully captured by instantaneous features. Cabin temperature carries most of the available information, consistent with a controller that heats hard while the cabin is far below its set-point and throttles back as it approaches it. The positive summer residuals and the B38 error show that the models extrapolate poorly at the edges of the ambient-temperature range, where few trips exist.

The sensitivity analysis clarifies the role of the battery electrical signals. They are not leakage in the strict sense, since a range-prediction system would have them available, but part of their value comes from the heater's draw. Without them, ranking performance for activation is almost unchanged (ROC AUC 0.967 against 0.973) while calibration at the default threshold degrades (balanced accuracy 0.891 against 0.953), and heating-power R² falls from 0.716 to 0.671. The results reported for the exogenous set should be read with this in mind, and the without-battery set gives the conservative estimate.

Compared with the original notebooks in this repository, the present analysis changes the target, the validation scheme and the feature set, and the headline numbers are not comparable. The earlier logistic-regression accuracy was a measure of trip recognition under a random split; the accuracies reported here are measured on trips absent from training. The earlier analysis also removed battery temperature, state of charge and ambient temperature as "multicollinear", which would have discarded the most informative predictor.

### Limitations

- **One vehicle, one region, one winter.** The 50 instrumented trips come from a single BMW i3 around Munich. Thresholds such as 10 °C and 22 °C describe this vehicle's controller and climate and should not be transferred to other vehicles without re-estimation.
- **Few summer trips with heater data.** Only 12 summer trips logged heater current, and the heater was used in 6 of them, so the mild-weather regime where predictions are hardest is also the least represented. Fold-to-fold standard deviations are correspondingly large.
- **Fixed hyper-parameters.** Models were not tuned, so the reported performance is a lower bound for each family. Tuning within a nested cross-validation would be needed before claiming that one ensemble is better than the other; the differences between random forest and gradient boosting here are well within uncertainty.
- **No temporal memory.** Features are instantaneous. Lagged features or sequence models would likely improve the prediction of warm-up bursts, at the cost of interpretability.
- **Partial dependence and permutation importance assume feature independence.** Several features are correlated (throttle, torque and acceleration; battery current and power; battery and ambient temperature), so importance is shared among correlated groups and partial-dependence curves outside the observed joint distribution are extrapolations.
- **Sensor faults.** The 38.5 kW readings in TripB02 were masked by a fixed 8 kW limit; other, smaller faults may remain.

### Future work

Lagged and cumulative features (time since the heater last switched, integrated heating energy), a hidden-state or sequence model for the warm-up phase, hyper-parameter tuning under nested leave-trips-out validation, and validation on a second vehicle or a second winter would each address one of the limitations above. Linking predicted heating energy to range would make the model directly useful for energy management.

## 5. Conclusion

In 50 instrumented trips of a BMW i3, the cabin heater's on/off state at a given second could be predicted on unseen trips with a ROC AUC of 0.973 and a balanced accuracy of 0.953 from driving, battery and environmental signals, with ambient temperature as the dominant driver and transitions near 10 °C and 22 °C. Instantaneous heating power was predicted with R² 0.72 and RMSE 622 W, limited by the timing of the warm-up burst, and was governed mainly by cabin temperature, battery current and motor torque. The dataset's `Heater Signal` flag is constant within trips and should not be used as a sample-level target.

## References

- Steinstraeter, M., Heinrich, T., Lienkamp, M. (2020). *Battery and Heating Data in Real Driving Cycles*. IEEE DataPort. DOI: 10.21227/6jr9-5235.
- Steinstraeter, M., Heinrich, T., Lienkamp, M. (2021). Effect of Low Temperature on Electric Vehicle Range. *World Electric Vehicle Journal*, 12(3), 115. DOI: 10.3390/wevj12030115.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. DOI: 10.1023/A:1010933404324.

The second reference is the dataset authors' companion paper as listed on IEEE DataPort; its bibliographic details were not verified against the publisher record in this session and should be checked before submission.

## Figure captions

1. **fig01_heater_signal_flag_by_trip.** Trip-level value of the raw `Heater Signal` column against trip-mean ambient temperature for all 70 trips. The flag is constant within every trip and is 0 in five summer trips only.
2. **fig02_trip_overview.** (a) Share of each trip with the heater on and (b) trip-mean heating power against trip-mean ambient temperature for the 50 heater-instrumented trips.
3. **fig03_example_trips.** Heating power (left) and cabin, ambient and target cabin temperatures (right, heater-on periods shaded) for TripA24 (summer), TripB04 (winter, intermittent) and TripB16 (winter, continuous).
4. **fig04_target_distributions.** (a) Density of heating power while the heater is on, by season. (b) Per-trip share of time with the heater on, by season; bar is the median.
5. **fig05_feature_correlation.** Spearman rank correlation between the exogenous features; values with |r| >= 0.5 are printed.
6. **fig06_model_comparison.** Mean ± SD over five leave-trips-out folds of ROC AUC and balanced accuracy (heater on/off) and R² and RMSE (heating power) for every model and feature set.
7. **fig07_roc_pr_curves.** Out-of-fold ROC and precision-recall curves of the four classifiers on the exogenous feature set.
8. **fig08_predicted_vs_actual.** (a) Out-of-fold predicted against observed heating power for gradient boosting, bin colour on a log count scale. (b) Residuals by season.
9. **fig09_per_trip_performance.** (a) Held-out accuracy per trip for the random-forest classifier and (b) held-out RMSE per trip for the gradient-boosting regressor, by season; bars are medians and the two worst trips are labelled.
10. **fig10_permutation_importance.** Permutation importance on held-out folds (mean ± SD) for gradient boosting on the exogenous set: (a) drop in ROC AUC, (b) drop in R².
11. **fig11a / fig11b_shap.** SHAP beeswarm plots for the gradient-boosting classifier (log-odds) and regressor (watts) refitted on all trips; 3 000 sampled rows.
12. **fig12_partial_dependence.** One-dimensional partial dependence of P(heater on) (top) and heating power (bottom) on the five most important features by permutation importance.
13. **fig13_linear_coefficients.** Standardised coefficients (mean ± SD across folds) of logistic regression (a) and ridge regression (b).
14. **fig14a / fig14b_decision_tree.** Depth-3 classification and regression trees fitted to all 50 trips. Sample proportions in the classification tree are class-weighted.
