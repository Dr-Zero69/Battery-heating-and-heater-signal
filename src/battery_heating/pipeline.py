"""End-to-end pipeline: prepare -> describe -> train -> interpret -> report.

Every stage reads and writes files under ``data/processed`` and ``reports`` so
that the stages can be rerun independently. ``reports/results.json`` is the
single registry of numbers quoted in the report.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from battery_heating import __version__, data, evaluation, features, interpret, models, plotting
from battery_heating import config as C

log = logging.getLogger(__name__)

EXAMPLE_TRIPS = ["TripA24", "TripB04", "TripB16"]
CV_CACHE = C.DATA_PROCESSED / "cv_results.joblib"


def _write_json(payload: dict, path: Path = C.RESULTS_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not serialisable: {type(o)}")


def _load_results() -> dict:
    if C.RESULTS_JSON.exists():
        return json.loads(C.RESULTS_JSON.read_text(encoding="utf-8"))
    return {}


def _environment() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "random_state": C.RANDOM_STATE,
        "n_folds": C.N_FOLDS,
        "resample_seconds": C.RESAMPLE_SECONDS,
    }


# --------------------------------------------------------------------------- #
# Stage 1: prepare
# --------------------------------------------------------------------------- #
def prepare() -> pd.DataFrame:
    """Build and save the 1 Hz analysis table and the trip-level descriptive tables."""
    raw_all = data.load_all_trips()
    overview = data.load_overview()
    flag = features.heater_signal_by_trip(raw_all, overview)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    flag.to_csv(C.TABLES / "heater_signal_flag_by_trip.csv", index=False)
    overview.to_csv(C.TABLES / "trip_overview_clean.csv", index=False)

    table = data.build_analysis_table(raw=raw_all)
    table = features.add_derived_features(table)
    path = data.save_analysis_table(table)
    summary = features.trip_summary(table)
    summary.to_csv(C.TABLES / "trip_summary.csv", index=False)

    results = _load_results()
    results["environment"] = _environment()
    results["data"] = {
        "n_trips_total": int(raw_all[C.TRIP].nunique()),
        "n_rows_raw_10hz": len(raw_all),
        "n_trips_heater_signal_zero": int((flag["heater_signal"] == 0).sum()),
        "trips_heater_signal_zero": flag.loc[flag["heater_signal"] == 0, C.TRIP].tolist(),
        "heater_signal_constant_within_trip": bool((flag["heater_signal_nunique"] == 1).all()),
        "n_trips_analysed": int(table[C.TRIP].nunique()),
        "n_trips_analysed_summer": int(summary[summary[C.SEASON] == "summer"].shape[0]),
        "n_trips_analysed_winter": int(summary[summary[C.SEASON] == "winter"].shape[0]),
        "n_rows_1hz": len(table),
        "hours_1hz": float(len(table) / 3600),
        "heater_on_share": float(table[C.HEATER_ON].mean()),
        "heater_on_share_summer": float(table.loc[table[C.SEASON] == "summer", C.HEATER_ON].mean()),
        "heater_on_share_winter": float(table.loc[table[C.SEASON] == "winter", C.HEATER_ON].mean()),
        "heating_power_mean_w": float(table[C.HEATING_POWER_W].mean()),
        "heating_power_mean_w_when_on": float(table.loc[table[C.HEATER_ON] == 1, C.HEATING_POWER_W].mean()),
        "heating_power_max_w": float(table[C.HEATING_POWER_W].max()),
        "ambient_temp_range_c": [float(table[C.AMBIENT_T].min()), float(table[C.AMBIENT_T].max())],
        "n_trips_with_switching": int((summary["heater_switches"] > 0).sum()),
        "analysis_table": str(path.relative_to(C.PROJECT_ROOT)),
    }
    _write_json(results)
    log.info("Prepared analysis table at %s", path)
    return table


# --------------------------------------------------------------------------- #
# Stage 2: describe
# --------------------------------------------------------------------------- #
def describe() -> None:
    table = data.load_analysis_table()
    flag = pd.read_csv(C.TABLES / "heater_signal_flag_by_trip.csv")
    summary = pd.read_csv(C.TABLES / "trip_summary.csv")
    plotting.fig_heater_signal_flag(flag, C.FIGURES)
    plotting.fig_trip_overview(summary, C.FIGURES)
    plotting.fig_example_trips(table, EXAMPLE_TRIPS, C.FIGURES)
    plotting.fig_target_distributions(table, C.FIGURES)
    X, _, _ = features.make_xy(table, "heater_on", "exogenous")
    plotting.fig_feature_correlation(X, C.FIGURES)

    desc = X.describe().T
    desc.insert(0, "label", [C.FEATURE_LABELS.get(f, f) for f in desc.index])
    desc.to_csv(C.TABLES / "feature_descriptives.csv")
    log.info("Descriptive figures written to %s", C.FIGURES)


# --------------------------------------------------------------------------- #
# Stage 3: train
# --------------------------------------------------------------------------- #
def train() -> list[evaluation.CVResult]:
    table = data.load_analysis_table()
    results_all: list[evaluation.CVResult] = []
    registry = _load_results()
    registry.setdefault("cv", {})
    registry["cv"]["feature_sets"] = {k: v for k, v in C.FEATURE_SETS.items()}
    registry["cv"]["rows_used"] = {}

    for task_name, task in C.TASKS.items():
        for fs_name in C.FEATURE_SETS:
            X, y, groups = features.make_xy(table, task_name, fs_name)
            season = table.loc[X.index, C.SEASON]
            folds = evaluation.grouped_folds(groups, season)
            registry["cv"]["rows_used"][f"{task_name}/{fs_name}"] = {
                "n_rows": len(X),
                "n_trips": int(groups.nunique()),
                "rows_dropped_for_missing_features": int(len(table) - len(X)),
            }
            log.info("Task %s | features %s | %d rows, %d trips", task_name, fs_name, len(X), groups.nunique())
            for model_name, model in models.models_for(task.kind).items():
                t0 = time.perf_counter()
                res = evaluation.cross_validate_model(model, X, y, folds, task_name, fs_name, model_name)
                log.info("  %s done in %.1f s", model_name, time.perf_counter() - t0)
                results_all.append(res)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(results_all, CV_CACHE, compress=3)
    long = evaluation.results_table(results_all)
    summary = evaluation.summary_table(results_all)
    long.to_csv(C.TABLES / "cv_fold_metrics.csv", index=False)
    summary.to_csv(C.TABLES / "cv_summary.csv", index=False)

    registry["cv"]["summary"] = summary.to_dict(orient="records")
    _write_json(registry)
    plotting.fig_model_comparison(summary, C.FIGURES)
    return results_all


def _best_model(summary: pd.DataFrame, task_name: str, fs_name: str) -> str:
    metric = C.TASKS[task_name].primary_metric
    sub = summary[(summary["task"] == task_name) & (summary["feature_set"] == fs_name)]
    return sub.sort_values(f"{metric}_mean", ascending=False)["model"].iloc[0]


def _find(results: list[evaluation.CVResult], task: str, fs: str, model: str) -> evaluation.CVResult:
    for r in results:
        if (r.task, r.feature_set, r.model) == (task, fs, model):
            return r
    raise KeyError((task, fs, model))


# --------------------------------------------------------------------------- #
# Stage 4: interpret
# --------------------------------------------------------------------------- #
def interpret_models() -> None:
    table = data.load_analysis_table()
    results: list[evaluation.CVResult] = joblib.load(CV_CACHE)
    summary = pd.read_csv(C.TABLES / "cv_summary.csv")
    registry = _load_results()
    registry["interpretation"] = {}
    fs = "exogenous"

    # ---- out-of-fold performance figures ----------------------------------
    Xc, yc, gc = features.make_xy(table, "heater_on", fs)
    sc = table.loc[Xc.index, C.SEASON]
    folds_c = evaluation.grouped_folds(gc, sc)
    oof_c = {m: _find(results, "heater_on", fs, m).oof_pred for m in models.classifiers()}
    plotting.fig_roc_pr(yc.to_numpy(), oof_c, C.FIGURES)

    Xr, yr, gr = features.make_xy(table, "heating_power", fs)
    sr = table.loc[Xr.index, C.SEASON]
    folds_r = evaluation.grouped_folds(gr, sr)
    best_c = _best_model(summary, "heater_on", fs)
    best_r = _best_model(summary, "heating_power", fs)
    registry["interpretation"]["best_model_heater_on"] = best_c
    registry["interpretation"]["best_model_heating_power"] = best_r
    res_r = _find(results, "heating_power", fs, best_r)
    plotting.fig_pred_vs_actual(yr.to_numpy(), res_r.oof_pred, sr.to_numpy(), models.MODEL_LABELS[best_r], C.FIGURES)

    res_c = _find(results, "heater_on", fs, best_c)
    pt_c = evaluation.per_trip_metrics(yc, res_c.oof_pred, gc, sc, "classification")
    pt_r = evaluation.per_trip_metrics(yr, res_r.oof_pred, gr, sr, "regression")
    pt_c.to_csv(C.TABLES / "per_trip_heater_on.csv", index=False)
    pt_r.to_csv(C.TABLES / "per_trip_heating_power.csv", index=False)
    plotting.fig_per_trip(pt_c, pt_r, C.FIGURES)
    registry["interpretation"]["per_trip"] = {
        "heater_on_accuracy_median": float(pt_c["accuracy"].median()),
        "heater_on_accuracy_min": float(pt_c["accuracy"].min()),
        "heater_on_accuracy_min_trip": pt_c.sort_values("accuracy")[C.TRIP].iloc[0],
        "heater_on_trips_with_both_classes": int(pt_c["roc_auc"].notna().sum()),
        "heater_on_roc_auc_median_switching_trips": float(pt_c["roc_auc"].median()),
        "heating_power_rmse_median_w": float(pt_r["rmse"].median()),
        "heating_power_rmse_max_w": float(pt_r["rmse"].max()),
        "heating_power_rmse_max_trip": pt_r.sort_values("rmse")[C.TRIP].iloc[-1],
    }

    # ---- permutation importance (held-out folds) ---------------------------
    gb_c = _find(results, "heater_on", fs, "hist_gradient_boosting")
    gb_r = _find(results, "heating_power", fs, "hist_gradient_boosting")
    pi_c = interpret.heldout_permutation_importance(gb_c, Xc, yc, folds_c, scoring="roc_auc")
    pi_r = interpret.heldout_permutation_importance(gb_r, Xr, yr, folds_r, scoring="r2")
    pi_c.to_csv(C.TABLES / "permutation_importance_heater_on.csv", index=False)
    pi_r.to_csv(C.TABLES / "permutation_importance_heating_power.csv", index=False)
    plotting.fig_permutation_importance(pi_c, pi_r, C.FIGURES)
    registry["interpretation"]["permutation_importance_top5_heater_on"] = pi_c.head(5)[
        ["feature", "importance_mean", "importance_sd"]
    ].to_dict(orient="records")
    registry["interpretation"]["permutation_importance_top5_heating_power"] = pi_r.head(5)[
        ["feature", "importance_mean", "importance_sd"]
    ].to_dict(orient="records")

    # ---- full-data gradient boosting for SHAP and partial dependence --------
    full_c = models.classifiers()["hist_gradient_boosting"].fit(Xc, yc)
    full_r = models.regressors()["hist_gradient_boosting"].fit(Xr, yr)
    joblib.dump(
        {"heater_on": full_c, "heating_power": full_r},
        C.DATA_PROCESSED / "final_models_gradient_boosting.joblib",
        compress=3,
    )

    sv_c, Xs_c = interpret.shap_values(full_c, Xc)
    sv_r, Xs_r = interpret.shap_values(full_r, Xr)
    si_c = interpret.shap_importance(sv_c, list(Xc.columns))
    si_r = interpret.shap_importance(sv_r, list(Xr.columns))
    si_c.to_csv(C.TABLES / "shap_importance_heater_on.csv", index=False)
    si_r.to_csv(C.TABLES / "shap_importance_heating_power.csv", index=False)
    plotting.fig_shap_summary(sv_c, Xs_c, "log-odds of heater on", "fig11a_shap_heater_on", C.FIGURES)
    plotting.fig_shap_summary(sv_r, Xs_r, "heating power, W", "fig11b_shap_heating_power", C.FIGURES)

    top_c = pi_c["feature"].head(5).tolist()
    top_r = pi_r["feature"].head(5).tolist()
    pd_c = interpret.partial_dependence_curves(full_c, Xc, top_c, "classification")
    pd_r = interpret.partial_dependence_curves(full_r, Xr, top_r, "regression")
    pd_c.to_csv(C.TABLES / "partial_dependence_heater_on.csv", index=False)
    pd_r.to_csv(C.TABLES / "partial_dependence_heating_power.csv", index=False)
    plotting.fig_partial_dependence(pd_c, pd_r, C.FIGURES)

    # ---- linear coefficients and shallow trees ------------------------------
    coef_c = interpret.linear_coefficients(_find(results, "heater_on", fs, "logistic"), list(Xc.columns))
    coef_r = interpret.linear_coefficients(_find(results, "heating_power", fs, "ridge"), list(Xr.columns))
    coef_c.to_csv(C.TABLES / "coefficients_logistic_heater_on.csv", index=False)
    coef_r.to_csv(C.TABLES / "coefficients_ridge_heating_power.csv", index=False)
    plotting.fig_linear_coefficients(coef_c, coef_r, C.FIGURES)

    tree_c = models.classifiers()["decision_tree"].fit(Xc, yc)
    tree_r = models.regressors()["decision_tree"].fit(Xr, yr)
    plotting.fig_decision_tree(tree_c, list(Xc.columns), ["off", "on"], C.FIGURES, "fig14a_decision_tree_heater_on")
    plotting.fig_decision_tree(tree_r, list(Xr.columns), None, C.FIGURES, "fig14b_decision_tree_heating_power")

    _write_json(registry)
    log.info("Interpretation outputs written")


# --------------------------------------------------------------------------- #
# Stage 5: report tables
# --------------------------------------------------------------------------- #
def _fmt(mean: float, sd: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def report() -> Path:
    """Render the cross-validation summary as Markdown tables for the report."""
    summary = pd.read_csv(C.TABLES / "cv_summary.csv")
    lines = ["# Cross-validated performance (mean ± SD over 5 leave-trips-out folds)", ""]
    lines += [
        "## Heater on/off (classification)",
        "",
        "| Model | Features | ROC AUC | Average precision | Balanced accuracy | F1 | MCC | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary[summary["task"] == "heater_on"].iterrows():
        lines.append(
            f"| {models.MODEL_LABELS[r['model']]} | {r['feature_set']} | {_fmt(r['roc_auc_mean'], r['roc_auc_sd'])} | "
            f"{_fmt(r['average_precision_mean'], r['average_precision_sd'])} | {_fmt(r['balanced_accuracy_mean'], r['balanced_accuracy_sd'])} | "
            f"{_fmt(r['f1_mean'], r['f1_sd'])} | {_fmt(r['mcc_mean'], r['mcc_sd'])} | {_fmt(r['brier_mean'], r['brier_sd'])} |"
        )
    lines += [
        "",
        "## Heating power (regression)",
        "",
        "| Model | Features | R² | RMSE (W) | MAE (W) |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary[summary["task"] == "heating_power"].iterrows():
        lines.append(
            f"| {models.MODEL_LABELS[r['model']]} | {r['feature_set']} | {_fmt(r['r2_mean'], r['r2_sd'])} | "
            f"{_fmt(r['rmse_mean'], r['rmse_sd'], 0)} | {_fmt(r['mae_mean'], r['mae_sd'], 0)} |"
        )
    path = C.TABLES / "cv_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report tables written to %s", path)
    return path


STAGES = {
    "prepare": prepare,
    "describe": describe,
    "train": train,
    "interpret": interpret_models,
    "report": report,
}


def run_all() -> None:
    for name, fn in STAGES.items():
        log.info("=== Stage: %s ===", name)
        t0 = time.perf_counter()
        fn()
        log.info("=== %s finished in %.1f s ===", name, time.perf_counter() - t0)
