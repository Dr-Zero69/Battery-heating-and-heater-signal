"""Leave-trips-out cross-validation and metric computation.

Samples from the same trip are strongly autocorrelated, so a random row-level
split would let a model recognise a trip rather than learn the physics. Every
estimate here comes from folds in which whole trips are held out, with folds
stratified by season so that each contains both summer and winter trips.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedGroupKFold

from battery_heating import config as C

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def grouped_folds(
    groups: pd.Series,
    strata: pd.Series,
    n_folds: int = C.N_FOLDS,
    random_state: int = C.RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test index arrays with whole trips held out.

    ``strata`` is a per-row label (season) used to balance the folds; the
    splitter guarantees that every trip appears in exactly one test fold.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    codes = pd.Categorical(strata).codes
    return [(tr, te) for tr, te in splitter.split(np.zeros(len(groups)), codes, groups)]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
    }
    return {k: float(v) for k, v in out.items()}


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {
        "r2": r2_score(y_true, pred),
        "rmse": root_mean_squared_error(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
    }
    return {k: float(v) for k, v in out.items()}


def metrics_for(task_kind: str, y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task_kind == "classification":
        return classification_metrics(y_true, pred)
    return regression_metrics(y_true, pred)


def predict(model: BaseEstimator, X: pd.DataFrame, task_kind: str) -> np.ndarray:
    """Probability of the positive class for classifiers, point prediction otherwise."""
    if task_kind == "classification":
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return np.clip(pred, 0.0, None)  # heating power cannot be negative


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    """Everything produced by cross-validating one model on one task."""

    task: str
    feature_set: str
    model: str
    fold_metrics: pd.DataFrame
    oof_pred: np.ndarray
    fold_models: list[BaseEstimator] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Mean and standard deviation of every metric across folds."""
        out: dict[str, float] = {}
        for col in self.fold_metrics.columns:
            if col == "fold":
                continue
            out[f"{col}_mean"] = float(self.fold_metrics[col].mean())
            out[f"{col}_sd"] = float(self.fold_metrics[col].std(ddof=1))
        return out


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    task_name: str,
    feature_set: str,
    model_name: str,
) -> CVResult:
    task = C.TASKS[task_name]
    oof = np.full(len(y), np.nan)
    rows = []
    fold_models = []
    for k, (tr, te) in enumerate(folds):
        est = clone(model)
        est.fit(X.iloc[tr], y.iloc[tr])
        pred = predict(est, X.iloc[te], task.kind)
        oof[te] = pred
        m = metrics_for(task.kind, y.iloc[te].to_numpy(), pred)
        m["fold"] = k
        rows.append(m)
        fold_models.append(est)
        log.info(
            "  %s | %s | fold %d | %s=%.3f", model_name, feature_set, k, task.primary_metric, m[task.primary_metric]
        )
    return CVResult(
        task=task_name,
        feature_set=feature_set,
        model=model_name,
        fold_metrics=pd.DataFrame(rows),
        oof_pred=oof,
        fold_models=fold_models,
    )


def per_trip_metrics(
    y: pd.Series, oof: np.ndarray, groups: pd.Series, season: pd.Series, task_kind: str
) -> pd.DataFrame:
    """Out-of-fold performance for every held-out trip.

    For classification, trips in which the heater never changes state have an
    undefined ROC AUC; accuracy is still reported for them.
    """
    rows = []
    for trip, idx in groups.groupby(groups).groups.items():
        yt = y.loc[idx].to_numpy()
        pt = oof[np.asarray(groups.index.get_indexer(idx))]
        row = {C.TRIP: trip, C.SEASON: season.loc[idx].iloc[0], "n": len(idx)}
        if task_kind == "classification":
            pred = (pt >= 0.5).astype(int)
            row["accuracy"] = accuracy_score(yt, pred)
            row["heater_on_share"] = float(yt.mean())
            row["roc_auc"] = roc_auc_score(yt, pt) if len(np.unique(yt)) == 2 else np.nan
            row["brier"] = brier_score_loss(yt, pt)
        else:
            row["rmse"] = root_mean_squared_error(yt, pt)
            row["mae"] = mean_absolute_error(yt, pt)
            row["mean_actual_w"] = float(yt.mean())
            row["mean_pred_w"] = float(pt.mean())
            row["r2"] = r2_score(yt, pt) if np.std(yt) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def results_table(results: list[CVResult]) -> pd.DataFrame:
    """Long table of per-fold metrics for every (task, feature set, model)."""
    frames = []
    for r in results:
        df = r.fold_metrics.copy()
        df.insert(0, "model", r.model)
        df.insert(0, "feature_set", r.feature_set)
        df.insert(0, "task", r.task)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_table(results: list[CVResult]) -> pd.DataFrame:
    """Mean and SD across folds for every (task, feature set, model)."""
    rows = []
    for r in results:
        row = {"task": r.task, "feature_set": r.feature_set, "model": r.model}
        row.update(r.summary())
        rows.append(row)
    return pd.DataFrame(rows)
