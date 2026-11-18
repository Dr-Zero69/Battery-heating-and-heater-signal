"""Tests for the leave-trips-out cross-validation machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from battery_heating import evaluation


def _toy(n_trips: int = 10, n_per: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.repeat([f"T{i:02d}" for i in range(n_trips)], n_per))
    season = pd.Series(np.where(groups.str[1:].astype(int) < n_trips // 2, "summer", "winter"))
    X = pd.DataFrame({"a": rng.normal(size=len(groups)), "b": rng.normal(size=len(groups))})
    y = pd.Series(((X["a"] + 0.5 * X["b"] + rng.normal(scale=0.3, size=len(X))) > 0).astype(int))
    return X, y, groups, season


def test_grouped_folds_hold_out_whole_trips():
    _X, _y, groups, season = _toy()
    folds = evaluation.grouped_folds(groups, season, n_folds=5)
    assert len(folds) == 5
    seen = []
    for tr, te in folds:
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])
        seen.extend(groups.iloc[te].unique())
    assert sorted(seen) == sorted(groups.unique())


def test_cross_validate_model_returns_oof_for_every_row():
    X, y, groups, season = _toy()
    folds = evaluation.grouped_folds(groups, season, n_folds=5)
    res = evaluation.cross_validate_model(LogisticRegression(), X, y, folds, "heater_on", "exogenous", "logistic")
    assert not np.isnan(res.oof_pred).any()
    assert res.fold_metrics.shape[0] == 5
    assert res.summary()["roc_auc_mean"] > 0.8
    assert len(res.fold_models) == 5


def test_per_trip_metrics_handles_single_class_trips():
    _X, y, groups, season = _toy()
    y = y.copy()
    y[groups == "T00"] = 1  # a trip with only one class
    oof = np.clip(y + 0.1, 0, 1).to_numpy()
    out = evaluation.per_trip_metrics(y, oof, groups, season, "classification")
    assert out.shape[0] == groups.nunique()
    assert np.isnan(out.loc[out["trip"] == "T00", "roc_auc"].iloc[0])
    assert out["accuracy"].between(0, 1).all()


def test_regression_metrics_keys():
    m = evaluation.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))
    assert set(m) == {"r2", "rmse", "mae"}
    assert m["r2"] > 0.9
