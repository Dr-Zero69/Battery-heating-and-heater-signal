"""Model interpretation: permutation importance, SHAP values, partial dependence
and linear-model coefficients.

Permutation importance is computed on held-out trips only, so it measures how
much each feature contributes to predictions on unseen trips rather than how
well it helps memorise the training trips.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence, permutation_importance

from battery_heating import config as C
from battery_heating.evaluation import CVResult
from battery_heating.models import final_estimator

log = logging.getLogger(__name__)


def heldout_permutation_importance(
    result: CVResult,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    scoring: str,
    n_repeats: int = C.PERMUTATION_REPEATS,
    random_state: int = C.RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance averaged over the held-out fold of each CV split."""
    per_fold = []
    for k, ((_, te), est) in enumerate(zip(folds, result.fold_models)):
        pi = permutation_importance(
            est,
            X.iloc[te],
            y.iloc[te],
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state + k,
            n_jobs=C.N_JOBS,
        )
        per_fold.append(pi.importances_mean)
    arr = np.vstack(per_fold)
    out = pd.DataFrame(
        {
            "feature": X.columns,
            "label": [C.FEATURE_LABELS.get(f, f) for f in X.columns],
            "importance_mean": arr.mean(axis=0),
            "importance_sd": arr.std(axis=0, ddof=1),
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    out["scoring"] = scoring
    return out


def shap_values(
    model: BaseEstimator,
    X: pd.DataFrame,
    sample_size: int = C.SHAP_SAMPLE_SIZE,
    random_state: int = C.RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values for a tree model on a random sample of rows.

    Returns ``(values, X_sample)``. For classifiers the values are in log-odds.
    """
    import shap

    est = final_estimator(model)
    X_s = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    explainer = shap.TreeExplainer(est)
    values = explainer.shap_values(X_s)
    if isinstance(values, list):  # older SHAP returns one array per class
        values = values[1]
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, -1]
    return np.asarray(values), X_s


def shap_importance(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "feature": columns,
            "label": [C.FEATURE_LABELS.get(f, f) for f in columns],
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def partial_dependence_curves(
    model: BaseEstimator,
    X: pd.DataFrame,
    features: list[str],
    task_kind: str,
    grid_resolution: int = 40,
    sample_size: int = 20000,
    random_state: int = C.RANDOM_STATE,
) -> pd.DataFrame:
    """One-dimensional partial dependence for the listed features.

    For classifiers the response is the predicted probability of ``heater_on``.
    """
    X_s = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    # "brute" is used for both tasks: the faster "recursion" method ignores the
    # boosting baseline and would report regression curves offset from the data.
    kwargs = {"response_method": "predict_proba"} if task_kind == "classification" else {}
    frames = []
    for f in features:
        pd_res = partial_dependence(
            model, X_s, [f], grid_resolution=grid_resolution, kind="average", method="brute", **kwargs
        )
        frames.append(
            pd.DataFrame(
                {
                    "feature": f,
                    "label": C.FEATURE_LABELS.get(f, f),
                    "x": pd_res["grid_values"][0],
                    "y": pd_res["average"][0],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def linear_coefficients(result: CVResult, columns: list[str]) -> pd.DataFrame:
    """Standardised coefficients of a scaler+linear pipeline, with fold-to-fold SD.

    Because features are standardised inside the pipeline, each coefficient is
    the change in log-odds (classification) or in watts (regression) per one
    standard deviation of the feature.
    """
    coefs = []
    for est in result.fold_models:
        lin = final_estimator(est)
        coefs.append(np.ravel(lin.coef_))
    arr = np.vstack(coefs)
    out = pd.DataFrame(
        {
            "feature": columns,
            "label": [C.FEATURE_LABELS.get(f, f) for f in columns],
            "coef_mean": arr.mean(axis=0),
            "coef_sd": arr.std(axis=0, ddof=1),
        }
    )
    out["abs_coef"] = out["coef_mean"].abs()
    return out.sort_values("abs_coef", ascending=False, ignore_index=True).drop(columns="abs_coef")
