"""Model factories.

Four model families are compared for each task, ordered from most to least
transparent: a regularised linear model, a shallow decision tree, a random
forest and histogram gradient boosting. Hyper-parameters are fixed a priori at
conservative, well-regularised values rather than tuned, because the aim is to
compare interpretable model classes on a fair footing and the sample size per
trip is large.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from battery_heating import config as C

MODEL_LABELS: dict[str, str] = {
    "logistic": "Logistic regression",
    "ridge": "Ridge regression",
    "decision_tree": "Decision tree (depth 3)",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
}


def classifiers(random_state: int = C.RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Classifiers for the ``heater_on`` task."""
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=random_state),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=200, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=C.N_JOBS,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def regressors(random_state: int = C.RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Regressors for the ``heating_power`` task."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "decision_tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=200, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=C.N_JOBS,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def models_for(task_kind: str) -> dict[str, BaseEstimator]:
    if task_kind == "classification":
        return classifiers()
    if task_kind == "regression":
        return regressors()
    raise ValueError(f"Unknown task kind: {task_kind}")


def final_estimator(model: BaseEstimator) -> BaseEstimator:
    """Return the last step of a pipeline, or the model itself."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model
