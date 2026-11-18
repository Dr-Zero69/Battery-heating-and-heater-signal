"""Feature engineering and feature-matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_heating import config as C


def add_derived_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated derived columns to the analysis table.

    * ``battery_power_kw``: instantaneous battery power (V x I / 1000).
    * ``cabin_temp_deficit_c``: target cabin temperature minus measured cabin
      temperature; the error signal a cabin-climate controller acts on.
    * ``ambient_temp_deficit_c``: target cabin temperature minus ambient
      temperature; a proxy for the steady-state heat load on the cabin.
    """
    df = table.copy()
    df[C.BAT_POWER_KW] = df[C.BAT_V] * df[C.BAT_I] / 1000.0
    df[C.CABIN_DEFICIT] = df[C.TARGET_CABIN_T] - df[C.CABIN_T]
    df[C.AMBIENT_DEFICIT] = df[C.TARGET_CABIN_T] - df[C.AMBIENT_T]
    return df


def check_no_leakage(features: list[str]) -> None:
    """Raise if any feature directly measures the heater."""
    leaked = sorted(set(features) & set(C.LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Leakage columns present in feature list: {leaked}")


def make_xy(
    table: pd.DataFrame,
    task_name: str,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, groups)`` for a task and feature set.

    Rows with any missing feature are dropped. The number dropped is small and
    is recorded in the results registry.
    """
    task = C.TASKS[task_name]
    features = C.FEATURE_SETS[feature_set]
    check_no_leakage(features)

    df = table.dropna(subset=[*features, task.target])
    X = df[features].astype(float)
    y = df[task.target].astype(float if task.kind == "regression" else int)
    groups = df[C.TRIP]
    return X, y, groups


def feature_labels(features: list[str]) -> list[str]:
    return [C.FEATURE_LABELS.get(f, f) for f in features]


def trip_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per trip with the quantities used in trip-level figures."""
    g = table.groupby([C.TRIP, C.SEASON], observed=True)
    out = g.agg(
        duration_min=(C.TIME_SINCE_START, lambda s: (s.max() + 1) / 60),
        ambient_temp_c=(C.AMBIENT_T, "mean"),
        cabin_temp_c=(C.CABIN_T, "mean"),
        target_cabin_temp_c=(C.TARGET_CABIN_T, "first"),
        heater_on_share=(C.HEATER_ON, "mean"),
        heating_power_mean_w=(C.HEATING_POWER_W, "mean"),
        heating_power_max_w=(C.HEATING_POWER_W, "max"),
        heating_energy_kwh=(C.HEATING_POWER_W, lambda s: s.sum() / 3600 / 1000),
        battery_temp_c=(C.BAT_T, "mean"),
        n_seconds=(C.TIME_SINCE_START, "size"),
    ).reset_index()
    switches = table.groupby(C.TRIP)[C.HEATER_ON].apply(lambda s: int(s.diff().abs().sum()))
    out["heater_switches"] = out[C.TRIP].map(switches).astype(int)
    return out


def heater_signal_by_trip(raw_all: pd.DataFrame, overview: pd.DataFrame) -> pd.DataFrame:
    """Trip-level view of the original ``Heater Signal`` flag across all 70 trips.

    The flag is constant within every trip, so the natural unit of analysis for
    it is the trip rather than the 10 Hz sample.
    """
    g = raw_all.groupby(C.TRIP)
    out = pd.DataFrame(
        {
            "heater_signal_mean": g[C.HEATER_SIGNAL].mean(),
            "heater_signal_nunique": g[C.HEATER_SIGNAL].nunique(),
            "ambient_temp_c": g[C.AMBIENT_T].mean(),
            "cabin_temp_c": g[C.CABIN_T].mean(),
            "heater_instrumented": g[C.HEATER_I].apply(lambda s: bool(s.notna().any())),
            "n_rows": g.size(),
        }
    ).reset_index()
    out = out.merge(overview[[C.TRIP, C.SEASON, C.TARGET_CABIN_T, "weather"]], on=C.TRIP, how="left")
    out["heater_signal"] = np.where(out["heater_signal_mean"] >= 0.5, 1, 0)
    return out
