"""Project-wide constants: paths, column names, feature sets, and modelling settings.

Everything that a reader might want to change lives here so that no module
hard-codes a path or a column name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
RESULTS_JSON = REPORTS / "results.json"

OVERVIEW_FILE = DATA_RAW / "Overview.csv"
TRIP_GLOB = "Trip*.csv"

# --------------------------------------------------------------------------- #
# Raw column names (after normalisation in ``data.normalise_columns``)
# --------------------------------------------------------------------------- #
TIME = "Time [s]"
VELOCITY = "Velocity [km/h]"
ELEVATION = "Elevation [m]"
THROTTLE = "Throttle [%]"
TORQUE = "Motor Torque [Nm]"
ACCEL = "Longitudinal Acceleration [m/s^2]"
REGEN = "Regenerative Braking Signal"
BAT_V = "Battery Voltage [V]"
BAT_I = "Battery Current [A]"
BAT_T = "Battery Temperature [°C]"
BAT_T_MAX = "max. Battery Temperature [°C]"
SOC = "SoC [%]"
SOC_DISP = "displayed SoC [%]"
HEAT_P_CAN = "Heating Power CAN [kW]"
HEAT_P_LIN = "Heating Power LIN [W]"
HEAT_P_REQ = "Requested Heating Power [W]"
AIRCON_P = "AirCon Power [kW]"
HEATER_SIGNAL = "Heater Signal"
HEATER_V = "Heater Voltage [V]"
HEATER_I = "Heater Current [A]"
AMBIENT_T = "Ambient Temperature [°C]"
COOLANT_T_CORE = "Coolant Temperature Heatercore [°C]"
COOLANT_T_REQ = "Requested Coolant Temperature [°C]"
COOLANT_T_IN = "Coolant Temperature Inlet [°C]"
HX_T = "Heat Exchanger Temperature [°C]"
CABIN_T = "Cabin Temperature Sensor [°C]"

# Overview (trip metadata) columns after cleaning
TRIP = "trip"
SEASON = "season"
TARGET_CABIN_T = "target_cabin_temp_c"
AMBIENT_T_START = "ambient_temp_start_c"

# --------------------------------------------------------------------------- #
# Engineered / tidy column names used in the analysis table
# --------------------------------------------------------------------------- #
TIME_SINCE_START = "time_since_start_s"
BAT_POWER_KW = "battery_power_kw"
CABIN_DEFICIT = "cabin_temp_deficit_c"  # target cabin temperature - cabin sensor
AMBIENT_DEFICIT = "ambient_temp_deficit_c"  # target cabin temperature - ambient
HEATER_ON = "heater_on"  # binary target: heater current > 0 A
HEATING_POWER_W = "heating_power_w"  # continuous target: Heating Power LIN [W]

# Physical plausibility limit for the high-voltage PTC heater (nominal 7 kW unit).
# Readings above this are sensor faults (TripB02 reports up to 38.5 kW for ~2.6 min).
HEATING_POWER_MAX_W = 8000.0

# Resampling period for the analysis table. The raw logs are 10 Hz; adjacent
# samples are almost identical, so 1 Hz keeps the dynamics while reducing
# autocorrelation and compute by an order of magnitude.
RESAMPLE_SECONDS = 1

# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
# Exogenous features: driving state, battery state, environment and cabin
# set-point. None of them is a direct measurement of the heater or of the
# heating-loop temperatures it controls. The derived quantities battery power
# (= V x I) and the two temperature deficits (= target - measured) are exact
# functions of features already in the list; they are kept for descriptive
# use but excluded here so that linear coefficients remain identifiable.
EXOGENOUS_FEATURES: list[str] = [
    TIME_SINCE_START,
    VELOCITY,
    ELEVATION,
    THROTTLE,
    TORQUE,
    ACCEL,
    REGEN,
    BAT_V,
    BAT_I,
    BAT_T,
    SOC,
    AIRCON_P,
    AMBIENT_T,
    CABIN_T,
    TARGET_CABIN_T,
]

# Thermal-loop features are downstream of the heater (the heater warms the
# coolant and the heat exchanger). They are informative for state estimation
# but are partly a consequence of the target, so they are reported separately.
THERMAL_LOOP_FEATURES: list[str] = [
    HX_T,
    COOLANT_T_CORE,
    COOLANT_T_IN,
    COOLANT_T_REQ,
]

# Battery voltage and current respond to the heater's own electrical draw
# (about 18 A at 7 kW), so a stricter feature set without them is also
# evaluated to show how much performance survives on purely non-electrical
# information.
NON_ELECTRICAL_FEATURES: list[str] = [f for f in EXOGENOUS_FEATURES if f not in (BAT_V, BAT_I)]

FEATURE_SETS: dict[str, list[str]] = {
    "exogenous": EXOGENOUS_FEATURES,
    "exogenous-electrical": NON_ELECTRICAL_FEATURES,
    "exogenous+thermal": EXOGENOUS_FEATURES + THERMAL_LOOP_FEATURES,
}

FEATURE_SET_LABELS: dict[str, str] = {
    "exogenous": "exogenous",
    "exogenous-electrical": "without battery V, I",
    "exogenous+thermal": "+ thermal loop",
}

# Columns that must never be used as predictors because they measure the heater
# itself (direct target leakage).
LEAKAGE_COLUMNS: list[str] = [
    HEATER_SIGNAL,
    HEATER_V,
    HEATER_I,
    HEAT_P_CAN,
    HEAT_P_LIN,
    HEAT_P_REQ,
]

# Short, publication-friendly labels for plots and tables.
FEATURE_LABELS: dict[str, str] = {
    TIME_SINCE_START: "Time since trip start (s)",
    VELOCITY: "Velocity (km/h)",
    ELEVATION: "Elevation (m)",
    THROTTLE: "Throttle (%)",
    TORQUE: "Motor torque (Nm)",
    ACCEL: "Longitudinal acceleration (m/s²)",
    REGEN: "Regenerative braking (0/1)",
    BAT_V: "Battery voltage (V)",
    BAT_I: "Battery current (A)",
    BAT_POWER_KW: "Battery power (kW)",
    BAT_T: "Battery temperature (°C)",
    SOC: "State of charge (%)",
    AIRCON_P: "Air-conditioning power (kW)",
    AMBIENT_T: "Ambient temperature (°C)",
    CABIN_T: "Cabin temperature (°C)",
    TARGET_CABIN_T: "Target cabin temperature (°C)",
    CABIN_DEFICIT: "Cabin temperature deficit (°C)",
    AMBIENT_DEFICIT: "Ambient temperature deficit (°C)",
    HX_T: "Heat-exchanger temperature (°C)",
    COOLANT_T_CORE: "Coolant temperature, heater core (°C)",
    COOLANT_T_IN: "Coolant temperature, inlet (°C)",
    COOLANT_T_REQ: "Requested coolant temperature (°C)",
    HEATER_ON: "Heater on",
    HEATING_POWER_W: "Heating power (W)",
}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = -1
SHAP_SAMPLE_SIZE = 3000
PERMUTATION_REPEATS = 5


@dataclass(frozen=True)
class Task:
    """A supervised learning task defined on the analysis table."""

    name: str
    target: str
    kind: str  # "classification" | "regression"
    primary_metric: str
    description: str = field(default="")


TASKS: dict[str, Task] = {
    "heater_on": Task(
        name="heater_on",
        target=HEATER_ON,
        kind="classification",
        primary_metric="roc_auc",
        description="Is the high-voltage cabin heater drawing current at this second?",
    ),
    "heating_power": Task(
        name="heating_power",
        target=HEATING_POWER_W,
        kind="regression",
        primary_metric="r2",
        description="Instantaneous electrical heating power in watts (0 W when off).",
    ),
}
