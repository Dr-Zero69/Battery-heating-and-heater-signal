"""Loading, cleaning and resampling of the raw trip logs and the trip overview.

The raw files are semicolon-separated, Latin-1 encoded CSVs sampled at 10 Hz.
Column headers differ slightly between trips (trailing spaces, a duplicated
"Temperature Vent right" column, one trip with a mis-typed velocity header), so
every file passes through :func:`normalise_columns` before anything else happens.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from battery_heating import config as C

log = logging.getLogger(__name__)

_ENCODING = "latin-1"
_SEP = ";"


# --------------------------------------------------------------------------- #
# Overview (trip metadata)
# --------------------------------------------------------------------------- #
def load_overview(path: Path = C.OVERVIEW_FILE) -> pd.DataFrame:
    """Load and clean ``Overview.csv`` into one row per trip.

    Blank spacer rows and the empty unnamed column are removed, percentages are
    converted to numbers, and the ``Unnamed: 8`` column (state-of-charge used
    during the trip) is renamed ``soc_used_pct``.
    """
    raw = pd.read_csv(path, encoding=_ENCODING)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed: 13")])
    raw = raw.dropna(subset=["Trip"]).copy()

    rename = {
        "Trip": C.TRIP,
        "Date": "date",
        "Route/Area": "route",
        "Weather": "weather",
        "Battery Temperature (Start) [°C]": "battery_temp_start_c",
        "Battery Temperature (End)": "battery_temp_end_c",
        "Battery State of Charge (Start)": "soc_start_pct",
        "Battery State of Charge (End)": "soc_end_pct",
        "Unnamed: 8": "soc_used_pct",
        "Ambient Temperature (Start) [°C]": C.AMBIENT_T_START,
        "Target Cabin Temperature": C.TARGET_CABIN_T,
        "Distance [km]": "distance_km",
        "Duration [min]": "duration_min",
        "Fan": "fan",
        "Note": "note",
    }
    df = raw.rename(columns=rename)
    for col in ("soc_start_pct", "soc_end_pct", "soc_used_pct"):
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d_%H-%M-%S", errors="coerce")
    df[C.SEASON] = np.where(df[C.TRIP].str.startswith("TripA"), "summer", "winter")
    df["note"] = df["note"].fillna("")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trip logs
# --------------------------------------------------------------------------- #
def normalise_columns(columns: list[str]) -> list[str]:
    """Return cleaned column names.

    * strips surrounding whitespace,
    * repairs the ``Velocity [km/h]]]`` typo found in one trip,
    * repairs the ``max. SoC [%)`` bracket typo present in every file,
    * normalises the degree sign, which Latin-1 decoding may leave as other glyphs.
    """
    out = []
    for c in columns:
        c = c.strip()
        c = re.sub(r"\]+$", "]", c)
        c = c.replace("[%)", "[%]")
        c = c.replace("�", "°").replace("Â°", "°")
        out.append(c)
    return out


def load_trip(path: Path) -> pd.DataFrame:
    """Load one raw trip file with normalised, de-duplicated column names."""
    df = pd.read_csv(path, sep=_SEP, encoding=_ENCODING)
    df.columns = normalise_columns(list(df.columns))
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df.insert(0, C.TRIP, path.stem)
    return df


def list_trip_files(raw_dir: Path = C.DATA_RAW) -> list[Path]:
    files = sorted(raw_dir.glob(C.TRIP_GLOB))
    if not files:
        raise FileNotFoundError(f"No trip CSV files found in {raw_dir}")
    return files


def load_all_trips(raw_dir: Path = C.DATA_RAW) -> pd.DataFrame:
    """Concatenate every trip file at the native 10 Hz resolution."""
    frames = [load_trip(p) for p in list_trip_files(raw_dir)]
    df = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Loaded %d trips, %d rows, %d columns", df[C.TRIP].nunique(), len(df), df.shape[1])
    return df


# --------------------------------------------------------------------------- #
# Analysis table
# --------------------------------------------------------------------------- #
def has_heater_instrumentation(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows belonging to trips that logged heater current."""
    per_trip = df.groupby(C.TRIP)[C.HEATER_I].apply(lambda s: s.notna().any())
    return df[C.TRIP].map(per_trip).astype(bool)


def resample_trip(trip: pd.DataFrame, seconds: int = C.RESAMPLE_SECONDS) -> pd.DataFrame:
    """Average one 10 Hz trip log into bins of ``seconds`` width.

    Numeric columns are averaged within each bin, so a binary signal becomes
    the fraction of the bin during which it was active. The bin is labelled by
    its left edge in seconds since the first sample of the trip.
    """
    t = trip[C.TIME] - trip[C.TIME].iloc[0]
    bins = np.floor(t / seconds).astype(int) * seconds
    numeric = trip.select_dtypes(include=np.number).columns
    out = trip[numeric].groupby(bins).mean()
    out.index.name = C.TIME_SINCE_START
    out = out.reset_index()
    out.insert(0, C.TRIP, trip[C.TRIP].iloc[0])
    return out


def build_analysis_table(
    raw_dir: Path = C.DATA_RAW,
    overview_path: Path = C.OVERVIEW_FILE,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the tidy 1 Hz analysis table used by every downstream step.

    Steps
    -----
    1. Load every trip and keep only trips whose heater current was logged
       (12 summer and 38 winter trips).
    2. Forward-fill short gaps in state-of-charge within each trip.
    3. Mask physically implausible heating-power readings.
    4. Resample each trip to 1 Hz.
    5. Define the targets: ``heater_on`` (heater current > 0 A for more than
       half of the bin) and ``heating_power_w`` (LIN-bus heating power).
    6. Attach trip metadata (season, target cabin temperature, route, weather).
    """
    raw = load_all_trips(raw_dir) if raw is None else raw
    mask = has_heater_instrumentation(raw)
    dropped = sorted(raw.loc[~mask, C.TRIP].unique())
    log.info("Dropping %d trips without heater instrumentation: %s", len(dropped), dropped)
    raw = raw.loc[mask].copy()

    fill_cols = [C.SOC, C.SOC_DISP]
    raw[fill_cols] = raw.groupby(C.TRIP)[fill_cols].transform(lambda s: s.ffill().bfill())

    bad = raw[C.HEAT_P_LIN] > C.HEATING_POWER_MAX_W
    log.info("Masking %d rows with heating power > %.0f W", int(bad.sum()), C.HEATING_POWER_MAX_W)
    raw.loc[bad, [C.HEAT_P_LIN, C.HEATER_I, C.HEATER_V, C.HEAT_P_CAN]] = np.nan

    resampled = pd.concat([resample_trip(g) for _, g in raw.groupby(C.TRIP, sort=True)], ignore_index=True)

    resampled[C.HEATER_ON] = (resampled[C.HEATER_I] > 0).astype(int)
    resampled[C.HEATING_POWER_W] = resampled[C.HEAT_P_LIN]

    overview = load_overview(overview_path)
    meta_cols = [C.TRIP, C.SEASON, C.TARGET_CABIN_T, C.AMBIENT_T_START, "route", "weather", "distance_km"]
    table = resampled.merge(overview[meta_cols], on=C.TRIP, how="left", validate="many_to_one")

    before = len(table)
    table = table.dropna(subset=[C.HEATER_I, C.HEAT_P_LIN]).reset_index(drop=True)
    log.info(
        "Analysis table: %d rows (%d dropped for missing target), %d trips",
        len(table),
        before - len(table),
        table[C.TRIP].nunique(),
    )
    return table


def analysis_table_path() -> Path:
    return C.DATA_PROCESSED / "analysis_table_1hz.parquet"


def save_analysis_table(table: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or analysis_table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def load_analysis_table(path: Path | None = None) -> pd.DataFrame:
    path = path or analysis_table_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m battery_heating prepare` first.")
    return pd.read_parquet(path)
