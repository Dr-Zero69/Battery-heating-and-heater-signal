"""Unit tests for data loading, cleaning and resampling on small synthetic inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_heating import config as C
from battery_heating import data, features

HEADER = (
    "Time [s];Velocity [km/h]]];Battery Voltage [V];Battery Current [A];Battery Temperature [°C];"
    "SoC [%];displayed SoC [%];max. SoC [%);Heating Power CAN [kW];Heating Power LIN [W];"
    "Requested Heating Power [W];AirCon Power [kW];Heater Signal;Heater Voltage [V];Heater Current [A];"
    "Ambient Temperature [°C];Cabin Temperature Sensor [°C];Temperature Vent right [°C] ;Temperature Vent right [°C]"
)


def _write_trip(path: Path, n: int, heater_on: bool, lin_w: float = 7000.0) -> None:
    rows = []
    for i in range(n):
        t = i / 10
        cur = 18.0 if heater_on else 0.0
        rows.append(
            f"{t};50;390;-10;20;80;100;90;{lin_w / 1000 if heater_on else 0};{lin_w if heater_on else 0};"
            f"7000;0;1;390;{cur};5;15;10;10"
        )
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="latin-1")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    _write_trip(tmp_path / "TripA01.csv", 40, heater_on=False)
    _write_trip(tmp_path / "TripB01.csv", 40, heater_on=True)
    _write_trip(tmp_path / "TripB02.csv", 40, heater_on=True, lin_w=38000.0)  # implausible
    overview = pd.DataFrame(
        {
            "Trip": ["TripA01", "TripB01", "TripB02"],
            "Date": ["2019-06-25_13-21-14", "2019-12-01_08-00-00", "2019-12-02_08-00-00"],
            "Route/Area": ["X", "Y", "Y"],
            "Weather": ["sunny", "dark", "dark"],
            "Battery Temperature (Start) [°C]": [20, 5, 5],
            "Battery Temperature (End)": [21, 6, 6],
            "Battery State of Charge (Start)": ["80%", "90%", "70%"],
            "Battery State of Charge (End)": ["75%", "80%", "60%"],
            "Unnamed: 8": ["5%", "10%", "10%"],
            "Ambient Temperature (Start) [°C]": [25, 3, 1],
            "Target Cabin Temperature": [23.0, 22.0, 22.0],
            "Distance [km]": [7.4, 10.0, 9.0],
            "Duration [min]": [16.8, 20.0, 18.0],
            "Unnamed: 13": [np.nan] * 3,
            "Fan": ["Automatic, Level 1"] * 3,
            "Note": [np.nan, "+70 kg", np.nan],
        }
    )
    overview.to_csv(tmp_path / "Overview.csv", index=False, encoding="latin-1")
    return tmp_path


def test_normalise_columns_fixes_known_typos():
    cols = normalised = data.normalise_columns(
        ["Velocity [km/h]]]", "max. SoC [%)", " Heater Signal ", "Temperature Vent right [°C] "]
    )
    assert cols[0] == "Velocity [km/h]"
    assert normalised[1] == "max. SoC [%]"
    assert normalised[2] == "Heater Signal"
    assert normalised[3] == "Temperature Vent right [°C]"


def test_load_trip_drops_duplicate_columns(raw_dir: Path):
    df = data.load_trip(raw_dir / "TripB01.csv")
    assert df.columns.is_unique
    assert df[C.TRIP].iloc[0] == "TripB01"
    assert C.VELOCITY in df.columns


def test_load_overview_parses_percentages(raw_dir: Path):
    ov = data.load_overview(raw_dir / "Overview.csv")
    assert ov["soc_start_pct"].tolist() == [80.0, 90.0, 70.0]
    assert ov[C.SEASON].tolist() == ["summer", "winter", "winter"]
    assert ov["note"].iloc[0] == ""


def test_resample_trip_averages_to_one_hertz():
    trip = pd.DataFrame({C.TRIP: "T", C.TIME: np.arange(0, 3, 0.1), C.HEATER_I: np.r_[np.zeros(15), np.ones(15)]})
    out = data.resample_trip(trip, seconds=1)
    assert out[C.TIME_SINCE_START].tolist() == [0, 1, 2]
    assert out[C.HEATER_I].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_build_analysis_table_masks_implausible_power_and_defines_targets(raw_dir: Path):
    table = data.build_analysis_table(raw_dir, raw_dir / "Overview.csv")
    # TripB02 had only implausible heating power, so all its rows are dropped
    assert set(table[C.TRIP]) == {"TripA01", "TripB01"}
    assert table.loc[table[C.TRIP] == "TripA01", C.HEATER_ON].eq(0).all()
    assert table.loc[table[C.TRIP] == "TripB01", C.HEATER_ON].eq(1).all()
    assert table[C.HEATING_POWER_W].max() <= C.HEATING_POWER_MAX_W
    assert C.TARGET_CABIN_T in table.columns


def test_derived_features_and_leakage_guard(raw_dir: Path):
    table = features.add_derived_features(data.build_analysis_table(raw_dir, raw_dir / "Overview.csv"))
    assert table[C.BAT_POWER_KW].iloc[0] == pytest.approx(390 * -10 / 1000)
    assert table[C.CABIN_DEFICIT].iloc[0] == pytest.approx(23.0 - 15.0)
    with pytest.raises(ValueError, match="Leakage"):
        features.check_no_leakage([C.HEATER_I, C.VELOCITY])
    for fs in C.FEATURE_SETS.values():
        features.check_no_leakage(fs)
