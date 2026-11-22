import argparse
from pathlib import Path
from typing import TextIO

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"

RENAME = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "CO": "co",
    "O3": "o3",
    "TEMP": "temp",
    "PRES": "pres",
    "DEWP": "dewp",
    "RAIN": "rain",
    "WSPM": "wspm",
}

WIND_GROUPS = {
    "NW": "N", "NNW": "N", "N": "N", "NNE": "N",
    "NE": "E", "ENE": "E", "E": "E", "ESE": "E",
    "SE": "S", "SSE": "S", "S": "S", "SSW": "S",
    "SW": "W", "WSW": "W", "W": "W", "WNW": "W",
}

TARGETS = ("pm25", "pm10")
HORIZON_HOURS = 6
TREND_HOURS = 3


def load_raw(path: Path | TextIO) -> pd.DataFrame:
    """Read one raw station file onto a complete hourly time index.

    Args:
        path: A raw ``PRSA_Data_*.csv`` path or open text stream.

    Returns:
        DataFrame indexed by hourly ``datetime`` with renamed columns. Missing
        hours, if any, appear as rows of NA.
    """
    raw = pd.read_csv(path)
    raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    df = raw.rename(columns=RENAME).set_index("datetime").sort_index()
    df = df.drop(columns=["No", "year", "day"]).asfreq("h")
    df["station"] = raw["station"].iloc[0]
    index = pd.DatetimeIndex(df.index)
    df["month"] = index.month
    df["hour"] = index.hour
    return df


def add_features(df: pd.DataFrame, horizon: int, trend_hours: int) -> pd.DataFrame:
    """Group wind direction and add trend and future pollution columns.

    Args:
        df: Hourly station data from ``load_raw``.
        horizon: Hours ahead for the ``*_future`` columns.
        trend_hours: Hours back for the ``*_change`` columns.

    Returns:
        A new DataFrame with the derived columns added.
    """
    out = df.copy()
    out["wd"] = out["wd"].map(WIND_GROUPS)
    for col in TARGETS:
        out[f"{col}_change"] = out[col] - out[col].shift(trend_hours)
        out[f"{col}_future"] = out[col].shift(-horizon)
    return out


def process_station(
    path: Path | TextIO, horizon: int = HORIZON_HOURS, trend_hours: int = TREND_HOURS
) -> pd.DataFrame:
    """Build hourly features for one station, preserving missing observations.

    Args:
        path: A raw station CSV path or open text stream.
        horizon: Hours ahead for the ``*_future`` columns.
        trend_hours: Hours back for the ``*_change`` columns.

    Returns:
        Station data with ``datetime`` as a column. Callers can drop missing
        values after choosing the predictors and outcome for an experiment.
    """
    df = add_features(load_raw(path), horizon, trend_hours)
    columns = ["station", "month", "hour", *RENAME.values(), "wd"]
    columns += [f"{c}_{s}" for s in ("change", "future") for c in TARGETS]
    return df[columns].reset_index()


def main() -> None:
    """Process every raw station file and print a row count per station."""
    parser = argparse.ArgumentParser(description="Process raw station files into data/.")
    parser.add_argument("--horizon", type=int, default=HORIZON_HOURS)
    parser.add_argument("--trend-hours", type=int, default=TREND_HOURS)
    args = parser.parse_args()

    paths = sorted(RAW_DIR.glob("PRSA_Data_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}")
    for path in paths:
        df = process_station(path, args.horizon, args.trend_hours)
        station = df["station"].iloc[0]
        df.to_csv(DATA_DIR / f"{station}.csv", index=False)
        print(f"{station}: {len(df)} rows")


if __name__ == "__main__":
    main()
