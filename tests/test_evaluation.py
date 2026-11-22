import unittest
from io import StringIO

import pandas as pd

import bayes_net as bn
from evaluate import complete_cases, fit_target
from process_data import process_station


class ForecastDataTest(unittest.TestCase):
    def test_unused_fields_and_other_future_can_be_missing(self):
        times = pd.date_range("2014-01-01", periods=14, freq="h")
        raw = pd.DataFrame({
            "No": range(14), "year": times.year, "month": times.month,
            "day": times.day, "hour": times.hour, "station": "Example",
            "PM2.5": range(14), "PM10": range(100, 114), "wd": "N",
            "WSPM": 1.0, "RAIN": 0.0, "SO2": float("nan"), "NO2": 1,
            "CO": 1, "O3": 1, "TEMP": 1, "PRES": 1, "DEWP": 1,
        })
        raw.loc[10, "PM10"] = float("nan")
        raw.loc[5, "WSPM"] = float("nan")
        raw = raw.drop(index=6)  # Exercise a missing clock hour during the forecast.
        data = process_station(StringIO(raw.to_csv(index=False))).set_index("datetime")
        self.assertEqual(len(data), len(times))
        pm25 = complete_cases(data, "pm25_future")
        self.assertEqual(pm25.at[times[4], "pm25_future"], 10)
        self.assertEqual(pm25.at[times[4], "pm25_change"], 3)
        self.assertNotIn(times[4], complete_cases(data, "pm10_future").index)
        self.assertNotIn(times[5], pm25.index)

    def test_outcome_at_cutoff_is_excluded_from_training_at_both_stations(self):
        times = pd.date_range("2016-02-29 17:00", periods=9, freq="h")
        data = pd.DataFrame(index=times.repeat(2))
        start = pd.Timestamp("2016-03-01")
        train, scoring = bn.split_by_date(data, start, horizon_hours=6)
        self.assertEqual(train.index.tolist(), [times[0], times[0]])
        self.assertEqual(scoring.index.tolist(), [times[7], times[7], times[8], times[8]])

    def test_weather_only_estimates_persistence_for_elevated_starts(self):
        high, normal = "elevated", "normal"
        data = pd.DataFrame({
            "station": "Example", "pm25": [high] * 4 + [normal] * 4,
            "pm10": [high, normal] * 4, "pm25_change": ["rising", "steady"] * 4,
            "pm10_change": "steady", "wd": "N", "wspm": "low",
            "rain": ["dry", "dry", "dry", "rain"] + ["dry"] * 4,
            "pm25_future": [high, high] + [normal] * 6,
        }, index=pd.date_range("2014-01-01", periods=8, freq="h"))
        predictions, _, tables, _ = fit_target(data, data.iloc[[0, 1, 4]], "pm25")
        # Two of three elevated, dry training starts persisted (one K2 count per state).
        self.assertEqual(len(predictions), 2)
        for p in predictions["p_weather_only"]:
            self.assertAlmostEqual(p, 3 / 5)
        self.assertTrue(predictions["p_training_rate"].eq(2 / 4).all())
        for table in tables:
            self.assertEqual(table["training_hours"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
