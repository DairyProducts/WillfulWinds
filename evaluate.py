"""Compare six-hour PM persistence forecasts on the same elevated station-hours."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import bayes_net as bn
from process_data import HORIZON_HOURS, RAW_DIR, TARGETS, TREND_HOURS, process_station

STATIONS = ("Changping", "Nongzhanguan")
PERIODS = {
    "validation": (pd.Timestamp("2015-03-01"), pd.Timestamp("2016-03-01")),
    "test": (pd.Timestamp("2016-03-01"), pd.Timestamp("2017-03-01")),
}
WEATHER = ["wd", "wspm", "rain"]
WITH_WEATHER = list(dict.fromkeys(p for p, c in bn.EDGES if c in {f"{t}_future" for t in TARGETS}))
VARIANTS = ["pm_only", "weather_only", "pm_weather"]


def variants(target: str) -> dict[str, list[str]]:
    """Split the target's DAG parents into PM-only, weather-only and combined inputs."""
    parents = [p for p, c in bn.EDGES if c == target]
    return {
        "pm_only": [p for p in parents if p not in WEATHER],
        "weather_only": [p for p in parents if p in WEATHER],
        "pm_weather": parents,
    }
HORIZON = pd.Timedelta(hours=HORIZON_HOURS)
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_study_data() -> pd.DataFrame:
    """Build the fixed study features from raw data, including hours with missing fields."""
    columns = ["station", *WITH_WEATHER, "pm25_future", "pm10_future"]
    frames = []
    for station in STATIONS:
        path = RAW_DIR / f"PRSA_Data_{station}_20130301-20170228.csv"
        hourly = process_station(path).set_index("datetime")
        # The hourly grid starts three hours before a first trend can be observed.
        frames.append(hourly.iloc[TREND_HOURS:][columns])
    return pd.concat(frames).sort_index(kind="stable")


def complete_cases(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Use exactly the evidence needed by either variant and this target's outcome."""
    return data[["station", *WITH_WEATHER, target]].dropna()


def target_cpt(model, train, pollutant, inputs):
    """Attach observed training counts to fitted CPT rows for elevated starts."""
    target = f"{pollutant}_future"
    probabilities = bn.cpt_table(model, target).T.reset_index()
    if pollutant in inputs:
        probabilities = probabilities.loc[probabilities[pollutant] == "elevated"]
    counts = (
        train.assign(persisted=train[target].eq("elevated"))
        .groupby(inputs, observed=True)["persisted"]
        .agg(training_hours="size", persisted="sum")
        .reset_index()
    )
    table = probabilities[[*inputs, "elevated"]].rename(columns={"elevated": "p_elevated"})
    table = table.merge(counts, on=inputs, how="left", validate="one_to_one")
    table[["training_hours", "persisted"]] = table[["training_hours", "persisted"]].fillna(0).astype(int)
    return table


def rain_example(table, inputs):
    """Show dry and rainy CPT entries with the other model inputs held fixed."""
    context = [col for col in inputs if col != "rain"]
    # Prefer a context with training support in both rows, without using scores.
    support = table.groupby(context, observed=True)["training_hours"].min()
    return table.set_index(context).loc[[support.idxmax()]].reset_index()


def fit_target(train, scoring, pollutant):
    target = f"{pollutant}_future"
    complete_training = complete_cases(train, target)
    # The weather-only network also estimates persistence conditional on an elevated start.
    training = complete_training.loc[complete_training[pollutant] == "elevated"]
    cases = complete_cases(scoring, target)
    cases = cases.loc[cases[pollutant] == "elevated"]
    if training.empty or cases.empty:
        raise ValueError(f"No elevated starts to score for {pollutant}.")

    coverage = train.groupby("station").size().rename("possible_hours").to_frame()
    coverage["usable_hours"] = complete_training.groupby("station").size().reindex(coverage.index, fill_value=0)
    coverage["usable_fraction"] = coverage["usable_hours"] / coverage["possible_hours"]
    coverage["elevated_training_starts"] = training.groupby("station").size().reindex(coverage.index, fill_value=0)

    predictions = cases[["station"]].rename_axis("issue_time").reset_index()
    predictions["outcome_time"] = predictions["issue_time"] + HORIZON
    predictions["pollutant"] = pollutant
    predictions["observed_elevated"] = cases[target].eq("elevated").to_numpy(dtype=int)
    predictions["p_training_rate"] = training[target].eq("elevated").mean()
    tables, notes = [], []
    for name, inputs in variants(target).items():
        model = bn.build_network(training, [(p, target) for p in inputs], prior="K2")
        forecasts = bn.predict_proba(model, cases[inputs], target)
        predictions[f"p_{name}"] = forecasts["elevated"].to_numpy()
        cpt = target_cpt(model, training, pollutant, inputs)
        tables.append(cpt.assign(pollutant=pollutant, model=name))
        arrows = "\n".join(f"    {a} --> {b}" for a, b in model.edges())
        notes.append(f"### {pollutant} · {name}\n\n```mermaid\nflowchart LR\n{arrows}\n```")
        if "rain" in inputs:
            example = rain_example(cpt, inputs)
            notes.append("Matched dry and rainy training cases\n\n" + example.to_markdown(index=False, floatfmt=".4f"))
    return predictions, coverage.reset_index().assign(pollutant=pollutant), tables, notes


def score_group(rows, start, block_draws):
    """Score paired forecasts. Resample common seven-day blocks by forecast time."""
    outcome = rows["observed_elevated"].to_numpy()
    losses = {
        name: (rows[f"p_{name}"].to_numpy() - outcome) ** 2
        for name in ["training_rate", *VARIANTS]
    }
    block_ids = ((rows["issue_time"] - start) // pd.Timedelta(days=7)).to_numpy(dtype=int)
    block_count = block_draws.shape[1]
    counts = np.bincount(block_ids, minlength=block_count)
    n_drawn = counts[block_draws].sum(axis=1)
    nonempty = n_drawn > 0
    result = {
        "elevated_starts": len(rows), "persisted": int(outcome.sum()),
        "persistence_rate": outcome.mean(),
        **{f"brier_{name}": loss.mean() for name, loss in losses.items()},
    }
    contrasts = {
        "gain": losses["pm_only"] - losses["pm_weather"],
        "weather_only_gain": losses["training_rate"] - losses["weather_only"],
    }
    for name, difference in contrasts.items():
        sums = np.bincount(block_ids, weights=difference, minlength=block_count)
        gains = sums[block_draws].sum(axis=1)[nonempty] / n_drawn[nonempty]
        result[name] = difference.mean()
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.quantile(gains, [0.025, 0.975])
    return result


def run(split: str, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    start, end = PERIODS[split]
    raw_train, later = bn.split_by_date(load_study_data(), start, horizon_hours=HORIZON_HOURS)
    bins = bn.Discretizer().fit(raw_train)
    train = bins.transform(raw_train)
    scoring = bins.transform(later.loc[later.index + HORIZON < end])
    # Include calendar weeks with zero events. Both stations share each sampled block.
    weeks = len(pd.date_range(start, end, freq="7D", inclusive="left"))
    draws = np.random.default_rng(797).integers(weeks, size=(5000, weeks))
    predictions, coverages, cpts, metrics = [], [], [], []
    cuts = pd.DataFrame([
        {"feature": feature, "cut_points": ", ".join(f"{v:g}" for v in values)}
        for feature, values in bins.thresholds().items()
    ])
    notes = [
        f"# {split.capitalize()} forecast models",
        f"Forecasts start {start.date()}. Training outcomes precede that date. "
        f"Scored outcomes precede {end.date()}.",
        "## Pooled training cut points\n\n" + cuts.to_markdown(index=False),
        "## Fitted networks",
        "All three models learn from the same currently elevated training starts. "
        "The weather-only model uses wind direction, wind speed and rain as its parents. "
        "Each arrow points from a forecast-time observation to the six-hour outcome. "
        "The target probability is (persisted + 1) / (training_hours + 2) with K2 smoothing.",
    ]
    for pollutant in TARGETS:
        rows, coverage, tables, model_notes = fit_target(train, scoring, pollutant)
        predictions.append(rows)
        coverages.append(coverage)
        cpts.extend(tables)
        notes.extend(model_notes)
        groups = [("Pooled", rows), *rows.groupby("station", sort=True)]
        for station, group in groups:
            metrics.append({
                "pollutant": pollutant, "station": station,
                "elevated_above_ug_m3": bins.edges_[pollutant][0],
                **score_group(group, start, draws),
            })

    directory = output_dir / split
    directory.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(metrics)
    summary.to_csv(directory / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    pd.concat(coverages, ignore_index=True).to_csv(directory / "training_coverage.csv", index=False)
    pd.concat(cpts, ignore_index=True).to_csv(directory / "target_cpts.csv", index=False)
    (directory / "models.md").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=PERIODS, default="validation")
    args = parser.parse_args()
    summary = run(args.split)
    columns = ["pollutant", "elevated_starts", "brier_training_rate", "brier_pm_only",
               "brier_weather_only", "brier_pm_weather", "gain", "weather_only_gain"]
    print(summary.loc[summary["station"] == "Pooled", columns].to_string(index=False, float_format="{:.5f}".format))
    print(f"Saved {args.split} results to {OUTPUT_DIR / args.split}")
