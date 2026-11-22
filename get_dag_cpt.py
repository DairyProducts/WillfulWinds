"""Build a Bayesian network from hand-specified edges and print its CPTs."""

from pathlib import Path

import pandas as pd

import bayes_net as bn
from process_data import HORIZON_HOURS, TREND_HOURS

OUTPUT_DIR = Path(__file__).parent / "outputs"
STATIONS = ["Nongzhanguan", "Changping"]
SPLIT_DATE = "2016-03-01"

EDGES = [
    ("month", "temp"),
    ("hour", "temp"),
    ("month", "dewp"),
    ("month", "pres"),
    ("dewp", "rain"),
    ("hour", "wspm"),
    ("station", "wspm"),
    ("month", "wd"),
    ("month", "pm25"),
    ("month", "so2"),
    ("month", "co"),
    ("hour", "no2"),
    ("temp", "o3"),
    ("no2", "o3"),
    ("pm25", "pm10"),
    ("hour", "pm25_change"),
    ("hour", "pm10_change"),
    ("pm25", "pm25_future"),
    ("pm25_change", "pm25_future"),
    ("pm10", "pm25_future"),
    ("pm10_change", "pm25_future"),
    ("wd", "pm25_future"),
    ("wspm", "pm25_future"),
    ("rain", "pm25_future"),
    ("pm10", "pm10_future"),
    ("pm10_change", "pm10_future"),
    ("pm25", "pm10_future"),
    ("pm25_change", "pm10_future"),
    ("wd", "pm10_future"),
    ("wspm", "pm10_future"),
    ("rain", "pm10_future"),
]
TARGETS = ["pm25_future", "pm10_future"]

frames = [
    pd.read_csv(bn.DATA_DIR / f"{station}.csv", parse_dates=["datetime"], index_col="datetime").iloc[TREND_HOURS:]
    for station in STATIONS
]
df = pd.concat(frames).sort_index(kind="stable")
train, test = bn.split_by_date(df, SPLIT_DATE, horizon_hours=HORIZON_HOURS)
train = bn.Discretizer().fit_transform(train)
model = bn.build_network(train.dropna(), EDGES, prior="K2")

for target in TARGETS:
    parents = list(model.get_parents(target))
    rows = train[[*parents, target]].dropna()
    refit = bn.build_network(rows, [(p, target) for p in parents], prior="K2")
    model.remove_cpds(target)
    model.add_cpds(bn.get_cpd(refit, target))

for parent, child in model.edges():
    print(f"({parent}) -> ({child})")
print()

sections = []
for node in model.nodes():
    table = bn.cpt_table(model, node).T.round(4).reset_index()
    sections.append(f"## {node}\n\n{table.to_markdown(index=False)}\n")

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "CPT.md").write_text("# CPTs\n\n" + "\n".join(sections))
bn.draw_network(
    model, OUTPUT_DIR / "dag.png", highlight=TARGETS, same_row=[["pm25", "pm10"]]
)
