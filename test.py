"""Build a Bayesian network from hand-specified edges and print its CPTs."""

from pathlib import Path

import bayes_net as bn
from process_data import HORIZON_HOURS

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
    ("wd", "pm25_future"),
    ("wspm", "pm25_future"),
    ("rain", "pm25_future"),
    ("pm10", "pm10_future"),
    ("pm10_change", "pm10_future"),
    ("wd", "pm10_future"),
    ("wspm", "pm10_future"),
    ("rain", "pm10_future"),
]

df = bn.load_data(STATIONS)
train, test = bn.split_by_date(df, SPLIT_DATE, horizon_hours=HORIZON_HOURS)
train = bn.Discretizer().fit_transform(train)
model = bn.build_network(train, EDGES)

for parent, child in model.edges():
    print(f"({parent}) -> ({child})")
print()

sections = []
for node in model.nodes():
    table = bn.cpt_table(model, node).T.round(4).reset_index()
    sections.append(f"## {node}\n\n{table.to_markdown(index=False)}\n")

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "CPT.md").write_text("# CPTs\n\n" + "\n".join(sections))
