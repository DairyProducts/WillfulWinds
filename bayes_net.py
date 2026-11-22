import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.textpath import TextPath
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

logging.getLogger("pgmpy").setLevel(logging.WARNING)

DATA_DIR = Path(__file__).parent / "data"

Edge = tuple[str, str]


@dataclass(frozen=True)
class Bins:
    """Cut points and labels for discretizing one numeric column.

    Attributes:
        edges: Inner cut points in increasing order. Read as quantiles in
            [0, 1] when ``quantile`` is True, otherwise as raw values.
        labels: One label per bin, so ``len(edges) + 1`` labels.
        quantile: If True, compute cut points from training-data quantiles.
        reference: Column to take quantiles from. Defaults to the column
            itself. Use this so ``pm25_future`` shares the ``pm25`` threshold.
        right: If True, a value equal to a cut point falls in the lower bin,
            so ``elevated`` means strictly above it. If False, it falls in
            the upper bin, matching labels such as ``">=150"``.
    """

    edges: tuple[float, ...]
    labels: tuple[str, ...]
    quantile: bool = False
    reference: str | None = None
    right: bool = True

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.edges) + 1:
            raise ValueError("Bins needs exactly one more label than edges.")


def elevated(reference: str, q: float = 0.9) -> Bins:
    """Return a two-state bin that splits at the ``q`` quantile of ``reference``.

    Args:
        reference: Column whose training quantile sets the threshold.
        q: Quantile used as the threshold.

    Returns:
        Bins labeled ``normal`` and ``elevated``.
    """
    return Bins((q,), ("normal", "elevated"), quantile=True, reference=reference)


TERCILES = Bins((1 / 3, 2 / 3), ("low", "mid", "high"), quantile=True)
TREND = Bins((1 / 3, 2 / 3), ("falling", "steady", "rising"), quantile=True)

DEFAULT_BINS: dict[str, Bins] = {
    "pm25": elevated("pm25"),
    "pm25_future": elevated("pm25"),
    "pm10": elevated("pm10"),
    "pm10_future": elevated("pm10"),
    "pm25_change": TREND,
    "pm10_change": TREND,
    "so2": Bins((150.0,), ("<150", ">=150"), right=False),
    "no2": Bins((100.0,), ("<100", ">=100"), right=False),
    "co": Bins((5000.0,), ("<5000", ">=5000"), right=False),
    "o3": Bins((160.0,), ("<160", ">=160"), right=False),
    "temp": Bins((0.0, 20.0), ("<0", "0-20", ">20"), right=False),
    "pres": TERCILES,
    "dewp": Bins((16.0,), ("<16", ">=16"), right=False),
    "wspm": TERCILES,
    "rain": Bins((0.0,), ("dry", "rain")),
}


def available_stations(data_dir: Path = DATA_DIR) -> list[str]:
    """List stations that have a processed file.

    Args:
        data_dir: Folder holding the processed ``<Station>.csv`` files.

    Returns:
        Sorted station names.
    """
    return sorted(p.stem for p in data_dir.glob("*.csv"))


def select_columns(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Keep ``columns`` (all if None), then drop ``exclude``.

    Args:
        df: Input data.
        columns: Columns to keep. None keeps every column.
        exclude: Columns to drop after selection.

    Returns:
        A DataFrame with the selected columns.

    Raises:
        KeyError: If a requested column is not in ``df``.
    """
    keep = list(df.columns) if columns is None else list(columns)
    missing = set(keep) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not in data: {sorted(missing)}")
    drop = set(exclude or ())
    return df[[c for c in keep if c not in drop]]


def load_data(
    stations: Sequence[str] | None = None,
    columns: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load processed data for one or more stations.

    Args:
        stations: Station names, such as ``["Aotizhongxin"]``. None loads all.
        columns: Columns to keep. None keeps every column.
        exclude: Columns to drop.
        data_dir: Folder holding the processed files.

    Returns:
        Rows from all requested stations, indexed by ``datetime``, with NA
        rows dropped.

    Raises:
        FileNotFoundError: If a station has no processed file.
    """
    names = available_stations(data_dir) if stations is None else stations
    frames = []
    for name in names:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No processed file for station {name!r}")
        frames.append(pd.read_csv(path, parse_dates=["datetime"], index_col="datetime"))
    df = pd.concat(frames).sort_index(kind="stable")
    return select_columns(df, columns, exclude).dropna()


def split_by_date(
    df: pd.DataFrame, date: str | pd.Timestamp, *, horizon_hours: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by forecast time, keeping every training outcome before the cutoff.

    Args:
        df: Data indexed by ``datetime``.
        date: First timestamp of the test set.
        horizon_hours: Hours from the forecast time to the observed outcome.

    Returns:
        ``(train, test)``.
    """
    if horizon_hours < 0:
        raise ValueError("The forecast horizon must be nonnegative.")
    cutoff = pd.Timestamp(date)
    outcome_time = df.index + pd.Timedelta(hours=horizon_hours)
    return df[outcome_time < cutoff], df[df.index >= cutoff]


class Discretizer:
    """Turn numeric columns into labeled states using cut points fit on training data.

    Columns without a ``Bins`` entry pass through unchanged, so already
    discrete columns such as ``wd``, ``hour``, and ``station`` keep their values.

    Attributes:
        bins: Mapping of column name to ``Bins``.
        edges_: Cut points per column, set by ``fit``.
    """

    def __init__(self, bins: Mapping[str, Bins] | None = None) -> None:
        """Store the binning rules.

        Args:
            bins: Rules per column. None uses ``DEFAULT_BINS``.
        """
        self.bins = dict(DEFAULT_BINS if bins is None else bins)
        self.edges_: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> "Discretizer":
        """Compute cut points from training data.

        Rules for columns missing from ``df`` are skipped.

        Args:
            df: Training data with raw numeric values.

        Returns:
            self.

        Raises:
            ValueError: If quantile cut points are not strictly increasing.
        """
        self.edges_ = {}
        for col, spec in self.bins.items():
            if col not in df:
                continue
            if spec.quantile:
                edges = df[spec.reference or col].quantile(list(spec.edges)).to_numpy()
            else:
                edges = np.asarray(spec.edges, dtype=float)
            if np.any(np.diff(edges) <= 0):
                raise ValueError(f"Cut points for {col!r} are not increasing: {edges}")
            self.edges_[col] = edges
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace fitted numeric columns with their bin labels.

        Args:
            df: Data with raw numeric values.

        Returns:
            A copy of ``df`` with ordered categorical labels in the fitted
            columns.
        """
        out = df.copy()
        for col, edges in self.edges_.items():
            if col in out:
                cuts = [-np.inf, *edges, np.inf]
                spec = self.bins[col]
                out[col] = pd.cut(out[col], cuts, labels=list(spec.labels), right=spec.right)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return it discretized.

        Args:
            df: Training data with raw numeric values.

        Returns:
            The discretized data.
        """
        return self.fit(df).transform(df)

    def thresholds(self) -> pd.Series:
        """Return the fitted cut points per column for reporting.

        Returns:
            Series of cut point arrays indexed by column name.
        """
        return pd.Series(self.edges_, dtype=object)


def as_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to categoricals, which pgmpy requires.

    Args:
        df: Input data.

    Returns:
        A copy of ``df`` with string columns as unordered categoricals.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) and not isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("category")
    return out


def fit_cpts(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    prior: str | None = None,
    equivalent_sample_size: float = 10,
) -> DiscreteBayesianNetwork:
    """Estimate every CPT in ``model`` from ``data``.

    Args:
        model: Network whose nodes are columns of ``data``.
        data: Discretized data.
        prior: None for maximum likelihood (observed frequencies), or a
            pgmpy prior type such as ``"BDeu"`` or ``"K2"`` for smoothing.
        equivalent_sample_size: Prior strength for ``"BDeu"``.

    Returns:
        The same model with CPTs attached.
    """
    df = as_categories(data[list(model.nodes())])
    if prior is None:
        model.fit(df, estimator=MaximumLikelihoodEstimator)
    else:
        model.fit(
            df,
            estimator=BayesianEstimator,
            prior_type=prior,
            equivalent_sample_size=equivalent_sample_size,
        )
    return model


def build_network(
    data: pd.DataFrame,
    edges: Iterable[Edge],
    columns: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    prior: str | None = None,
    equivalent_sample_size: float = 10,
) -> DiscreteBayesianNetwork:
    """Build a network from a hand-specified DAG and fit its CPTs.

    Edges that touch a node outside the selected columns are dropped, so you
    can remove a variable by listing it in ``exclude``.

    Args:
        data: Discretized data.
        edges: ``(parent, child)`` pairs.
        columns: Nodes to include. None uses every node named in ``edges``.
            Columns not in any kept edge become parentless nodes.
        exclude: Nodes to remove.
        prior: See ``fit_cpts``.
        equivalent_sample_size: See ``fit_cpts``.

    Returns:
        A fitted ``DiscreteBayesianNetwork``.
    """
    edges = list(edges)
    if columns is None:
        columns = dict.fromkeys(node for edge in edges for node in edge)
    nodes = list(select_columns(data, columns, exclude).columns)
    model = DiscreteBayesianNetwork([(u, v) for u, v in edges if u in nodes and v in nodes])
    model.add_nodes_from(nodes)
    return fit_cpts(model, data, prior, equivalent_sample_size)


def get_cpd(model: DiscreteBayesianNetwork, node: str) -> TabularCPD:
    """Return the fitted CPD of ``node``.

    Args:
        model: Fitted network.
        node: Node name.

    Returns:
        The node's ``TabularCPD``.

    Raises:
        ValueError: If ``node`` has no fitted CPD.
    """
    cpd = model.get_cpds(node)
    if not isinstance(cpd, TabularCPD):
        raise ValueError(f"No fitted CPD for node {node!r}")
    return cpd


def query_target(
    infer: VariableElimination,
    target: str,
    evidence: dict | None = None,
) -> np.ndarray:
    """Return P(target | evidence) as an array ordered like the target's states.

    Args:
        infer: Inference engine for a fitted network.
        target: Node to query.
        evidence: Mapping of node name to observed state.

    Returns:
        Probabilities of each state of ``target``.
    """
    factor = infer.query([target], evidence=evidence, show_progress=False)
    if not isinstance(factor, DiscreteFactor):
        raise TypeError(f"Expected a DiscreteFactor, got {type(factor).__name__}")
    return np.asarray(factor.values)


def cpt_table(model: DiscreteBayesianNetwork, node: str) -> pd.DataFrame:
    """Return the CPT of ``node`` as a DataFrame.

    Args:
        model: Fitted network.
        node: Node name.

    Returns:
        Rows are states of ``node``. Columns are combinations of parent states,
        or a single ``P`` column for a parentless node. Each column sums to 1.
    """
    cpd = get_cpd(model, node)
    parents = cpd.variables[1:]
    values = np.asarray(cpd.get_values())
    index = pd.Index(cpd.state_names[node], name=node)
    if not parents:
        return pd.DataFrame(values, index=index, columns=["P"])
    columns = pd.MultiIndex.from_product([cpd.state_names[p] for p in parents], names=parents)
    return pd.DataFrame(values, index=index, columns=columns)


def estimate_cpt(
    data: pd.DataFrame,
    node: str,
    parents: Sequence[str] = (),
    prior: str | None = None,
    equivalent_sample_size: float = 10,
) -> pd.DataFrame:
    """Estimate one CPT without building a full network.

    Args:
        data: Discretized data.
        node: Child column.
        parents: Parent columns.
        prior: See ``fit_cpts``.
        equivalent_sample_size: See ``fit_cpts``.

    Returns:
        The CPT in the format of ``cpt_table``.
    """
    edges = [(p, node) for p in parents]
    model = build_network(data, edges, [*parents, node], prior=prior, equivalent_sample_size=equivalent_sample_size)
    return cpt_table(model, node)


def predict_proba(model: DiscreteBayesianNetwork, data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute P(target | evidence) for each row of ``data``.

    Evidence is every model node in ``data`` other than ``target``. Each
    distinct evidence combination is queried once.

    Args:
        model: Fitted network.
        data: Discretized data. Extra columns are ignored.
        target: Node to predict.

    Returns:
        One column per state of ``target``, aligned to ``data.index``.
    """
    evidence_cols = [c for c in model.nodes() if c in data and c != target]
    states = get_cpd(model, target).state_names[target]
    infer = VariableElimination(model)
    if not evidence_cols:
        marginal = query_target(infer, target)
        return pd.DataFrame([marginal] * len(data), columns=states, index=data.index)
    combos = data[evidence_cols].drop_duplicates().reset_index(drop=True)
    probs = [query_target(infer, target, row) for row in combos.to_dict("records")]
    table = pd.concat([combos, pd.DataFrame(probs, columns=states)], axis=1)
    merged = data[evidence_cols].merge(table, on=evidence_cols, how="left")
    return merged[states].set_axis(data.index)


Point = tuple[float, float]


def assign_rows(model: DiscreteBayesianNetwork, same_row: Iterable[Iterable[str]] = ()) -> dict[str, int]:
    """Give each node a row below all its parents, sharing rows within groups.

    Args:
        model: Network to lay out.
        same_row: Groups of nodes to put on one row.

    Returns:
        Row number per node, with roots in row 0.

    Raises:
        ValueError: If a group cannot share a row, such as when a path
            between two members runs through a node outside the group.
    """
    group_of = {node: i for i, group in enumerate(same_row) for node in group}
    members: dict[int, list[str]] = {}
    for node, i in group_of.items():
        members.setdefault(i, []).append(node)
    order = list(nx.topological_sort(model))
    depth = dict.fromkeys(order, 0)
    for _ in range(len(order) + 1):
        changed = False
        for node in order:
            rows = [depth[p] + (group_of.get(p, -1) != group_of.get(node, -2)) for p in model.predecessors(node)]
            if node in group_of:
                rows += [depth[m] for m in members[group_of[node]]]
            row = max(rows, default=0)
            if row > depth[node]:
                depth[node], changed = row, True
        if not changed:
            return depth
    raise ValueError("The same_row groups conflict with the edge directions.")


def layered_layout(
    model: DiscreteBayesianNetwork,
    widths: Mapping[str, float] | None = None,
    gap: float = 0.3,
    same_row: Iterable[Iterable[str]] = (),
    sweeps: int = 8,
    tries: int = 40,
) -> tuple[dict[str, Point], dict[Edge, list[Point]]]:
    """Place nodes in rows, with waypoints for edges that skip rows.

    Args:
        model: Network to lay out.
        widths: Label width per node. Default 1.
        gap: Minimum space between neighbors in a row.
        same_row: Groups of nodes to keep side by side on one row.
        sweeps: Ordering passes per try.
        tries: Starting orders to try. The layout keeps the fewest crossings.

    Returns:
        ``(nodes, waypoints)``: ``(x, row)`` per node, and the ``(x, row)``
        points between each edge's ends.
    """
    groups = [list(group) for group in same_row]
    group_of = {node: i for i, group in enumerate(groups) for node in group}
    depth = assign_rows(model, groups)
    size = {node: 1.0 if widths is None else widths.get(node, 1.0) for node in model.nodes()}
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    chains: dict[Edge, list[str]] = {}
    ends: dict[str, tuple[str, str]] = {}
    for parent, child in model.edges():
        crossed = range(depth[parent] + 1, depth[child])
        hidden = [f"\0{parent}->{child}@{r}" for r in crossed]
        for r, name in zip(crossed, hidden):
            depth[name], size[name], ends[name] = r, 0.0, (parent, child)
        nx.add_path(graph, [parent, *hidden, child])
        chains[(parent, child)] = hidden

    n_rows = max(depth.values()) + 1
    layers = [(r, r + 1) for r in range(n_rows - 1)]

    def crossings(rows: list[list[str]]) -> int:
        pos = {name: i for row in rows for i, name in enumerate(row)}
        total = 0
        for upper, _ in layers:
            segs = sorted((pos[u], pos[v]) for u in rows[upper] for v in graph.successors(u) if depth[v] == upper + 1)
            total += sum(c > a and d < b for i, (a, b) in enumerate(segs) for c, d in segs[i + 1 :])
        return total

    def reorder(row: list[str], pos: dict[str, int], neighbors) -> list[str]:
        def mean_pos(name: str) -> float:
            xs = [pos[n] for n in neighbors(name) if depth[n] != depth[name]]
            return sum(xs) / len(xs) if xs else pos[name]

        own = {name: mean_pos(name) for name in row}
        block = dict(own)
        for group in groups:
            inside = [n for n in group if n in own]
            if inside:
                block.update(dict.fromkeys(inside, sum(own[n] for n in inside) / len(inside)))
        return sorted(row, key=lambda n: (block[n], min(groups[group_of[n]]) if n in group_of else n, own[n]))

    rng = np.random.default_rng(0)
    best, best_count = None, None
    for trial in range(tries):
        rows = [sorted(n for n in graph.nodes() if depth[n] == r) for r in range(n_rows)]
        if trial:
            rows = [list(rng.permutation(row)) for row in rows]
        for sweep in range(sweeps):
            downward = sweep % 2 == 0
            for r in range(1, n_rows) if downward else range(n_rows - 2, -1, -1):
                pos = {name: i for row in rows for i, name in enumerate(row)}
                rows[r] = reorder(rows[r], pos, graph.predecessors if downward else graph.successors)
        count = crossings(rows)
        if best_count is None or count < best_count:
            best, best_count = rows, count
    rows = best

    def pack(row: list[str], want: list[float]) -> dict[str, float]:
        sep = [0.12 if a in ends and b in ends else (size[a] + size[b]) / 2 + gap for a, b in zip(row, row[1:])]
        right, left = list(want), list(want)
        for i in range(1, len(row)):
            right[i] = max(right[i], right[i - 1] + sep[i - 1])
        for i in range(len(row) - 2, -1, -1):
            left[i] = min(left[i], left[i + 1] - sep[i])
        return {name: (a + b) / 2 for name, a, b in zip(row, right, left)}

    x: dict[str, float] = {}
    for row in rows:
        x.update(pack(row, [0.0] * len(row)))
    for _ in range(10):
        for row in rows:
            want = []
            for name in row:
                if name in ends:
                    parent, child = ends[name]
                    t = (depth[name] - depth[parent]) / (depth[child] - depth[parent])
                    want.append(x[parent] + t * (x[child] - x[parent]))
                else:
                    xs = [x[n] for n in nx.all_neighbors(graph, name)]
                    want.append(sum(xs) / len(xs) if xs else x[name])
            x.update(pack(row, want))

    nodes = {node: (x[node], depth[node]) for node in model.nodes()}
    waypoints = {edge: [(x[h], depth[h]) for h in hidden] for edge, hidden in chains.items()}
    return nodes, waypoints


def draw_network(
    model: DiscreteBayesianNetwork,
    path: Path | str,
    title: str | None = None,
    highlight: Iterable[str] = (),
    same_row: Iterable[Iterable[str]] = (),
    fontsize: float = 10,
    dpi: int = 200,
) -> Path:
    """Save a top-down diagram of the network's DAG with straight arrows.

    Args:
        model: Network to draw. CPTs are optional.
        path: Output file. The extension sets the format (``.png``, ``.svg``).
        title: Optional heading.
        highlight: Nodes to draw in orange.
        same_row: Groups of nodes to draw side by side on one row.
        fontsize: Label size in points.
        dpi: Resolution for PNG output.

    Returns:
        The path written.

    Raises:
        ValueError: If the model has no nodes.
    """
    if not model.nodes():
        raise ValueError("Cannot draw a network with no nodes.")
    pad, gap, clearance, y_step, margin = 0.4, 0.35, 0.03, 1.0, 0.4
    widths = {
        node: (TextPath((0, 0), node, size=fontsize).get_extents().width + 2 * pad * fontsize) / 72
        for node in model.nodes()
    }
    layout, _ = layered_layout(model, widths, gap=gap, same_row=same_row)
    pos = {node: (x, -r * y_step) for node, (x, r) in layout.items()}
    n_rows = max(r for _, r in layout.values()) + 1

    fig = Figure(figsize=(1, 1))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    def fit_canvas() -> None:
        left = min(x - widths[n] / 2 for n, (x, _) in pos.items()) - margin
        right = max(x + widths[n] / 2 for n, (x, _) in pos.items()) + margin
        fig.set_size_inches(right - left, (n_rows - 1) * y_step + 2 * margin)
        ax.set_xlim(left, right)
        ax.set_ylim(-(n_rows - 1) * y_step - margin, margin)

    fit_canvas()
    marked = set(highlight)
    boxes = {}
    for node, xy in pos.items():
        fill, edge = ("#fde2c8", "#c2410c") if node in marked else ("#e8eef7", "#334e7a")
        boxes[node] = ax.text(
            *xy, node, ha="center", va="center", fontsize=fontsize, zorder=3,
            bbox={"boxstyle": f"round,pad={pad}", "facecolor": fill, "edgecolor": edge, "linewidth": 1.2},
        )

    fig.draw_without_rendering()
    to_data = ax.transData.inverted()
    half = {}
    for node, text in boxes.items():
        (x0, y0), (x1, y1) = to_data.transform(text.get_bbox_patch().get_window_extent().get_points())
        half[node] = ((x1 - x0) / 2, (y1 - y0) / 2)
    edges = list(model.edges())

    def attach() -> dict[tuple[Edge, int], Point]:
        ports: dict[tuple[Edge, int], Point] = {}
        outgoing: dict[str, list[tuple[float, Edge]]] = {node: [] for node in pos}
        incoming: dict[str, list[tuple[float, Edge]]] = {node: [] for node in pos}
        for parent, child in edges:
            (px, py), (cx, cy) = pos[parent], pos[child]
            if py == cy:
                side = 1 if cx > px else -1
                ports[((parent, child), -1)] = px + side * (half[parent][0] + clearance), py
                ports[((parent, child), 1)] = cx - side * (half[child][0] + clearance), cy
            else:
                outgoing[parent].append((cx, (parent, child)))
                incoming[child].append((px, (parent, child)))
        for side, attached in ((-1, outgoing), (1, incoming)):
            for node, ends in attached.items():
                x0, y0 = pos[node]
                for i, (_, edge) in enumerate(sorted(ends)):
                    offset = half[node][0] * 0.7 * (2 * (i + 0.5) / len(ends) - 1)
                    ports[(edge, side)] = x0 + offset, y0 + side * (half[node][1] + clearance)
        return ports

    for _ in range(10):
        ports, moved = attach(), False
        for node, (x0, y0) in pos.items():
            hw, hh = half[node]
            low, high = y0 - hh - 0.08, y0 + hh + 0.08
            blocked = []
            for edge in edges:
                (ax_, ay), (bx, by) = ports[(edge, -1)], ports[(edge, 1)]
                if node in edge or ay == by or max(ay, by) < low or min(ay, by) > high:
                    continue
                xs = [ax_ + (y - ay) * (bx - ax_) / (by - ay) for y in (max(low, min(ay, by)), min(high, max(ay, by)))]
                blocked.append((min(xs) - hw - 0.08, max(xs) + hw + 0.08))
            if not any(a < x0 < b for a, b in blocked):
                continue
            for other, (xo, yo) in pos.items():
                if other != node and yo == y0:
                    blocked.append((xo - half[other][0] - hw - gap, xo + half[other][0] + hw + gap))
            spots = sorted({x0, *(v for span in blocked for v in span)}, key=lambda v: abs(v - x0))
            pos[node] = next(v for v in spots if not any(a < v < b for a, b in blocked)), y0
            moved = True
        if not moved:
            break

    fit_canvas()
    for node, text in boxes.items():
        text.set_position(pos[node])
    ports = attach()
    for edge in edges:
        ax.add_patch(FancyArrowPatch(
            ports[(edge, -1)], ports[(edge, 1)], arrowstyle="-|>", mutation_scale=14,
            shrinkA=0, shrinkB=0, color="#444444", linewidth=1.0, zorder=2,
        ))
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    return out
