from __future__ import annotations

"""Utilities for visualising benchmark results."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Literal, overload

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pandas as pd

try:  # seaborn provides the high level styling requested by the paper.
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional for automated tests
    sns = None

try:  # ``Backend`` is optional when plotting cached CSV results.
    from quasar.cost import Backend as _BackendEnum
except Exception:  # pragma: no cover - avoid import errors when running standalone
    _BackendEnum = None


@dataclass(frozen=True)
class BackendStyle:
    """Palette entry describing how a backend should appear in a plot."""

    color: str
    marker: str
    label: str
    tag: str | None = None
    short_label: str | None = None


_DEFAULT_BACKEND_STYLES: Mapping[str, BackendStyle] = {
    "quasar": BackendStyle(
        "#1b9e77", "o", "QuASAr", tag="QA", short_label="QuASAr"
    ),
    "baseline_best": BackendStyle(
        "#264653", "s", "Baseline best", tag="BB", short_label="Baseline"
    ),
    "sv": BackendStyle(
        "#1f77b4", "o", "Statevector", tag="SV", short_label="SV"
    ),
    "tab": BackendStyle(
        "#ff7f0e", "^", "Tableau", tag="Tab", short_label="Tab"
    ),
    "mps": BackendStyle("#2ca02c", "D", "MPS", tag="MPS", short_label="MPS"),
    "dd": BackendStyle("#d62728", "s", "DD", tag="DD", short_label="DD"),
    "stim": BackendStyle("#9467bd", "P", "Stim", tag="ST", short_label="Stim"),
    "mqt_dd": BackendStyle(
        "#8c564b", "s", "MQT-DD", tag="MQ", short_label="MQT-DD"
    ),
}


_BACKEND_ALIASES: Mapping[str, str] = {
    "baseline": "baseline_best",
    "baseline_best": "baseline_best",
    "auto": "quasar",
    "quasar": "quasar",
    "statevector": "sv",
    "sv": "sv",
    "STATEVECTOR": "sv",
    "Backend.STATEVECTOR": "sv",
    "tableau": "tab",
    "tab": "tab",
    "TABLEAU": "tab",
    "mps": "mps",
    "MPS": "mps",
    "mqc_dd": "mqt_dd",
    "mqt_dd": "mqt_dd",
    "MQT_DD": "mqt_dd",
    "decision_diagram": "dd",
    "dd": "dd",
    "DECISION_DIAGRAM": "dd",
    "stim": "stim",
    "STIM": "stim",
}


def _normalise_backend(value: object) -> str | None:
    """Return canonical backend key for ``value`` or ``None``.

    The helper accepts backend enum values, lowercase/uppercase strings and
    legacy labels that appear in cached CSV/JSON results.  Unknown entries
    return ``None`` so that callers can fall back to default matplotlib
    styling without raising errors.
    """

    if value is None:
        return None
    if _BackendEnum is not None and isinstance(value, _BackendEnum):
        value = value.value
    if isinstance(value, str):
        return _BACKEND_ALIASES.get(value, _BACKEND_ALIASES.get(value.lower()))
    return None


def _as_seaborn_palette(
    palette: Mapping[object, str] | Sequence[str] | str | None,
) -> Sequence[str] | str | None:
    """Convert mapping palettes to the sequence format expected by Seaborn."""

    if palette is None:
        return None
    if isinstance(palette, Mapping):
        return list(palette.values())
    if isinstance(palette, str):
        return palette
    return list(palette)


def setup_benchmark_style(
    context: str = "talk",
    font_scale: float = 0.9,
    *,
    palette: Mapping[object, str] | Sequence[str] | str | None = None,
    legend_fontsize: float | str = "medium",
    legend_title_fontsize: float | str = "medium",
) -> None:
    """Apply the shared matplotlib/Seaborn styling used in the paper plots.

    Parameters
    ----------
    context:
        Seaborn context to apply.  ``"talk"`` offers a good balance for paper
        figures.  When seaborn is unavailable the function gracefully
        downgrades to basic matplotlib styling.
    font_scale:
        Scale factor applied to tick and label fonts.
    """

    rc = {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": legend_fontsize,
        "legend.title_fontsize": legend_title_fontsize,
    }
    resolved_palette = _as_seaborn_palette(palette)
    if sns is not None:  # pragma: no branch - optional dependency
        sns.set_theme(
            context=context,
            style="whitegrid",
            font_scale=font_scale,
            palette=resolved_palette,
            rc=rc,
        )
    else:
        for key, value in rc.items():
            plt.rcParams.setdefault(key, value)
    plt.rcParams.setdefault("axes.spines.top", False)
    plt.rcParams.setdefault("axes.spines.right", False)
    plt.rcParams.setdefault("figure.dpi", 110)
    plt.rcParams.setdefault("figure.autolayout", True)
    plt.rcParams.setdefault("legend.frameon", False)


def _apply_legend_style(
    legend: Legend | None,
    *,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float] | None = (0.5, 1.02),
    ncol: int = 1,
    frameon: bool | None = None,
    title: str | None = None,
) -> None:
    """Apply consistent styling to matplotlib legends."""

    if legend is None:
        return
    legend.set_loc(loc)
    if hasattr(legend, "set_ncols"):
        legend.set_ncols(max(1, ncol))
    else:  # pragma: no cover - compatibility with older matplotlib
        legend._ncols = max(1, ncol)  # type: ignore[attr-defined]
    if bbox_to_anchor is not None:
        legend.set_bbox_to_anchor(bbox_to_anchor)
    if frameon is not None:
        legend.set_frame_on(frameon)
    else:
        legend.set_frame_on(bool(plt.rcParams.get("legend.frameon", False)))
    if title is not None:
        legend.set_title(title)
    fontsize = plt.rcParams.get("legend.fontsize", "medium")
    for text in legend.get_texts():
        text.set_fontsize(fontsize)
    legend_title = legend.get_title()
    if legend_title is not None and legend_title.get_text():
        legend_title.set_fontsize(plt.rcParams.get("legend.title_fontsize", "medium"))


def _style_for_backend(value: object) -> BackendStyle | None:
    key = _normalise_backend(value)
    if key is None:
        return None
    return _DEFAULT_BACKEND_STYLES.get(key)


def backend_palette(order: Iterable[object] | None = None) -> Mapping[object, str]:
    """Return consistent colours for a sequence of backend identifiers."""

    if order is None:
        order = _DEFAULT_BACKEND_STYLES.keys()
    palette: dict[object, str] = {}
    for backend in order:
        style = _style_for_backend(backend)
        if style is not None:
            palette[backend] = style.color
    return palette


def backend_markers(order: Iterable[object] | None = None) -> Mapping[object, str]:
    """Return marker symbols for backend identifiers."""

    if order is None:
        order = _DEFAULT_BACKEND_STYLES.keys()
    markers: dict[object, str] = {}
    for backend in order:
        style = _style_for_backend(backend)
        if style is not None:
            markers[backend] = style.marker
    return markers


def backend_labels(
    order: Iterable[object] | None = None,
    *,
    abbreviated: bool = False,
) -> Mapping[object, str]:
    """Return publication friendly labels for backends.

    Parameters
    ----------
    order:
        Optional iterable specifying the backends to include.  When omitted
        the helper returns all known backends in their default ordering.
    abbreviated:
        When set, prefer compact identifiers such as ``"Tab"`` for Tableau in
        place of the full backend names.  ``backend_labels`` falls back to the
        full label if a backend does not define a short variant.
    """

    if order is None:
        order = _DEFAULT_BACKEND_STYLES.keys()
    labels: dict[object, str] = {}
    for backend in order:
        style = _style_for_backend(backend)
        if style is None:
            continue
        label = style.short_label if abbreviated and style.short_label else style.label
        if label:
            labels[backend] = label
    return labels


def backend_tags(order: Iterable[object] | None = None) -> Mapping[object, str]:
    """Return abbreviated identifiers suitable for point annotations."""

    if order is None:
        order = _DEFAULT_BACKEND_STYLES.keys()
    tags: dict[object, str] = {}
    for backend in order:
        style = _style_for_backend(backend)
        if style is None:
            continue
        tag = style.tag if style.tag else style.label
        if tag:
            tags[backend] = tag
    return tags


def _annotate_backends(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    backend_col: str,
    offset: tuple[float, float] = (0.0, 6.0),
    min_pixel_distance: float = 18.0,
) -> None:
    """Annotate each backend once per circuit (or ``x_col``) using compact labels."""

    required = {x_col, y_col, backend_col}
    if not required.issubset(data.columns):
        return

    group_columns: list[str] = [backend_col]
    optional_group = "circuit"
    if optional_group in data.columns:
        group_columns.append(optional_group)
    elif x_col not in group_columns:
        group_columns.append(x_col)

    subset_cols = list(dict.fromkeys([x_col, y_col, *group_columns]))
    annotations = data[subset_cols].dropna(subset=[x_col, y_col])
    if annotations.empty:
        return

    try:
        grouped = annotations.groupby(group_columns, dropna=False, sort=False)
    except TypeError:  # pragma: no cover - older pandas without ``dropna`` keyword
        grouped = annotations.groupby(group_columns, sort=False)

    candidates: list[pd.Series] = []
    for _, group in grouped:
        if group.empty:
            continue
        idx = group[y_col].idxmin()
        candidates.append(group.loc[idx])

    if not candidates:
        return

    if min_pixel_distance <= 0:
        min_pixel_distance = 0.0
    labels = backend_tags(annotations[backend_col])
    used_positions: list[tuple[float, float]] = []
    for row in candidates:
        backend = row.get(backend_col)
        label = labels.get(backend)
        if not label:
            continue
        x_value = row.get(x_col)
        y_value = row.get(y_col)
        if pd.isna(x_value) or pd.isna(y_value):
            continue
        position = ax.transData.transform((x_value, y_value))
        if used_positions and min_pixel_distance > 0:
            if any(
                np.hypot(position[0] - px, position[1] - py) < min_pixel_distance
                for px, py in used_positions
            ):
                continue
        used_positions.append(tuple(position))
        ax.annotate(
            label,
            (x_value, y_value),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize="small",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )


def _metric_label(name: str) -> str:
    pretty = name.replace("_", " ")
    return pretty.replace(" mean", "")


def summarise_speedups(
    baseline: pd.DataFrame,
    quasar: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    key_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return table comparing baseline and QuASAr metrics.

    Parameters
    ----------
    baseline, quasar:
        DataFrames containing the results to compare.  Both frames must include
        ``metric`` and share a set of grouping columns (``circuit`` and
        ``qubits`` by default).
    metric:
        Column name holding the value to compare.
    key_columns:
        Optional override for the join columns.  When omitted the helper uses
        the intersection of ``("circuit", "qubits", "depth", "alpha")``.
    """

    candidates = ["circuit", "qubits", "depth", "alpha"]
    if key_columns is None:
        key_columns = [
            col for col in candidates if col in baseline.columns and col in quasar.columns
        ]
    if not key_columns:
        raise ValueError("unable to determine common grouping columns for speedup table")

    left_cols = list(key_columns) + [metric]
    right_cols = list(key_columns) + [metric]
    if "backend" in baseline.columns:
        left_cols.append("backend")
    if "backend" in quasar.columns:
        right_cols.append("backend")

    merged = baseline[left_cols].merge(
        quasar[right_cols],
        on=key_columns,
        how="inner",
        suffixes=("_baseline", "_quasar"),
    )
    if merged.empty:
        return merged

    merged["speedup"] = merged[f"{metric}_baseline"] / merged[f"{metric}_quasar"]
    merged.rename(columns={metric: f"{metric}_baseline"}, inplace=True)

    if "backend_baseline" in merged.columns:
        merged["baseline_backend"] = merged["backend_baseline"].apply(
            lambda value: backend_labels([value]).get(value, str(value))
        )
    if "backend_quasar" in merged.columns:
        merged["quasar_backend"] = merged["backend_quasar"].apply(
            lambda value: backend_labels([value]).get(value, str(value))
        )

    return merged


def _draw_speedup_table(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    metric: str,
) -> None:
    """Render ``table`` (from :func:`summarise_speedups`) below a plot."""

    ax.axis("off")
    if table.empty:
        ax.text(0.5, 0.5, "No overlapping measurements", ha="center", va="center")
        return

    display = table.copy()
    metric_label = _metric_label(metric)
    display = display[[
        *(col for col in ("circuit", "qubits", "depth", "alpha") if col in display.columns),
        *(col for col in ("baseline_backend", "quasar_backend") if col in display.columns),
        f"{metric}_baseline",
        f"{metric}_quasar",
        "speedup",
    ]]

    def _format(value, precision=3):
        if pd.isna(value):
            return "—"
        return f"{value:.{precision}g}"

    display[f"{metric}_baseline"] = display[f"{metric}_baseline"].map(_format)
    display[f"{metric}_quasar"] = display[f"{metric}_quasar"].map(_format)
    display["speedup"] = display["speedup"].map(lambda x: f"{x:.2f}×" if pd.notna(x) else "—")

    headers = [
        *(col.title() for col in ("circuit", "qubits", "depth", "alpha") if col in display.columns),
        *("Baseline" if col == "baseline_backend" else "QuASAr" for col in ("baseline_backend", "quasar_backend") if col in display.columns),
        f"Baseline {metric_label}",
        f"QuASAr {metric_label}",
        "Speedup",
    ]

    body = display.values.tolist()
    table_obj = ax.table(cellText=body, colLabels=headers, loc="center")
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(9)
    table_obj.scale(1.2, 1.4)


def plot_backend_timeseries(
    forced: pd.DataFrame,
    auto: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    facet: str = "circuit",
    hue: str = "backend",
    x: str = "qubits",
    log_scale: bool = True,
    annotate_auto: bool = True,
    annotate_offset: tuple[float, float] = (0.0, 6.0),
    annotation_min_distance: float = 18.0,
    col_wrap: int | None = 3,
    height: float | None = 3.2,
    aspect: float | None = 1.3,
    facet_kws: Mapping[str, object] | None = None,
) -> "sns.axisgrid.FacetGrid | None":
    """Plot forced backend measurements alongside automatic selections.

    This helper mirrors the multi-panel figures used in
    ``notebooks/comparison.ipynb``.  Forced runs are rendered as lines per
    backend while automatic runs appear as highlighted markers.

    Parameters
    ----------
    annotate_offset:
        Offset (in display points) applied to annotation text.
    annotation_min_distance:
        Minimum spacing (in display pixels) required between annotations to
        avoid overlapping labels.  Set to ``0`` to disable collision detection.
    col_wrap, height, aspect, facet_kws:
        Layout controls passed through to :func:`seaborn.relplot`.  ``col_wrap``
        defaults to 3 to prevent overly wide grids, while ``height`` and
        ``aspect`` tweak subplot sizing for readability.
    """

    if sns is None:  # pragma: no cover - seaborn optional
        raise RuntimeError("seaborn is required for plot_backend_timeseries")

    order = list(dict.fromkeys(forced[hue].tolist() + auto[hue].tolist()))
    palette = backend_palette(order)
    setup_benchmark_style(palette=palette)

    resolved_facet_kws = dict(facet_kws or {})
    resolved_facet_kws.setdefault("sharey", False)
    facet_kwargs = resolved_facet_kws or None

    if log_scale:
        forced = forced.copy()
        auto = auto.copy()
        forced[metric] = forced[metric].clip(lower=1e-9)
        auto[metric] = auto[metric].clip(lower=1e-9)
    grid = sns.relplot(
        data=forced,
        x=x,
        y=metric,
        hue=hue,
        hue_order=order,
        col=facet,
        kind="line",
        marker="o",
        palette=palette,
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        facet_kws=facet_kwargs,
    )
    grid.set_axis_labels(x.replace("_", " ").title(), _metric_label(metric))
    if log_scale:
        grid.set(yscale="log")

    for circuit_name, ax in grid.axes_dict.items():
        sub = auto[auto[facet] == circuit_name]
        if sub.empty:
            continue
        sns.scatterplot(
            data=sub,
            x=x,
            y=metric,
            hue=hue,
            hue_order=order,
            palette=palette,
            marker="X",
            s=120,
            edgecolor="black",
            linewidth=0.7,
            ax=ax,
            legend=False,
        )
        if annotate_auto and hue in sub.columns:
            _annotate_backends(
                ax,
                sub,
                x_col=x,
                y_col=metric,
                backend_col=hue,
                offset=annotate_offset,
                min_pixel_distance=annotation_min_distance,
            )

    grid.fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.35)
    title = _metric_label(metric)
    grid.fig.suptitle(f"Forced (lines) vs auto (markers): {title}")
    _apply_legend_style(
        getattr(grid, "_legend", None),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=max(1, min(len(order), 4)),
    )
    return grid


def plot_metric_trend(
    x: Sequence[float] | pd.Series,
    y: Sequence[float] | pd.Series,
    *,
    label: str | None = None,
    ax: plt.Axes | None = None,
    logx: bool = False,
    logy: bool = False,
    marker: str = "o",
    color: str | None = None,
) -> plt.Axes:
    """Plot a single metric trend with consistent styling.

    Used by ``runtime_vs_alpha.ipynb``, ``plan_cache_speedup.ipynb`` and
    ``dd_frontier_extraction_time.ipynb``.
    """

    setup_benchmark_style()
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, marker=marker, label=label, color=color)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if label:
        ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    return ax


def plot_heatmap(
    pivot: pd.DataFrame,
    *,
    annot: bool = True,
    fmt: str = "",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Render a heatmap with the shared style settings.

    Matches the visual style of ``plan_choice_heatmap.ipynb``.
    """

    setup_benchmark_style()
    if sns is None:  # pragma: no cover - seaborn optional
        raise RuntimeError("seaborn is required for plot_heatmap")
    if ax is None:
        ax = plt.gca()
    sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    return ax


def plot_speedup_bars(
    speedups: Mapping[str, float],
    *,
    ax: plt.Axes | None = None,
    sort: bool = True,
) -> plt.Axes:
    """Bar chart visualising relative speedups.

    Convenience wrapper for ``relative_speedup_bar_chart.ipynb``.
    """

    setup_benchmark_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    items = list(speedups.items())
    if sort:
        items.sort(key=lambda item: item[1], reverse=True)
    labels, values = zip(*items) if items else ([], [])
    bars = ax.bar(labels, values, color="#1b9e77")
    ax.set_ylabel("Speedup (×)")
    ax.set_xlabel("Circuit")
    ax.set_ylim(bottom=0)
    if values:
        ymax = max(values)
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(top=ymax * 1.08)
    ax.bar_label(bars, fmt="{:.2f}", padding=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.xaxis.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ax


def compute_baseline_best(
    df: pd.DataFrame, metrics: Sequence[str] = ("run_time_mean", "total_time_mean")
) -> pd.DataFrame:
    """Return per-circuit minima across all baseline backends.

    Parameters
    ----------
    df:
        DataFrame containing at least ``framework`` and the metrics in ``metrics``.
        Rows with ``framework == 'quasar'`` are ignored.
    metrics:
        Iterable of column names for which to compute the minimum. By default the
        function considers ``run_time_mean`` and ``total_time_mean``.  The backend
        responsible for the minimum of the first metric is recorded under the
        ``backend`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same grouping columns and the minimal metric values.
        The returned frame contains an additional ``framework`` column set to
        ``"baseline_best"`` and a ``backend`` column identifying the baseline
        backend that achieved the minimum for the primary metric.
    """

    if df.empty:
        raise ValueError("results DataFrame is empty")
    if "framework" not in df.columns:
        raise ValueError("results DataFrame lacks 'framework' column")

    baselines = df[df["framework"] != "quasar"]
    if "unsupported" in baselines.columns:
        unsupported = baselines["unsupported"].astype("boolean", copy=False)
        mask = unsupported.fillna(False)
        baselines = baselines[~mask.to_numpy(dtype=bool)]
    if baselines.empty:
        raise ValueError("no baseline entries in results")

    metrics = list(metrics)
    if not metrics:
        raise ValueError("no metrics provided for baseline comparison")

    missing_metrics = [metric for metric in metrics if metric not in baselines.columns]
    if missing_metrics:
        raise ValueError(
            "results DataFrame lacks required metric columns: " + ", ".join(missing_metrics)
        )

    numeric_metrics = baselines[metrics].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric_metrics)
    if isinstance(finite_mask, pd.DataFrame):
        valid_rows = finite_mask.all(axis=1)
    else:  # pragma: no cover - ``metrics`` contains a single column producing a Series
        valid_rows = finite_mask
    baselines = baselines[valid_rows.to_numpy(dtype=bool)]

    group_cols = [
        c
        for c in ("circuit", "qubits", "scenario", "variant")
        if c in df.columns
    ]
    extra_cols = [c for c in ("repetitions",) if c in baselines.columns]
    std_columns = []
    for metric in metrics:
        std_col = metric.replace("_mean", "_std")
        if std_col in baselines.columns and std_col not in std_columns:
            std_columns.append(std_col)

    if group_cols:
        try:
            groups = baselines.groupby(group_cols, dropna=False)
        except TypeError:  # pragma: no cover - older pandas without ``dropna`` argument
            groups = baselines.groupby(group_cols)
    else:
        groups = [((), baselines)]
    rows: list[dict[str, object]] = []
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in extra_cols:
            row[col] = group[col].iloc[0]
        if group.empty:
            continue
        for metric in metrics:
            idx = group[metric].idxmin()
            row[metric] = group.loc[idx, metric]
            std_col = metric.replace("_mean", "_std")
            if std_col in group.columns:
                row[std_col] = group.loc[idx, std_col]
            if metric == metrics[0]:
                row["backend"] = group.loc[idx, "framework"]
        rows.append(row)

    base_columns = list(dict.fromkeys([*group_cols, *extra_cols, *metrics, *std_columns, "backend"]))
    mins = pd.DataFrame(rows, columns=base_columns)
    mins["framework"] = "baseline_best"
    return mins


@overload
def plot_quasar_vs_baseline_best(
    df: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    annotate_backend: bool = False,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
    show_speedup_table: bool = False,
    table_ax: plt.Axes | None = None,
    return_table: Literal[False] = False,
    return_figure: Literal[False] = False,
    speedup_metric: str | None = None,
    palette: Mapping[object, str] | None = None,
    markers: Mapping[object, str] | None = None,
) -> plt.Axes:
    ...


@overload
def plot_quasar_vs_baseline_best(
    df: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    annotate_backend: bool = False,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
    show_speedup_table: bool = False,
    table_ax: plt.Axes | None = None,
    return_table: Literal[True],
    return_figure: Literal[False] = False,
    speedup_metric: str | None = None,
    palette: Mapping[object, str] | None = None,
    markers: Mapping[object, str] | None = None,
) -> tuple[plt.Axes, pd.DataFrame]:
    ...


@overload
def plot_quasar_vs_baseline_best(
    df: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    annotate_backend: bool = False,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
    show_speedup_table: bool = False,
    table_ax: plt.Axes | None = None,
    return_table: Literal[False] = False,
    return_figure: Literal[True],
    speedup_metric: str | None = None,
    palette: Mapping[object, str] | None = None,
    markers: Mapping[object, str] | None = None,
) -> tuple[plt.Axes, plt.Figure]:
    ...


@overload
def plot_quasar_vs_baseline_best(
    df: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    annotate_backend: bool = False,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
    show_speedup_table: bool = False,
    table_ax: plt.Axes | None = None,
    return_table: Literal[True],
    return_figure: Literal[True],
    speedup_metric: str | None = None,
    palette: Mapping[object, str] | None = None,
    markers: Mapping[object, str] | None = None,
) -> tuple[plt.Axes, pd.DataFrame, plt.Figure]:
    ...


def plot_quasar_vs_baseline_best(
    df: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    annotate_backend: bool = False,
    ax: plt.Axes | None = None,
    log_scale: bool = True,
    show_speedup_table: bool = False,
    table_ax: plt.Axes | None = None,
    return_table: bool = False,
    return_figure: bool = False,
    speedup_metric: str | None = None,
    palette: Mapping[object, str] | None = None,
    markers: Mapping[object, str] | None = None,
) -> plt.Axes | tuple[object, ...]:
    """Plot QuASAr against the best baseline backend for ``metric``.

    Parameters
    ----------
    df:
        Benchmark results including both baseline and QuASAr entries.
    metric:
        Column to plot on the y-axis. Defaults to ``"run_time_mean"``.
    annotate_backend:
        When ``True`` annotate points with the backend used for each
        measurement.  Requires a ``backend`` column in ``df`` and leverages the
        ``backend`` column returned by :func:`compute_baseline_best` for
        baseline points.
    ax:
        Optional matplotlib axes to draw on. A new axes is created when omitted.
    return_table:
        When ``True`` the helper also returns a dataframe summarising speedups
        between baseline best and QuASAr for the selected metric.
    return_figure:
        When ``True`` include the matplotlib figure in the return value.  This
        is helpful when the helper created the plot/table stack internally and
        the caller wishes to adjust layout or persist the figure.

    Returns
    -------
    matplotlib.axes.Axes or tuple of matplotlib objects
        Returns the plot axes by default.  When ``return_table`` and/or
        ``return_figure`` are enabled the return value expands to include the
        requested objects while preserving backward compatible ordering.
    """

    default_order = ["baseline_best", "quasar"]
    palette = palette or backend_palette(default_order)
    markers = markers or backend_markers(default_order)
    setup_benchmark_style(palette=palette)

    figure: plt.Figure | None = None
    if show_speedup_table and table_ax is None and ax is not None:
        raise ValueError("table_ax must be provided when show_speedup_table=True and ax is pre-supplied")
    if show_speedup_table and table_ax is None and ax is None:
        figure, (ax, table_ax) = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        figure.set_size_inches(10.5, 6.5)
        figure.subplots_adjust(hspace=0.35)
    if ax is None:
        ax = plt.gca()
    if figure is None:
        figure = ax.figure

    baseline_best = compute_baseline_best(df, metrics=[metric])
    quasar = df[df["framework"] == "quasar"]
    if "unsupported" in df.columns:
        unsupported_mask = df["unsupported"].astype("boolean", copy=False).fillna(False)
        unsupported = df[unsupported_mask.to_numpy(dtype=bool)]
    else:
        unsupported = df.iloc[0:0]

    x_col = "qubits" if "qubits" in df.columns else "circuit"
    std_col = metric.replace("_mean", "_std")

    base_color = palette.get("baseline_best", "#264653")
    base_marker = markers.get("baseline_best", "s")
    quasar_color = palette.get("quasar", "#1b9e77")
    quasar_marker = markers.get("quasar", "o")

    group_column = "circuit" if ("circuit" in baseline_best.columns or "circuit" in quasar.columns) else None

    def _iter_groups(data: pd.DataFrame) -> Iterable[tuple[object, pd.DataFrame]]:
        if group_column is None or group_column not in data.columns:
            yield None, data
            return
        try:
            groups = data.groupby(group_column, dropna=False, sort=False)
        except TypeError:  # pragma: no cover - compatibility with older pandas
            groups = data.groupby(group_column, sort=False)
        for key, group in groups:
            yield key, group

    def _plot_series(
        data: pd.DataFrame,
        *,
        color: str,
        marker: str,
        label: str,
        with_std: bool,
        fill: bool = False,
    ) -> None:
        label_used = False
        for _, group in _iter_groups(data):
            if group.empty:
                continue
            ordered = group.sort_values(x_col)
            plot_label = label if not label_used else None
            if with_std and std_col in ordered.columns:
                ax.errorbar(
                    ordered[x_col],
                    ordered[metric],
                    yerr=ordered[std_col],
                    color=color,
                    marker=marker,
                    linestyle="-",
                    linewidth=1.5,
                    capsize=3,
                    label=plot_label,
                )
            else:
                ax.plot(
                    ordered[x_col],
                    ordered[metric],
                    marker=marker,
                    color=color,
                    linewidth=1.5,
                    label=plot_label,
                )
            if fill and std_col in ordered.columns:
                ax.fill_between(
                    ordered[x_col],
                    ordered[metric] - ordered[std_col],
                    ordered[metric] + ordered[std_col],
                    alpha=0.15,
                    color=color,
                )
            label_used = True

    _plot_series(
        baseline_best,
        color=base_color,
        marker=base_marker,
        label="Baseline best",
        with_std=std_col in baseline_best.columns,
    )

    _plot_series(
        quasar,
        color=quasar_color,
        marker=quasar_marker,
        label="QuASAr",
        with_std=std_col in quasar.columns,
        fill=True,
    )

    if not unsupported.empty:
        ax.scatter(
            unsupported[x_col],
            [0] * len(unsupported),
            marker="x",
            label="not supported",
            color="red",
        )

    if annotate_backend:
        for source in (baseline_best, quasar):
            if "backend" not in source.columns:
                continue
            _annotate_backends(ax, source, x_col=x_col, y_col=metric, backend_col="backend")

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(_metric_label(metric))
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.margins(x=0.02)
    legend = ax.legend()
    _apply_legend_style(
        legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(palette)),
    )

    speed_metric = speedup_metric or metric
    summary: pd.DataFrame | None = None
    if show_speedup_table or return_table:
        try:
            summary = summarise_speedups(baseline_best, quasar, metric=speed_metric)
        except ValueError:
            summary = pd.DataFrame()
    if show_speedup_table and table_ax is not None and summary is not None:
        _draw_speedup_table(table_ax, summary, metric=speed_metric)

    if summary is not None:
        figure._quasar_speedup_table = summary  # type: ignore[attr-defined]

    outputs: list[object] = [ax]
    if return_table:
        outputs.append(summary if summary is not None else pd.DataFrame())
    if return_figure:
        outputs.append(figure)
    if len(outputs) == 1:
        return ax
    return tuple(outputs)


__all__ = [
    "setup_benchmark_style",
    "backend_palette",
    "backend_markers",
    "backend_labels",
    "compute_baseline_best",
    "summarise_speedups",
    "plot_quasar_vs_baseline_best",
    "plot_backend_timeseries",
    "plot_metric_trend",
    "plot_heatmap",
    "plot_speedup_bars",
]
