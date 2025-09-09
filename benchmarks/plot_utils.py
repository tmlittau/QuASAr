from __future__ import annotations

"""Utilities for visualising benchmark results."""

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


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

    baselines = df[df["framework"] != "quasar"]
    if "unsupported" in baselines.columns:
        baselines = baselines[~baselines["unsupported"].fillna(False)]
    if baselines.empty:
        raise ValueError("no baseline entries in results")

    group_cols = [c for c in ("circuit", "qubits") if c in df.columns]
    extra_cols = [c for c in ("repetitions",) if c in baselines.columns]
    rows: list[dict[str, object]] = []
    for keys, group in baselines.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in extra_cols:
            row[col] = group[col].iloc[0]
        for metric in metrics:
            idx = group[metric].idxmin()
            row[metric] = group.loc[idx, metric]
            std_col = metric.replace("_mean", "_std")
            if std_col in group.columns:
                row[std_col] = group.loc[idx, std_col]
            if metric == metrics[0]:
                row["backend"] = group.loc[idx, "framework"]
        rows.append(row)

    mins = pd.DataFrame(rows)
    mins["framework"] = "baseline_best"
    return mins


def plot_quasar_vs_baseline_best(
    df: pd.DataFrame,
    *,
    metric: str = "run_time_mean",
    annotate_backend: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
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

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """

    if ax is None:
        ax = plt.gca()

    baseline_best = compute_baseline_best(df, metrics=[metric])
    quasar = df[df["framework"] == "quasar"]
    if "unsupported" in df.columns:
        unsupported = df[df["unsupported"].fillna(False)]
    else:
        unsupported = df.iloc[0:0]

    x_col = "qubits" if "qubits" in df.columns else "circuit"
    std_col = metric.replace("_mean", "_std")

    if std_col in baseline_best.columns:
        ax.errorbar(
            baseline_best[x_col],
            baseline_best[metric],
            yerr=baseline_best[std_col],
            fmt="o",
            label="baseline_best",
        )
    else:
        ax.plot(
            baseline_best[x_col],
            baseline_best[metric],
            marker="o",
            label="baseline_best",
        )

    ax.plot(
        quasar[x_col],
        quasar[metric],
        marker="o",
        label="QuASAr",
    )
    if std_col in quasar.columns:
        sub = quasar.sort_values(x_col)
        ax.fill_between(
            sub[x_col],
            sub[metric] - sub[std_col],
            sub[metric] + sub[std_col],
            alpha=0.2,
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
            for _, row in source.iterrows():
                backend = row.get("backend")
                if backend:
                    ax.annotate(str(backend), (row[x_col], row[metric]))

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(bottom=0)
    ax.legend()
    return ax


__all__ = ["compute_baseline_best", "plot_quasar_vs_baseline_best"]
