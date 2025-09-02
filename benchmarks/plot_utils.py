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
        function considers ``run_time_mean`` and ``total_time_mean``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same grouping columns and the minimal metric values.
        The returned frame contains an additional ``framework`` column set to
        ``"baseline_best"``.
    """

    if df.empty:
        raise ValueError("results DataFrame is empty")

    baselines = df[df["framework"] != "quasar"]
    if baselines.empty:
        raise ValueError("no baseline entries in results")

    group_cols = [c for c in ("circuit", "qubits") if c in df.columns]
    mins = (
        baselines.groupby(group_cols)[list(metrics)]
        .min()
        .reset_index()
    )
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
        When ``True`` annotate QuASAr points with the backend chosen by the
        scheduler. Requires a ``backend`` column in ``df``.
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

    x_col = "qubits" if "qubits" in df.columns else "circuit"
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

    if annotate_backend and "backend" in quasar.columns:
        for _, row in quasar.iterrows():
            backend = row.get("backend")
            if backend:
                ax.annotate(str(backend), (row[x_col], row[metric]))

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend()
    return ax


__all__ = ["compute_baseline_best", "plot_quasar_vs_baseline_best"]
