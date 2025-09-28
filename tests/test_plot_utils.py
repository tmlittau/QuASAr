import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from benchmarks.plot_utils import (
    _annotate_backends,
    backend_tags,
    compute_baseline_best,
    plot_quasar_vs_baseline_best,
    summarise_speedups,
)


def _sample_results() -> pd.DataFrame:
    data = [
        {
            "framework": "STATEVECTOR",
            "backend": "STATEVECTOR",
            "circuit": "qft",
            "qubits": 3,
            "run_time_mean": 1.2,
            "run_time_std": 0.1,
        },
        {
            "framework": "TABLEAU",
            "backend": "TABLEAU",
            "circuit": "qft",
            "qubits": 3,
            "run_time_mean": 0.8,
            "run_time_std": 0.05,
        },
        {
            "framework": "quasar",
            "backend": "TABLEAU",
            "circuit": "qft",
            "qubits": 3,
            "run_time_mean": 0.5,
            "run_time_std": 0.02,
        },
    ]
    return pd.DataFrame(data)


def test_summarise_speedups_uses_minimum_baseline():
    df = _sample_results()
    baseline = compute_baseline_best(df, metrics=["run_time_mean"])
    quasar = df[df["framework"] == "quasar"]
    summary = summarise_speedups(baseline, quasar, metric="run_time_mean")
    assert not summary.empty
    row = summary.iloc[0]
    assert row["baseline_backend"] == "Tableau"
    assert row["quasar_backend"] == "Tableau"
    assert row["speedup"] == pytest.approx(0.8 / 0.5)


def test_plot_quasar_vs_baseline_best_returns_speedup_table():
    df = _sample_results()
    ax, summary = plot_quasar_vs_baseline_best(
        df,
        annotate_backend=True,
        log_scale=False,
        return_table=True,
        show_speedup_table=True,
    )
    try:
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty
        assert "speedup" in summary.columns
    finally:
        plt.close(ax.figure)


def test_compute_baseline_best_handles_all_nan_metrics():
    df = pd.DataFrame(
        [
            {
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "circuit": "qft",
                "qubits": 3,
                "run_time_mean": float("nan"),
                "run_time_std": float("nan"),
            }
        ]
    )

    result = compute_baseline_best(df, metrics=["run_time_mean"])

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["framework"] == "baseline_best"
    assert np.isnan(row["run_time_mean"])
    assert row["backend"] == "unavailable"
    assert row["status"]


def test_compute_baseline_best_records_unavailable_baselines():
    df = pd.DataFrame(
        [
            {
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "circuit": "qft",
                "qubits": 5,
                "unsupported": True,
                "failed": False,
                "comment": "requires >8GB",
            },
            {
                "framework": "TABLEAU",
                "backend": "TABLEAU",
                "circuit": "qft",
                "qubits": 5,
                "unsupported": True,
                "failed": True,
                "error": "timeout",
            },
        ]
    )

    result = compute_baseline_best(df, metrics=["run_time_mean"])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["framework"] == "baseline_best"
    assert row["backend"] == "unavailable"
    assert np.isnan(row["run_time_mean"])
    assert bool(row["failed"]) is True
    assert bool(row["unsupported"]) is True
    assert "STATEVECTOR" in row["status"]
    assert "TABLEAU" in row["status"]


def test_compute_baseline_best_handles_missing_baseline_with_quasar_runs():
    df = pd.DataFrame(
        [
            {
                "framework": "quasar",
                "backend": "quasar-sim",
                "circuit": "qft",
                "qubits": 6,
                "run_time_mean": 0.42,
                "run_time_std": 0.03,
            }
        ]
    )

    result = compute_baseline_best(df, metrics=["run_time_mean"])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["framework"] == "baseline_best"
    assert row["backend"] == "unavailable"
    assert np.isnan(row["run_time_mean"])
    assert row["status"] == "no baseline measurement available"
    assert bool(row["failed"]) is False
    assert bool(row["unsupported"]) is True


def test_compute_baseline_best_fills_missing_metric_columns():
    df = pd.DataFrame(
        [
            {
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "circuit": "qft",
                "qubits": 4,
            }
        ]
    )

    result = compute_baseline_best(df, metrics=["run_time_mean"])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["framework"] == "baseline_best"
    assert np.isnan(row["run_time_mean"])
    assert row["backend"] == "unavailable"


def test_plot_quasar_vs_baseline_best_handles_missing_baseline():
    df = pd.DataFrame(
        [
            {
                "framework": "quasar",
                "backend": "quasar-sim",
                "circuit": "qft",
                "qubits": 3,
                "run_time_mean": 0.5,
                "run_time_std": 0.02,
            },
            {
                "framework": "quasar",
                "backend": "quasar-sim",
                "circuit": "qft",
                "qubits": 5,
                "run_time_mean": 0.8,
                "run_time_std": 0.05,
            },
        ]
    )

    ax, summary = plot_quasar_vs_baseline_best(
        df,
        return_table=True,
        show_speedup_table=True,
        log_scale=False,
    )
    try:
        assert isinstance(summary, pd.DataFrame)
        if not summary.empty:
            assert "run_time_mean_baseline" in summary.columns
            assert "baseline_backend" in summary.columns
            assert summary["baseline_backend"].iloc[0] == "unavailable"
    finally:
        plt.close(ax.figure)


def test_backend_annotations_collapse_per_circuit():
    df = pd.DataFrame(
        [
            {
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "circuit": "qft",
                "qubits": 3,
                "run_time_mean": 1.2,
            },
            {
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "circuit": "qft",
                "qubits": 5,
                "run_time_mean": 0.9,
            },
            {
                "framework": "STATEVECTOR",
                "backend": "STATEVECTOR",
                "circuit": "grover",
                "qubits": 4,
                "run_time_mean": 0.7,
            },
            {
                "framework": "TABLEAU",
                "backend": "TABLEAU",
                "circuit": "qft",
                "qubits": 3,
                "run_time_mean": 0.6,
            },
            {
                "framework": "TABLEAU",
                "backend": "TABLEAU",
                "circuit": "grover",
                "qubits": 4,
                "run_time_mean": 0.4,
            },
            {
                "framework": "TABLEAU",
                "backend": "TABLEAU",
                "circuit": "grover",
                "qubits": 5,
                "run_time_mean": 0.35,
            },
            {
                "framework": "quasar",
                "backend": "TABLEAU",
                "circuit": "qft",
                "qubits": 5,
                "run_time_mean": 0.5,
            },
        ]
    )

    fig, ax = plt.subplots()
    try:
        _annotate_backends(
            ax,
            df,
            x_col="qubits",
            y_col="run_time_mean",
            backend_col="backend",
        )
        tags = backend_tags(df["backend"])
        inverse_tags = {tag: backend for backend, tag in tags.items()}
        expected_pairs = {
            (backend, circuit)
            for (backend, circuit), group in df.groupby(["backend", "circuit"], sort=False)
            if not group["run_time_mean"].isna().all()
        }
        observed_pairs: set[tuple[str, str]] = set()
        for annotation in ax.texts:
            backend = inverse_tags.get(annotation.get_text())
            if backend is None:
                continue
            x_value, y_value = annotation.xy
            mask = (
                (df["backend"] == backend)
                & np.isclose(df["qubits"], x_value)
                & np.isclose(df["run_time_mean"], y_value)
            )
            if not mask.any():
                continue
            circuits = df.loc[mask, "circuit"].unique()
            assert len(circuits) == 1
            observed_pairs.add((backend, circuits[0]))

        assert observed_pairs == expected_pairs
    finally:
        plt.close(fig)
