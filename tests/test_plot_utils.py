import matplotlib.pyplot as plt
import pandas as pd
import pytest

from benchmarks.plot_utils import (
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
    assert result.empty
    assert isinstance(result.index, pd.RangeIndex)
    assert "run_time_mean" in result.columns
    assert "framework" in result.columns
    assert "backend" in result.columns
