"""Regression tests for the partitioning benchmark summary helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "benchmarks"))
from run_benchmark import summarise_partitioning  # type: ignore


def test_summarise_partitioning_merges_metadata() -> None:
    """Baseline-best rows inherit scenario metadata from the QuASAr record."""

    df = pd.DataFrame(
        [
            {
                "scenario": "tableau_boundary",
                "variant": "tableau_boundary_1",
                "framework": "baseline_best",
                "backend": "dd",
                "run_time_mean": 0.002,
                "run_peak_memory_mean": 1024,
            },
            {
                "scenario": "tableau_boundary",
                "variant": "tableau_boundary_1",
                "framework": "quasar",
                "backend": "mps",
                "run_time_mean": 0.005,
                "run_peak_memory_mean": 2048,
                "conversion_count": 3,
                "conversion_boundary_mean": 2.0,
                "conversion_rank_mean": 4.0,
                "conversion_frontier_mean": 1.0,
                "conversion_primitive_summary": "b2b:3",
                "boundary": 2,
                "schmidt_layers": 1,
                "dense_qubits": 4,
                "clifford_qubits": 3,
                "total_qubits": 7,
            },
        ]
    )

    summary = summarise_partitioning(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["boundary"] == 2
    assert row["schmidt_layers"] == 1
    assert row["baseline_backend"] == "dd"
    assert row["baseline_runtime_mean"] == 0.002
    assert row["quasar_backend"] == "mps"
    assert row["quasar_runtime_mean"] == 0.005
    assert row["quasar_conversions"] == 3
    assert row["quasar_boundary_mean"] == 2.0
    assert row["runtime_speedup"] == 0.4
