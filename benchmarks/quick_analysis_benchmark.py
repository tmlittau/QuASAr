"""Benchmark quick-path thresholds for QuASAr.

This script compares scheduler runtime with the quick-analysis path
enabled and disabled for randomly generated circuits.  Results are
written to ``benchmarks/quick_analysis_results.csv`` and the timings plot
is displayed for inspection.  Pass ``--verbose`` to see per-configuration
progress messages while the benchmark executes.

Running this script can guide selection of suitable default thresholds
for the quick path based on observed speedups.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit.circuit.random import random_circuit
from qiskit import transpile

sys.path.append(str(Path(__file__).resolve().parents[1]))

from quasar.circuit import Circuit
from quasar.planner import Planner


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbosity: int) -> None:
    """Initialise logging for CLI usage."""

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _build_circuit(num_qubits: int, depth: int) -> Circuit:
    """Generate a random circuit and convert it to ``Circuit``."""

    qc = random_circuit(num_qubits, depth, seed=0)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def _time_planning(circuit: Circuit, *, quick: bool) -> float:
    """Measure planning time for ``circuit``.

    When ``quick`` is ``True`` very high thresholds force the quick
    analysis path.  Setting all thresholds to zero disables it and triggers
    the full dynamic-programming planner.
    """

    if quick:
        planner = Planner(
            quick_max_qubits=10_000,
            quick_max_gates=1_000_000,
            quick_max_depth=10_000,
        )
    else:
        planner = Planner(quick_max_qubits=0, quick_max_gates=0, quick_max_depth=0)

    start = time.perf_counter()
    planner.plan(circuit)
    end = time.perf_counter()
    return end - start


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the quick-analysis planner thresholds",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for debug output).",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    results = []

    qubit_sizes = [4, 8, 12]
    depths = [4, 8, 12]

    for q in qubit_sizes:
        for d in depths:
            LOGGER.info("Benchmarking configuration qubits=%s depth=%s", q, d)
            circ = _build_circuit(q, d)
            quick_time = _time_planning(circ, quick=True)
            slow_time = _time_planning(circ, quick=False)
            results.append(
                {
                    "qubits": q,
                    "depth": circ.depth,
                    "gates": circ.num_gates,
                    "quick_time": quick_time,
                    "full_time": slow_time,
                }
            )

    df = pd.DataFrame(results)
    LOGGER.info("Completed benchmarking %d configuration(s)", len(df))
    df["speedup"] = df["full_time"] / df["quick_time"]

    out_csv = Path(__file__).with_name("quick_analysis_results.csv")
    LOGGER.info("Writing quick-analysis results to %s", out_csv)
    df.to_csv(out_csv, index=False)
    sns.set_theme()
    melted = df.melt(
        id_vars=["qubits", "depth"],
        value_vars=["quick_time", "full_time"],
        var_name="mode",
        value_name="time",
    )
    plt.figure()
    sns.lineplot(
        data=melted,
        x="depth",
        y="time",
        hue="mode",
        style="qubits",
        markers=True,
    )
    plt.yscale("log")
    plt.title("Scheduler runtime with/without quick analysis")
    plt.show()

    # Provide a simple recommendation for thresholds based on observed speedup
    faster = df[df["speedup"] > 1.0]
    if not faster.empty:
        LOGGER.info("Suggested quick-path thresholds based on measured speedups:")
        LOGGER.info(
            "%s",
            {
                "qubits": int(faster["qubits"].max()),
                "gates": int(faster["gates"].max()),
                "depth": int(faster["depth"].max()),
            },
        )


if __name__ == "__main__":
    main()

