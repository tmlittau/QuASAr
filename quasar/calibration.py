"""Microbenchmark-based calibration of QuASAr cost coefficients.

This module executes small Python microbenchmarks intended to provide
rough timing information for the different simulation backends and
conversion primitives used by :class:`quasar.cost.CostEstimator`.
The results can be written to a JSON file and later re-loaded to tune
cost predictions.

The benchmarks are intentionally lightweight and avoid external
dependencies.  They serve only as a coarse guide for relative
performance on the current machine.
"""

from __future__ import annotations

from datetime import datetime
from statistics import fmean, pvariance
from time import perf_counter
from typing import Dict, TYPE_CHECKING, TypedDict, Mapping, Callable
import argparse
import json
from pathlib import Path
import math
import os
import platform
import subprocess

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .cost import CostEstimator


CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"


class CalibrationMetadata(TypedDict, total=False):
    """Descriptive information captured alongside calibration results."""

    hardware: Dict[str, str]
    date: str
    software_commit: str


class CalibrationRecord(TypedDict):
    """Complete calibration payload including coefficients and metadata."""

    coeff: Dict[str, float]
    variance: Dict[str, float]
    metadata: CalibrationMetadata


def _current_commit() -> str:
    """Return the git commit hash of the working tree or ``"unknown"``."""

    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - best effort
        return "unknown"
    return commit.decode().strip()


def _hardware_identifiers() -> Dict[str, str]:
    """Gather lightweight identifiers describing the executing machine."""

    identifiers = {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count() or ""),
    }
    return {key: value for key, value in identifiers.items() if value}


def _calibration_metadata() -> CalibrationMetadata:
    """Construct metadata attached to every calibration run."""

    metadata: CalibrationMetadata = {
        "hardware": _hardware_identifiers(),
        "date": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "software_commit": _current_commit(),
    }
    return metadata


def _run_with_variance(
    func: "Callable[[], Dict[str, float]]", repeats: int
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Execute ``func`` ``repeats`` times returning mean and variance."""

    samples: list[Dict[str, float]] = []
    for _ in range(max(1, repeats)):
        samples.append(func())
    keys = samples[0].keys()
    mean_values: Dict[str, float] = {}
    variance_values: Dict[str, float] = {}
    for key in keys:
        measurements = [float(sample[key]) for sample in samples]
        mean_values[key] = float(fmean(measurements))
        variance_values[key] = float(pvariance(measurements)) if len(measurements) > 1 else 0.0
    return mean_values, variance_values


def _bench_loop(iters: int) -> float:
    """Run a tight Python loop ``iters`` times and return elapsed time."""
    start = perf_counter()
    total = 0
    for _ in range(iters):
        total += 1
    end = perf_counter()
    # Prevent optimisation
    if total == -1:
        print("impossible")
    return end - start


def benchmark_statevector(num_qubits: int = 8, num_gates: int = 50) -> Dict[str, float]:
    amp = 1 << num_qubits
    # Simulate state update on a dense vector
    elapsed = _bench_loop(amp * num_gates)
    coeff = elapsed / (num_gates * amp)
    return {
        "sv_gate_1q": coeff,
        "sv_gate_2q": coeff,
        "sv_meas": coeff,
        # Rough buffer factor for statevector simulation.
        "sv_bytes_per_amp": 2.0,
    }


def benchmark_statevector_baseline(num_qubits: int = 1) -> Dict[str, float]:
    """Measure fixed overhead for statevector simulation using Aer."""

    from .backends.statevector import StatevectorBackend
    from qiskit import QuantumCircuit
    import tracemalloc

    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    backend = StatevectorBackend()
    backend.prepare_benchmark(circuit)
    tracemalloc.start()
    start = perf_counter()
    backend.run_benchmark()
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {"sv_base_time": elapsed, "sv_base_mem": float(peak)}


def benchmark_tableau(num_qubits: int = 8, num_gates: int = 50) -> Dict[str, float]:
    quad = num_qubits * num_qubits
    elapsed = _bench_loop(quad * num_gates)
    return {
        "tab_gate": elapsed / (num_gates * quad),
        "tab_mem": 1.0,
    }


def benchmark_mps(
    num_qubits: int = 8,
    num_1q_gates: int = 50,
    num_2q_gates: int = 50,
    chi: int = 8,
) -> Dict[str, float]:
    chi2 = chi * chi
    chi3 = chi2 * chi
    ops_1q = num_1q_gates * num_qubits * chi2
    ops_2q = num_2q_gates * num_qubits * chi3
    logchi = math.log2(chi) if chi > 1 else 0.0
    ops_trunc = num_2q_gates * num_qubits * chi3 * logchi
    elapsed_1q = _bench_loop(int(ops_1q)) if ops_1q else 0.0
    elapsed_2q = _bench_loop(int(ops_2q)) if ops_2q else 0.0
    elapsed_trunc = _bench_loop(int(ops_trunc)) if ops_trunc else 0.0
    return {
        "mps_gate_1q": elapsed_1q / ops_1q if ops_1q else 0.0,
        "mps_gate_2q": elapsed_2q / ops_2q if ops_2q else 0.0,
        "mps_trunc": elapsed_trunc / ops_trunc if ops_trunc else 0.0,
        "mps_mem": float(chi2),
    }


def benchmark_mps_baseline(num_qubits: int = 3) -> Dict[str, float]:
    """Measure fixed overhead for MPS simulation using a small W-state."""

    from .backends.mps import MPSBackend
    from qiskit import QuantumCircuit
    import numpy as np
    import tracemalloc

    amp = np.zeros(2**num_qubits, dtype=complex)
    for i in range(num_qubits):
        amp[1 << i] = 1 / math.sqrt(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(amp, range(num_qubits))

    backend = MPSBackend()
    backend.prepare_benchmark(circuit)
    tracemalloc.start()
    start = perf_counter()
    backend.run_benchmark()
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {"mps_base_time": elapsed, "mps_base_mem": float(peak)}


def benchmark_dd_baseline(num_qubits: int = 3) -> Dict[str, float]:
    """Measure fixed overhead for decision diagram simulation."""

    from .backends.mqt_dd import DecisionDiagramBackend
    import tracemalloc

    backend = DecisionDiagramBackend()
    backend.load(num_qubits)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])
    for qubit in range(1, num_qubits):
        backend.apply_gate("CX", [0, qubit])

    tracemalloc.start()
    start = perf_counter()
    backend.run_benchmark()
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {"dd_base_time": elapsed, "dd_base_mem": float(peak)}


def benchmark_dd(num_gates: int = 50, frontier: int = 32) -> Dict[str, float]:
    ops = num_gates * frontier
    elapsed = _bench_loop(ops)
    return {
        "dd_gate": elapsed / ops,
        "dd_mem": 1.0,
    }


def benchmark_b2b(q: int = 6, s: int = 4) -> Dict[str, float]:
    ops = (s**3) + q * (s**2)
    elapsed = _bench_loop(ops)
    coeff = elapsed / ops
    return {"b2b_svd": coeff, "b2b_copy": coeff}


def benchmark_lw(w: int = 4) -> Dict[str, float]:
    dense = 1 << w
    elapsed = _bench_loop(dense)
    coeff = elapsed / dense
    return {"lw_extract": coeff}


def benchmark_st(s: int = 8) -> Dict[str, float]:
    chi = min(s, 16)
    ops = chi**3
    elapsed = _bench_loop(ops)
    coeff = elapsed / ops
    return {"st_stage": coeff}


def benchmark_full(q: int = 8) -> Dict[str, float]:
    full = 1 << q
    elapsed = _bench_loop(full)
    coeff = elapsed / full
    return {"full_extract": coeff}


def run_calibration(repeats: int = 3) -> CalibrationRecord:
    """Execute all benchmarks, returning coefficients with metadata."""

    repeats = max(1, int(repeats))
    coeff: Dict[str, float] = {}
    variance: Dict[str, float] = {}

    benchmarks: tuple[Callable[[], Dict[str, float]], ...] = (
        benchmark_statevector,
        benchmark_statevector_baseline,
        benchmark_tableau,
        benchmark_mps,
        benchmark_mps_baseline,
        benchmark_dd,
        benchmark_dd_baseline,
        benchmark_b2b,
        benchmark_lw,
        benchmark_st,
        benchmark_full,
    )

    for bench in benchmarks:
        mean_values, variance_values = _run_with_variance(bench, repeats)
        coeff.update(mean_values)
        variance.update(variance_values)

    return {
        "coeff": coeff,
        "variance": variance,
        "metadata": _calibration_metadata(),
    }


def save_coefficients(path: str | Path, coeff: CalibrationRecord) -> None:
    with Path(path).open("w") as fh:
        json.dump(coeff, fh, indent=2, sort_keys=True)


def load_coefficients(path: str | Path) -> CalibrationRecord:
    """Load calibration data from ``path`` with backward compatibility."""

    with Path(path).open() as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "coeff" in data:
        coeff_data = data.get("coeff", {})
        variance_data = data.get("variance", {})
        metadata_raw = data.get("metadata", {})
    else:
        coeff_data = data if isinstance(data, dict) else {}
        variance_data = {key: 0.0 for key in coeff_data}
        metadata_raw = {}

    coeff = {str(key): float(value) for key, value in coeff_data.items()}
    variance = {str(key): float(value) for key, value in variance_data.items()}
    metadata: CalibrationMetadata = {}
    hardware = metadata_raw.get("hardware") if isinstance(metadata_raw, dict) else None
    if isinstance(hardware, dict):
        metadata["hardware"] = {str(k): str(v) for k, v in hardware.items()}
    date = metadata_raw.get("date") if isinstance(metadata_raw, dict) else None
    if isinstance(date, str) and date:
        metadata["date"] = date
    commit = metadata_raw.get("software_commit") if isinstance(metadata_raw, dict) else None
    if isinstance(commit, str) and commit:
        metadata["software_commit"] = commit

    return {"coeff": coeff, "variance": variance, "metadata": metadata}


def latest_coefficients() -> CalibrationRecord | None:
    """Return coefficients from the newest JSON file in ``CALIBRATION_DIR``.

    Files follow the ``coeff_v*.json`` naming convention.  ``None`` is
    returned if the directory does not contain any calibration file.
    """

    if not CALIBRATION_DIR.exists():
        return None
    files = sorted(CALIBRATION_DIR.glob("coeff_v*.json"))
    if not files:
        return None
    return load_coefficients(files[-1])


def apply_calibration(
    estimator: "CostEstimator", coeff: Mapping[str, float] | CalibrationRecord
) -> None:
    """Update ``estimator`` with calibrated ``coeff`` values."""

    if isinstance(coeff, dict) and "coeff" in coeff and isinstance(coeff["coeff"], dict):
        updates = coeff["coeff"]
    else:
        updates = coeff
    estimator.update_coefficients(dict(updates))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate QuASAr cost coefficients")
    parser.add_argument(
        "--output", "-o", default="coeff.json", help="Destination JSON file"
    )
    args = parser.parse_args(argv)
    coeff = run_calibration()
    save_coefficients(args.output, coeff)
    print(f"Saved coefficients to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
