"""Estimate the runtime of :mod:`paper_figures` without executing benchmarks.

This helper mirrors the circuit families and backend combinations used by
``benchmarks/paper_figures.py`` and relies on QuASAr's static cost model to
approximate how expensive each run will be.  The script surfaces the per-run
estimate for every forced backend execution as well as the automatic QuASAr
schedule.  Estimates are reported both in raw cost-model "operations" and as an
approximate runtime assuming a configurable operations-per-second throughput.

The output acts as an early warning system: combinations that require
exponentially many operations (e.g., forcing dense statevector simulation of
32-qubit circuits) are easy to spot before starting the full benchmark suite.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

if __package__ in {None, ""}:  # pragma: no cover - script execution
    import sys

    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Ensure the local ``quasar`` package is importable before loading
    # ``paper_figures`` which depends on it.
    import importlib

    importlib.import_module("quasar")
    import paper_figures as paper_figures  # type: ignore[no-redef]
    from quasar.analyzer import CircuitAnalyzer  # type: ignore[no-redef]
    from quasar.calibration import (  # type: ignore[no-redef]
        apply_calibration,
        latest_coefficients,
        load_coefficients,
    )
    from quasar.cost import Backend, Cost, CostEstimator  # type: ignore[no-redef]
    from progress import ProgressReporter  # type: ignore[no-redef]
else:  # pragma: no cover - package import path
    from . import paper_figures
    from quasar.analyzer import CircuitAnalyzer
    from quasar.calibration import (
        apply_calibration,
        latest_coefficients,
        load_coefficients,
    )
    from quasar.cost import Backend, Cost, CostEstimator
    from .progress import ProgressReporter


OPS_PER_SECOND_DEFAULT = 1_000_000_000.0
"""Fallback throughput (1 GFLOP/s) used to approximate wall-clock time."""


@dataclass
class EstimateRecord:
    """Container for a single runtime estimate."""

    circuit: str
    qubits: int
    mode: str
    backend: str
    supported: bool
    cost: Cost | None
    note: str | None = None

    def approx_seconds(self, ops_per_second: float | None) -> float | None:
        """Return the runtime in seconds under the supplied throughput."""

        if not self.supported or self.cost is None or ops_per_second in (None, 0):
            return None
        return self.cost.time / ops_per_second


def _human_readable(value: float, *, unit: str = "", precision: int = 2) -> str:
    """Return ``value`` formatted using SI prefixes."""

    if value is None or math.isnan(value):
        return "n/a"
    if value == 0:
        return f"0{unit}"
    magnitude = int(math.floor(math.log10(abs(value)) / 3)) if value else 0
    magnitude = max(min(magnitude, 8), -8)
    scaled = value / (10 ** (3 * magnitude))
    prefixes = {
        -8: "y",
        -7: "z",
        -6: "a",
        -5: "f",
        -4: "p",
        -3: "n",
        -2: "µ",
        -1: "m",
        0: "",
        1: "k",
        2: "M",
        3: "G",
        4: "T",
        5: "P",
        6: "E",
        7: "Z",
        8: "Y",
    }
    prefix = prefixes.get(magnitude, "")
    return f"{scaled:.{precision}f} {prefix}{unit}".strip()


def _format_seconds(seconds: float | None) -> str:
    """Format seconds using human-friendly units."""

    if seconds is None:
        return "n/a"
    if seconds < 1:
        return _human_readable(seconds, unit="s", precision=3)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {rem:.0f} s"
    hours, rem = divmod(minutes, 60)
    if hours < 24:
        return f"{int(hours)} h {int(rem)} min"
    days, rem = divmod(hours, 24)
    if days < 365:
        return f"{int(days)} d {int(rem)} h"
    years, rem = divmod(days, 365)
    return f"{int(years)} y {int(rem)} d"


def _load_estimator(calibration: Path | None) -> CostEstimator:
    """Return a cost estimator optionally initialised from calibration data."""

    estimator = CostEstimator()
    coeff: dict[str, float] | None = None
    if calibration is not None:
        coeff = load_coefficients(calibration)
    else:
        coeff = latest_coefficients()
    if coeff:
        apply_calibration(estimator, coeff)
    return estimator


def _estimate_forced(
    specs: Iterable[paper_figures.CircuitSpec],
    backends: Sequence[Backend],
    estimator: CostEstimator,
) -> list[EstimateRecord]:
    """Return cost estimates for the forced backend runs."""

    spec_list = list(specs)
    backend_list = list(backends)
    total_steps = sum(len(spec.qubits) * len(backend_list) for spec in spec_list)
    progress = (
        ProgressReporter(total_steps, prefix="Forced estimates")
        if total_steps
        else None
    )

    records: list[EstimateRecord] = []
    for spec in spec_list:
        for n in spec.qubits:
            circuit = paper_figures._build_circuit(
                spec, n, use_classical_simplification=False
            )
            analyzer = CircuitAnalyzer(circuit, estimator=estimator)
            resources = analyzer.resource_estimates()
            for backend in backend_list:
                supported = paper_figures._supports_backend(circuit, backend)
                cost = resources.get(backend)
                note = None
                if not supported:
                    note = "unsupported gate set"
                records.append(
                    EstimateRecord(
                        circuit=spec.name,
                        qubits=n,
                        mode="forced",
                        backend=backend.name.lower(),
                        supported=supported,
                        cost=cost,
                        note=note,
                    )
                )
                if progress:
                    progress.advance(f"{spec.name}@{n} {backend.name.lower()}")

    if progress:
        progress.close()
    return records


def _estimate_auto(
    specs: Iterable[paper_figures.CircuitSpec],
    estimator: CostEstimator,
) -> list[EstimateRecord]:
    """Return heuristic cost estimates for automatic scheduling runs."""

    spec_list = list(specs)
    total_steps = sum(len(spec.qubits) for spec in spec_list)
    progress = (
        ProgressReporter(total_steps, prefix="Auto estimates") if total_steps else None
    )

    records: list[EstimateRecord] = []
    for spec in spec_list:
        for n in spec.qubits:
            circuit = paper_figures._build_circuit(
                spec, n, use_classical_simplification=True
            )
            analyzer = CircuitAnalyzer(circuit, estimator=estimator)
            resources = analyzer.resource_estimates()
            supported: list[tuple[Backend, Cost]] = []
            for backend, cost in resources.items():
                if paper_figures._supports_backend(circuit, backend):
                    supported.append((backend, cost))
            if not supported:
                records.append(
                    EstimateRecord(
                        circuit=spec.name,
                        qubits=n,
                        mode="auto",
                        backend="n/a",
                        supported=False,
                        cost=None,
                        note="no compatible backend available",
                    )
                )
                continue
            backend, cost = min(supported, key=lambda item: item[1].time)
            note = "single-backend approximation"
            records.append(
                EstimateRecord(
                    circuit=spec.name,
                    qubits=n,
                    mode="auto",
                    backend=backend.name.lower(),
                    supported=True,
                    cost=cost,
                    note=note,
                )
            )
            if progress:
                progress.advance(f"{spec.name}@{n}")

    if progress:
        progress.close()
    return records


def _print_records(
    title: str,
    records: Sequence[EstimateRecord],
    repetitions: int,
    ops_per_second: float | None,
) -> float:
    """Print ``records`` as a table and return the cumulative seconds."""

    print(f"\n{title}")
    header = f"{'Circuit':<18}{'Qubits':>6}  {'Backend':<18}{'Per-run cost':>15}  {'Approx time':>13}  {'Peak memory':>14}  Notes"
    print(header)
    print("-" * len(header))
    total_seconds = 0.0
    for rec in records:
        cost = rec.cost
        ops = cost.time if cost is not None else float("nan")
        approx = rec.approx_seconds(ops_per_second)
        mem = cost.memory if cost is not None else float("nan")
        if rec.supported and approx is not None:
            total_seconds += approx * repetitions
        notes = rec.note or ("unsupported" if not rec.supported else "")
        ops_str = _human_readable(ops, unit="ops") if cost is not None else "n/a"
        approx_str = _format_seconds(approx)
        mem_str = _human_readable(mem, unit="B") if cost is not None else "n/a"
        print(
            f"{rec.circuit:<18}{rec.qubits:>6}  {rec.backend:<18}{ops_str:>15}  {approx_str:>13}  {mem_str:>14}  {notes}"
        )
    return total_seconds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the runtime of benchmarks/paper_figures.py",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions used in paper_figures (default: 3).",
    )
    parser.add_argument(
        "--ops-per-second",
        type=float,
        default=OPS_PER_SECOND_DEFAULT,
        help=(
            "Throughput used to convert cost model operations to seconds. "
            "Set to zero to disable the conversion (default: 1e9)."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with calibrated cost coefficients.  When not "
            "provided the newest file from calibration/ is used if present."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to store the raw estimate data as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")

    estimator = _load_estimator(args.calibration)
    forced = _estimate_forced(paper_figures.CIRCUITS, paper_figures.BACKENDS, estimator)
    auto = _estimate_auto(paper_figures.CIRCUITS, estimator)

    ops_per_second = args.ops_per_second if args.ops_per_second > 0 else None
    if ops_per_second is None:
        print("Runtime conversion disabled (ops-per-second <= 0).")
    else:
        print(
            "Assuming throughput of "
            f"{_human_readable(ops_per_second, unit='ops/s', precision=3)}"
        )
    forced_total = _print_records(
        "Forced backend runs (per repetition)", forced, args.repetitions, ops_per_second
    )
    auto_total = _print_records(
        "Automatic QuASAr runs (per repetition)", auto, args.repetitions, ops_per_second
    )

    grand_total = forced_total + auto_total
    if ops_per_second is not None:
        print("\nEstimated cumulative runtime with repetitions:")
        print(f"  Forced runs : {_format_seconds(forced_total)}")
        print(f"  Auto runs   : {_format_seconds(auto_total)}")
        print(f"  Grand total : {_format_seconds(grand_total)}")

    if args.json is not None:
        payload = [
            {
                "circuit": rec.circuit,
                "qubits": rec.qubits,
                "mode": rec.mode,
                "backend": rec.backend,
                "supported": rec.supported,
                "cost_time": rec.cost.time if rec.cost else None,
                "cost_memory": rec.cost.memory if rec.cost else None,
                "note": rec.note,
            }
            for rec in (*forced, *auto)
        ]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
