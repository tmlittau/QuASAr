from __future__ import annotations

"""Execute benchmark circuits and record baseline-best results.

The script evaluates a parameterised circuit family across all single-method
simulation backends provided by :class:`quasar.cost.Backend` and QuASAr's
automatic scheduler.  For each configuration the fastest non-QuASAr backend is
determined via :func:`compute_baseline_best` and only this aggregated
"baseline_best" entry is stored alongside the QuASAr measurement.
"""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from benchmark_cli import parse_qubit_range, resolve_circuit
from circuits import is_clifford
from plot_utils import compute_baseline_best
from runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend


BASELINE_BACKENDS: tuple[Backend, ...] = (
    Backend.STATEVECTOR,
    Backend.TABLEAU,
    Backend.MPS,
    Backend.DECISION_DIAGRAM,
)


def run_all(
    circuit_fn,
    qubits: Iterable[int],
    repetitions: int,
    *,
    use_classical_simplification: bool = True,
) -> pd.DataFrame:
    """Execute ``circuit_fn`` for each qubit count on all backends.

    The function returns a :class:`pandas.DataFrame` containing one row per
    configuration for QuASAr and the aggregated baseline best.
    """

    engine = SimulationEngine()
    runner = BenchmarkRunner()
    records: list[dict[str, object]] = []

    for n in qubits:
        circuit = circuit_fn(n)
        if use_classical_simplification:
            enable = getattr(circuit, "enable_classical_simplification", None)
            if callable(enable):
                enable()
            else:
                circuit.use_classical_simplification = True
        else:
            circuit.use_classical_simplification = False

        if is_clifford(circuit):
            continue

        for backend in BASELINE_BACKENDS:
            try:
                rec = runner.run_quasar_multiple(
                    circuit,
                    engine,
                    backend=backend,
                    repetitions=repetitions,
                    quick=True,
                )
            except RuntimeError as exc:
                records.append(
                    {
                        "circuit": circuit_fn.__name__,
                        "qubits": n,
                        "framework": backend.value,
                        "backend": backend.value,
                        "unsupported": True,
                        "error": str(exc),
                    }
                )
                continue

            rec.pop("result", None)
            rec.update(
                {
                    "circuit": circuit_fn.__name__,
                    "qubits": n,
                    "framework": backend.value,
                    "backend": backend.value,
                }
            )
            records.append(rec)

        quasar_rec = runner.run_quasar_multiple(
            circuit, engine, repetitions=repetitions, quick=True
        )
        quasar_rec.pop("result", None)
        backend_name = quasar_rec.get("backend")
        if isinstance(backend_name, str) and backend_name in Backend.__members__:
            quasar_rec["backend"] = Backend[backend_name].value
        quasar_rec.update(
            {"circuit": circuit_fn.__name__, "qubits": n, "framework": "quasar"}
        )
        records.append(quasar_rec)

    df = pd.DataFrame(records)
    if df.empty or "framework" not in df.columns:
        return df
    try:
        baseline_best = compute_baseline_best(df)
        quasar_df = df[df["framework"] == "quasar"]
        return pd.concat([baseline_best, quasar_df], ignore_index=True)
    except ValueError:
        # All baselines failed or are unsupported; return QuASAr data only.
        return df[df["framework"] == "quasar"].reset_index(drop=True)


def save_results(df: pd.DataFrame, output: Path) -> None:
    """Persist ``df`` as CSV and JSON using ``output`` as base path."""

    base = output.with_suffix("")
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="Execute benchmark circuits and record baseline-best results"
    )
    parser.add_argument(
        "--circuit", required=True, help="Circuit family name (e.g. ghz, qft)"
    )
    parser.add_argument(
        "--qubits",
        required=True,
        type=parse_qubit_range,
        help="Qubit range as start:end[:step]",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per configuration",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output file path without extension"
    )
    parser.add_argument(
        "--disable-classical-simplify",
        action="store_true",
        help="Disable classical control simplification",
    )
    args = parser.parse_args()

    circuit_fn = resolve_circuit(args.circuit)
    df = run_all(
        circuit_fn,
        args.qubits,
        args.repetitions,
        use_classical_simplification=not args.disable_classical_simplify,
    )
    save_results(df, args.output)


# Import surface-code protected circuits so the CLI can discover them.
from circuits import surface_corrected_qaoa_circuit  # noqa: E402,F401


if __name__ == "__main__":
    main()

