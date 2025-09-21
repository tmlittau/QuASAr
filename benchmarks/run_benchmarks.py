from __future__ import annotations

"""Execute benchmark circuits and record baseline-best results.

The script evaluates a parameterised circuit family across all single-method
simulation backends provided by :class:`quasar.cost.Backend` and QuASAr's
automatic scheduler.  For each configuration the fastest non-QuASAr backend is
determined via :func:`compute_baseline_best` and only this aggregated
"baseline_best" entry is stored alongside the QuASAr measurement.

Use the ``--verbose`` flag (repeat it for debug logging) to monitor progress
while the CLI iterates over qubit widths, backends and scenario instances.
"""

import argparse
import logging
import statistics
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from benchmark_cli import parse_qubit_range, resolve_circuit
from circuits import is_clifford
from plot_utils import compute_baseline_best
from partitioning_workloads import SCENARIOS, WorkloadInstance, iter_scenario
from runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend
from quasar.circuit import Circuit
from memory_utils import max_qubits_statevector


BASELINE_BACKENDS: tuple[Backend, ...] = (
    Backend.STATEVECTOR,
    Backend.TABLEAU,
    Backend.MPS,
    Backend.DECISION_DIAGRAM,
)


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbosity: int) -> None:
    """Initialise logging for CLI usage."""

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def run_all(
    circuit_fn,
    qubits: Iterable[int],
    repetitions: int,
    *,
    use_classical_simplification: bool = True,
    memory_bytes: int | None = None,
) -> pd.DataFrame:
    """Execute ``circuit_fn`` for each qubit count on all backends.

    The function returns a :class:`pandas.DataFrame` containing one row per
    configuration for QuASAr and the aggregated baseline best.
    """

    LOGGER.info(
        "Running %s across %d repetition(s) with %s classical simplification",
        getattr(circuit_fn, "__name__", circuit_fn),
        repetitions,
        "enabled" if use_classical_simplification else "disabled",
    )

    engine = SimulationEngine()
    runner = BenchmarkRunner()
    records: list[dict[str, object]] = []

    for n in qubits:
        LOGGER.info("Starting benchmarks for circuit width %s", n)
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
            LOGGER.info("Skipping width %s because the circuit is Clifford-only", n)
            LOGGER.info("Finished benchmarks for circuit width %s", n)
            continue

        for backend in BASELINE_BACKENDS:
            LOGGER.info(
                "Running backend %s for width %s", backend.value, n
            )
            try:
                rec = runner.run_quasar_multiple(
                    circuit,
                    engine,
                    backend=backend,
                    repetitions=repetitions,
                    quick=False,
                    memory_bytes=memory_bytes,
                )
            except (RuntimeError, ValueError) as exc:
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
                LOGGER.warning(
                    "Backend %s failed for width %s: %s",
                    backend.value,
                    n,
                    exc,
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
            LOGGER.info(
                "Completed backend %s for width %s", backend.value, n
            )

        LOGGER.info("Running QuASAr scheduler for width %s", n)
        quasar_rec = runner.run_quasar_multiple(
            circuit,
            engine,
            repetitions=repetitions,
            quick=False,
            memory_bytes=memory_bytes,
        )
        quasar_rec.pop("result", None)
        backend_name = quasar_rec.get("backend")
        if isinstance(backend_name, str) and backend_name in Backend.__members__:
            quasar_rec["backend"] = Backend[backend_name].value
        quasar_rec.update(
            {"circuit": circuit_fn.__name__, "qubits": n, "framework": "quasar"}
        )
        records.append(quasar_rec)
        LOGGER.info("Completed QuASAr run for width %s", n)
        LOGGER.info("Finished benchmarks for circuit width %s", n)

    df = pd.DataFrame(records)
    if df.empty or "framework" not in df.columns:
        return df
    try:
        baseline_best = compute_baseline_best(
            df,
            metrics=(
                "run_time_mean",
                "total_time_mean",
                "run_peak_memory_mean",
            ),
        )
        quasar_df = df[df["framework"] == "quasar"]
        return pd.concat([baseline_best, quasar_df], ignore_index=True)
    except ValueError:
        # All baselines failed or are unsupported; return QuASAr data only.
        return df[df["framework"] == "quasar"].reset_index(drop=True)


def run_scenarios(
    instances: Iterable[WorkloadInstance],
    repetitions: int,
    *,
    memory_bytes: int | None = None,
) -> pd.DataFrame:
    """Execute scenario workloads across all backends."""

    LOGGER.info(
        "Running partitioning scenarios for %d repetition(s)", repetitions
    )

    engine = SimulationEngine()
    runner = BenchmarkRunner()
    scheduler = getattr(engine, "scheduler", engine)
    planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
    records: List[dict[str, object]] = []

    def _conversion_summary(layers: List[Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "conversion_count": len(layers),
            "conversion_boundary_mean": None,
            "conversion_rank_mean": None,
            "conversion_frontier_mean": None,
            "conversion_primitive_summary": None,
        }
        if not layers:
            return summary
        boundaries = [len(getattr(layer, "boundary", ())) for layer in layers]
        ranks = [getattr(layer, "rank", None) for layer in layers]
        frontiers = [getattr(layer, "frontier", None) for layer in layers]
        primitives = Counter(getattr(layer, "primitive", None) for layer in layers)
        if any(boundaries):
            summary["conversion_boundary_mean"] = statistics.fmean(boundaries)
        if any(ranks):
            summary["conversion_rank_mean"] = statistics.fmean(
                [r for r in ranks if r is not None]
            )
        if any(frontiers):
            summary["conversion_frontier_mean"] = statistics.fmean(
                [f for f in frontiers if f is not None]
            )
        if primitives:
            parts = [
                f"{name}:{count}"
                for name, count in sorted(primitives.items())
                if name
            ]
            summary["conversion_primitive_summary"] = ", ".join(parts) if parts else None
        return summary

    for instance in instances:
        instance_label = f"{instance.scenario}:{instance.variant}"
        LOGGER.info("Starting scenario instance %s", instance_label)
        metadata: Dict[str, object] = {}

        def _build_circuit() -> Circuit:
            circuit = instance.build()
            meta = dict(getattr(circuit, "metadata", {}))
            metadata.clear()
            metadata.update(meta)
            if instance.enable_classical_simplification:
                enable = getattr(circuit, "enable_classical_simplification", None)
                if callable(enable):
                    enable()
                else:
                    circuit.use_classical_simplification = True
            else:
                circuit.use_classical_simplification = False
            return circuit

        def _execute_backend(backend: Backend | None) -> dict[str, object]:
            circuit = _build_circuit()
            total_qubits = metadata.get(
                "total_qubits", getattr(circuit, "num_qubits", None)
            )
            if (
                backend == Backend.STATEVECTOR
                and total_qubits is not None
                and total_qubits > max_qubits_statevector(memory_bytes)
            ):
                limit = max_qubits_statevector(memory_bytes)
                LOGGER.info(
                    "Skipping backend %s for %s: width %s exceeds statevector limit %s",
                    backend.value if backend else Backend.STATEVECTOR.value,
                    instance_label,
                    total_qubits,
                    limit,
                )
                return {
                    "framework": backend.value if backend else "quasar",
                    "backend": backend.value if backend else Backend.STATEVECTOR.value,
                    "unsupported": True,
                    "comment": (
                        f"circuit width {total_qubits} exceeds statevector limit {limit}"
                    ),
                }

            tracemalloc.start()
            start_prepare = time.perf_counter()
            try:
                if planner is not None:
                    LOGGER.debug(
                        "Preparing plan for %s using backend %s",
                        instance_label,
                        backend.value if backend else "auto",
                    )
                    plan = planner.plan(circuit, backend=backend)
                    plan = scheduler.prepare_run(circuit, plan, backend=backend)
                else:
                    LOGGER.debug(
                        "Preparing scheduler run for %s using backend %s",
                        instance_label,
                        backend.value if backend else "auto",
                    )
                    plan = scheduler.prepare_run(circuit, backend=backend)
            except ValueError as exc:
                tracemalloc.stop()
                LOGGER.warning(
                    "Preparation failed for %s on backend %s: %s",
                    instance_label,
                    backend.value if backend else "auto",
                    exc,
                )
                return {
                    "framework": backend.value if backend else "quasar",
                    "backend": backend.value if backend else str(backend),
                    "unsupported": True,
                    "error": str(exc),
                }
            prepare_time = time.perf_counter() - start_prepare
            _, prepare_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            conversion_layers = list(getattr(plan, "conversions", []))
            try:
                LOGGER.info(
                    "Executing scheduler for %s on backend %s",
                    instance_label,
                    backend.value if backend else "auto",
                )
                result, metrics = scheduler.run(circuit, plan, instrument=True)
            except Exception as exc:
                LOGGER.warning(
                    "Execution failed for %s on backend %s: %s",
                    instance_label,
                    backend.value if backend else "auto",
                    exc,
                )
                return {
                    "framework": backend.value if backend else "quasar",
                    "backend": backend.value if backend else str(backend),
                    "failed": True,
                    "error": str(exc),
                }
            cost = getattr(metrics, "cost", metrics)
            run_time = getattr(cost, "time", 0.0)
            run_peak = int(getattr(cost, "memory", 0.0))
            backend_choice = None
            if backend is not None:
                backend_choice = backend.value
            elif hasattr(result, "partitions") and getattr(result, "partitions"):
                backend_obj = result.partitions[0].backend
                backend_choice = getattr(backend_obj, "value", str(backend_obj))
            record = {
                "framework": backend.value if backend is not None else "quasar",
                "prepare_time_mean": prepare_time,
                "prepare_time_std": 0.0,
                "run_time_mean": run_time,
                "run_time_std": 0.0,
                "total_time_mean": prepare_time + run_time,
                "total_time_std": 0.0,
                "prepare_peak_memory_mean": int(prepare_peak),
                "prepare_peak_memory_std": 0.0,
                "run_peak_memory_mean": run_peak,
                "run_peak_memory_std": 0.0,
                "repetitions": 1,
                "backend": backend_choice,
                "result": result,
                "failed": False,
            }
            record.update(_conversion_summary(conversion_layers))
            if total_qubits is not None:
                record["qubits"] = total_qubits
            return record

        for backend in BASELINE_BACKENDS:
            LOGGER.info(
                "Benchmarking backend %s for scenario %s",
                backend.value,
                instance_label,
            )
            record = _execute_backend(backend)
            record.update({"circuit": instance.builder.__name__, "framework": backend.value})
            record.update(metadata)
            record.pop("result", None)
            records.append(record)
            LOGGER.info(
                "Completed backend %s for scenario %s",
                backend.value,
                instance_label,
            )

        LOGGER.info("Benchmarking QuASAr for scenario %s", instance_label)
        quasar_record = _execute_backend(None)
        quasar_record.update({"circuit": instance.builder.__name__, "framework": "quasar"})
        quasar_record.update(metadata)
        quasar_record.pop("result", None)
        records.append(quasar_record)
        LOGGER.info("Completed QuASAr run for scenario %s", instance_label)
        LOGGER.info("Completed scenario %s", instance_label)

    df = pd.DataFrame(records)
    if df.empty or "framework" not in df.columns:
        return df
    try:
        baseline_best = compute_baseline_best(
            df,
            metrics=(
                "run_time_mean",
                "total_time_mean",
                "run_peak_memory_mean",
            ),
        )
        quasar_df = df[df["framework"] == "quasar"]
        return pd.concat([baseline_best, quasar_df], ignore_index=True)
    except ValueError:
        return df[df["framework"] == "quasar"].reset_index(drop=True)

def save_results(df: pd.DataFrame, output: Path) -> None:
    """Persist ``df`` as CSV and JSON using ``output`` as base path."""

    base = output.with_suffix("")
    csv_path = base.with_suffix(".csv")
    json_path = base.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)


def summarise_partitioning(df: pd.DataFrame) -> pd.DataFrame:
    """Return an aggregate table comparing QuASAr and baseline metrics."""

    if df.empty:
        return df

    relevant = df[df["framework"].isin({"baseline_best", "quasar"})]
    if relevant.empty:
        return relevant

    base_keys = [
        col
        for col in ("scenario", "variant", "circuit", "qubits")
        if col in relevant.columns
    ]
    metadata_columns = [
        col
        for col in (
            "boundary",
            "schmidt_layers",
            "cross_layers",
            "suffix_sparsity",
            "prefix_core_boundary",
            "core_suffix_boundary",
            "dense_qubits",
            "clifford_qubits",
            "core_qubits",
            "suffix_qubits",
            "total_qubits",
        )
        if col in relevant.columns
    ]
    rows: List[dict[str, object]] = []

    baseline_frameworks = {backend.value for backend in BASELINE_BACKENDS}

    for keys, group in relevant.groupby(base_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(base_keys, keys))

        quasar = group[group["framework"] == "quasar"]
        meta_source = quasar if not quasar.empty else group
        if not meta_source.empty:
            head = meta_source.iloc[0]
            for col in metadata_columns:
                row.setdefault(col, head.get(col))

        baseline = group[group["framework"] == "baseline_best"]
        if not baseline.empty:
            b = baseline.iloc[0]
            row["baseline_runtime_mean"] = b.get("run_time_mean")
            row["baseline_backend"] = b.get("backend")
            peak_memory = b.get("run_peak_memory_mean")
            if pd.isna(peak_memory):
                mask = df["framework"].isin(baseline_frameworks)
                for name, value in zip(base_keys, keys):
                    mask &= df[name] == value
                candidates = df[mask]
                if (
                    not candidates.empty
                    and "run_time_mean" in candidates.columns
                    and "run_peak_memory_mean" in candidates.columns
                ):
                    idx = candidates["run_time_mean"].idxmin()
                    peak_memory = candidates.loc[idx, "run_peak_memory_mean"]
            row["baseline_peak_memory_mean"] = peak_memory
        else:
            row["baseline_runtime_mean"] = None
            row["baseline_peak_memory_mean"] = None
            row["baseline_backend"] = None

        if not quasar.empty:
            q = quasar.iloc[0]
            row["quasar_runtime_mean"] = q.get("run_time_mean")
            row["quasar_peak_memory_mean"] = q.get("run_peak_memory_mean")
            row["quasar_backend"] = q.get("backend")
            row["quasar_conversions"] = q.get("conversion_count")
            row["quasar_boundary_mean"] = q.get("conversion_boundary_mean")
            row["quasar_rank_mean"] = q.get("conversion_rank_mean")
            row["quasar_frontier_mean"] = q.get("conversion_frontier_mean")
            row["quasar_conversion_primitives"] = q.get(
                "conversion_primitive_summary"
            )
        else:
            row["quasar_runtime_mean"] = None
            row["quasar_peak_memory_mean"] = None
            row["quasar_backend"] = None
            row["quasar_conversions"] = None
            row["quasar_boundary_mean"] = None
            row["quasar_rank_mean"] = None
            row["quasar_frontier_mean"] = None
            row["quasar_conversion_primitives"] = None

        try:
            if row["baseline_runtime_mean"] and row["quasar_runtime_mean"]:
                row["runtime_speedup"] = (
                    row["baseline_runtime_mean"] / row["quasar_runtime_mean"]
                )
            else:
                row["runtime_speedup"] = None
        except ZeroDivisionError:
            row["runtime_speedup"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def write_partitioning_tables(df: pd.DataFrame, output: Path) -> None:
    """Create CSV and Markdown summaries for partitioning benchmarks."""

    summary = summarise_partitioning(df)
    if summary.empty:
        return
    base = output.with_suffix("")
    summary_base = base.with_name(base.name + "_summary")
    summary_csv = summary_base.with_suffix(".csv")
    summary_md = summary_base.with_suffix(".md")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    try:
        with summary_md.open("w", encoding="utf-8") as handle:
            summary.to_markdown(handle, index=False)
    except ImportError:
        with summary_md.open("w", encoding="utf-8") as handle:
            handle.write(summary.to_string(index=False))
            handle.write("\n")


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="Execute benchmark circuits and record baseline-best results"
    )
    parser.add_argument("--circuit", help="Circuit family name (e.g. ghz, qft)")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        help="Named partitioning scenario defined in partitioning_workloads",
    )
    parser.add_argument(
        "--qubits",
        type=parse_qubit_range,
        help="Qubit range as start:end[:step] (for --circuit runs)",
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
    parser.add_argument(
        "--memory-bytes",
        type=int,
        help="Approximate peak memory budget per backend (bytes)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase logging verbosity (use -vv for debug output)."
        ),
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if bool(args.circuit) == bool(args.scenario):
        parser.error("exactly one of --circuit or --scenario must be provided")

    if args.circuit:
        if args.qubits is None:
            parser.error("--qubits is required when --circuit is used")
        circuit_fn = resolve_circuit(args.circuit)
        df = run_all(
            circuit_fn,
            args.qubits,
            args.repetitions,
            use_classical_simplification=not args.disable_classical_simplify,
            memory_bytes=args.memory_bytes,
        )
        save_results(df, args.output)
    else:
        instances = iter_scenario(args.scenario)
        df = run_scenarios(instances, args.repetitions, memory_bytes=args.memory_bytes)
        save_results(df, args.output)
        write_partitioning_tables(df, args.output)


# Import surface-code protected circuits so the CLI can discover them.
from circuits import surface_corrected_qaoa_circuit  # noqa: E402,F401


if __name__ == "__main__":
    main()

