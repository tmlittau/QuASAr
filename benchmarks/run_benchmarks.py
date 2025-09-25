from __future__ import annotations

"""Execute benchmark circuits and record baseline-best results.

The script evaluates a parameterised circuit family across all single-method
simulation backends provided by :class:`quasar.cost.Backend` and QuASAr's
automatic scheduler.  For each configuration the fastest non-QuASAr backend is
determined via :func:`compute_baseline_best` and only this aggregated
"baseline_best" entry is stored alongside the QuASAr measurement.

Use the ``--verbose`` flag (repeat it for debug logging) to monitor progress
while the CLI iterates over qubit widths, backends and scenario instances.

Example
-------
Run the dual magic injection scenario with a custom memory cap::

    python benchmarks/run_benchmarks.py \
        --scenario dual_magic_injection \
        --repetitions 3 \
        --memory-bytes 2147483648 \
        --output benchmarks/results/dual_magic_injection
"""

import argparse
import copy
import logging
import os
import statistics
import time
import tracemalloc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from benchmark_cli import parse_qubit_range, resolve_circuit
from circuits import is_clifford, is_clifford_plus_t
from plot_utils import compute_baseline_best
from partitioning_workloads import SCENARIOS, WorkloadInstance, iter_scenario
from runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.cost import Backend
from quasar.circuit import Circuit
from memory_utils import max_qubits_statevector

try:  # shared utilities for both package and script execution
    from .progress import ProgressReporter
    from .ssd_metrics import partition_metrics_from_result
    from .threading_utils import resolve_worker_count, thread_engine
except ImportError:  # pragma: no cover - fallback when executed as a script
    from progress import ProgressReporter  # type: ignore
    from ssd_metrics import partition_metrics_from_result  # type: ignore
    from threading_utils import resolve_worker_count, thread_engine  # type: ignore


BASELINE_BACKENDS: tuple[Backend, ...] = (
    Backend.STATEVECTOR,
    Backend.EXTENDED_STABILIZER,
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


def _run_all_for_width(
    engine: SimulationEngine,
    circuit_fn,
    n: int,
    repetitions: int,
    use_classical_simplification: bool,
    memory_bytes: int | None,
) -> tuple[list[dict[str, object]], list[str]]:
    """Execute all baselines and QuASAr for a single circuit width."""

    runner = BenchmarkRunner()
    records: list[dict[str, object]] = []
    messages: list[str] = []

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
        skip_msg = f"skip width {n} (Clifford)"
        messages.extend([skip_msg] * (len(BASELINE_BACKENDS) + 1))
        return records, messages

    for backend in BASELINE_BACKENDS:
        status_msg = f"{backend.value}@{n}"
        if (
            backend == Backend.EXTENDED_STABILIZER
            and not is_clifford_plus_t(circuit)
        ):
            messages.append(f"{status_msg} (skipped)")
            continue
        circuit_copy = copy.deepcopy(circuit)
        try:
            rec = runner.run_quasar_multiple(
                circuit_copy,
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
        else:
            result = rec.pop("result", None)
            rec.update(partition_metrics_from_result(result))
            rec.update(
                {
                    "circuit": circuit_fn.__name__,
                    "qubits": n,
                    "framework": backend.value,
                    "backend": backend.value,
                }
            )
            records.append(rec)
            LOGGER.info("Completed backend %s for width %s", backend.value, n)
        finally:
            messages.append(status_msg)

    quasar_status = f"quasar@{n}"
    circuit_copy = copy.deepcopy(circuit)
    try:
        quasar_rec = runner.run_quasar_multiple(
            circuit_copy,
            engine,
            repetitions=repetitions,
            quick=False,
            memory_bytes=memory_bytes,
        )
    finally:
        messages.append(quasar_status)

    result = quasar_rec.pop("result", None)
    quasar_rec.update(partition_metrics_from_result(result))
    backend_name = quasar_rec.get("backend")
    if isinstance(backend_name, str) and backend_name in Backend.__members__:
        quasar_rec["backend"] = Backend[backend_name].value
    quasar_rec.update(
        {"circuit": circuit_fn.__name__, "qubits": n, "framework": "quasar"}
    )
    records.append(quasar_rec)

    return records, messages


def _run_all_for_width_worker(
    circuit_fn,
    n: int,
    repetitions: int,
    use_classical_simplification: bool,
    memory_bytes: int | None,
) -> tuple[list[dict[str, object]], list[str]]:
    engine = thread_engine()
    return _run_all_for_width(
        engine,
        circuit_fn,
        n,
        repetitions,
        use_classical_simplification,
        memory_bytes,
    )


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


def run_all(
    circuit_fn,
    qubits: Iterable[int],
    repetitions: int,
    *,
    use_classical_simplification: bool = True,
    memory_bytes: int | None = None,
    max_workers: int | None = None,
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

    records: list[dict[str, object]] = []

    qubit_list = list(qubits)
    if not qubit_list:
        return pd.DataFrame()

    total_steps = len(qubit_list) * (len(BASELINE_BACKENDS) + 1)
    progress = ProgressReporter(total_steps, prefix="Circuit benchmark")
    worker_count = resolve_worker_count(max_workers, len(qubit_list))

    LOGGER.info("Using %d worker thread(s) for circuit benchmarks", worker_count)

    try:
        if worker_count <= 1:
            engine = SimulationEngine()
            for n in qubit_list:
                LOGGER.info("Starting benchmarks for circuit width %s", n)
                recs, messages = _run_all_for_width(
                    engine,
                    circuit_fn,
                    n,
                    repetitions,
                    use_classical_simplification,
                    memory_bytes,
                )
                records.extend(recs)
                for msg in messages:
                    progress.advance(msg)
                LOGGER.info("Finished benchmarks for circuit width %s", n)
        else:
            futures: Dict[Any, int] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for idx, n in enumerate(qubit_list):
                    LOGGER.info("Submitting benchmarks for circuit width %s", n)
                    future = executor.submit(
                        _run_all_for_width_worker,
                        circuit_fn,
                        n,
                        repetitions,
                        use_classical_simplification,
                        memory_bytes,
                    )
                    futures[future] = idx

                ordered: Dict[int, list[dict[str, object]]] = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    n = qubit_list[idx]
                    try:
                        recs, messages = future.result()
                    except Exception:
                        progress.close()
                        raise
                    ordered[idx] = recs
                    for msg in messages:
                        progress.advance(msg)
                    LOGGER.info("Completed benchmarks for circuit width %s", n)

            for idx in range(len(qubit_list)):
                records.extend(ordered.get(idx, []))
    finally:
        progress.close()

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


def _run_scenario_instance(
    engine: SimulationEngine,
    instance: WorkloadInstance,
    memory_bytes: int | None,
) -> tuple[list[dict[str, object]], list[str]]:
    """Execute all baselines and QuASAr for a single workload instance."""

    scheduler = getattr(engine, "scheduler", engine)
    planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
    instance_label = f"{instance.scenario}:{instance.variant}"
    records: List[dict[str, object]] = []
    messages: List[str] = []
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
        except Exception as exc:  # pragma: no cover - defensive guard
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
        record.update(partition_metrics_from_result(result))
        if total_qubits is not None:
            record["qubits"] = total_qubits
        return record

    for backend in BASELINE_BACKENDS:
        LOGGER.info(
            "Benchmarking backend %s for scenario %s",
            backend.value,
            instance_label,
        )
        status_msg = f"{instance_label} {backend.value}"
        record = _execute_backend(backend)
        record.update({"circuit": instance.builder.__name__, "framework": backend.value})
        record.update(metadata)
        record.pop("result", None)
        records.append(record)
        messages.append(status_msg)
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
    messages.append(f"{instance_label} quasar")
    LOGGER.info("Completed QuASAr run for scenario %s", instance_label)

    return records, messages


def _run_scenario_instance_worker(
    instance: WorkloadInstance,
    memory_bytes: int | None,
) -> tuple[list[dict[str, object]], list[str]]:
    engine = _thread_engine()
    return _run_scenario_instance(engine, instance, memory_bytes)


def run_scenarios(
    instances: Iterable[WorkloadInstance],
    repetitions: int,
    *,
    memory_bytes: int | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Execute scenario workloads across all backends."""

    LOGGER.info(
        "Running partitioning scenarios for %d repetition(s)", repetitions
    )

    records: List[dict[str, object]] = []

    instance_list = list(instances)
    if not instance_list:
        return pd.DataFrame()

    total_steps = len(instance_list) * (len(BASELINE_BACKENDS) + 1)
    progress = ProgressReporter(total_steps, prefix="Scenario benchmark")
    worker_count = _resolve_workers(max_workers, len(instance_list))

    LOGGER.info("Using %d worker thread(s) for scenario benchmarks", worker_count)

    try:
        if worker_count <= 1:
            engine = SimulationEngine()
            for instance in instance_list:
                label = f"{instance.scenario}:{instance.variant}"
                LOGGER.info("Starting scenario instance %s", label)
                recs, messages = _run_scenario_instance(
                    engine,
                    instance,
                    memory_bytes,
                )
                records.extend(recs)
                for msg in messages:
                    progress.advance(msg)
                LOGGER.info("Completed scenario %s", label)
        else:
            futures: Dict[Any, int] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for idx, instance in enumerate(instance_list):
                    label = f"{instance.scenario}:{instance.variant}"
                    LOGGER.info("Submitting scenario instance %s", label)
                    future = executor.submit(
                        _run_scenario_instance_worker,
                        instance,
                        memory_bytes,
                    )
                    futures[future] = idx

                ordered: Dict[int, list[dict[str, object]]] = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    instance = instance_list[idx]
                    label = f"{instance.scenario}:{instance.variant}"
                    try:
                        recs, messages = future.result()
                    except Exception:
                        progress.close()
                        raise
                    ordered[idx] = recs
                    for msg in messages:
                        progress.advance(msg)
                    LOGGER.info("Completed scenario %s", label)

            for idx in range(len(instance_list)):
                records.extend(ordered.get(idx, []))
    finally:
        progress.close()

    df = pd.DataFrame(records)
    if df.empty or "framework" not in df.columns:
        return df
    try:
        baseline_df = df
        if "scenario" in df.columns:
            w_mask = df["scenario"] == "w_state_oracle"
            if w_mask.any():
                allowed = {
                    Backend.STATEVECTOR.value,
                    Backend.DECISION_DIAGRAM.value,
                    "quasar",
                }
                baseline_df = df[
                    (~w_mask)
                    | df["framework"].isin(allowed)
                ].reset_index(drop=True)
        baseline_best = compute_baseline_best(
            baseline_df,
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
            "patch_distance",
            "boundary_width",
            "gadget_width",
            "stabilizer_rounds",
            "gadget",
            "scheme",
            "chi_target",
            "gadget_size",
            "gadget_layers",
            "dense_gadgets",
            "gadget_spacing",
            "ladder_layers",
            "chain_length",
            "w_state_width",
            "oracle_layers",
            "oracle_rotation_gate_count",
            "oracle_rotation_unique",
            "oracle_rotation_density",
            "oracle_rotation_per_layer",
            "oracle_parameterised_rotations",
            "oracle_entangling_count",
            "oracle_entangling_per_layer",
            "oracle_sparsity",
            "rotation_set",
            "partition_count",
            "partition_total_subsystems",
            "partition_unique_backends",
            "partition_max_multiplicity",
            "partition_mean_multiplicity",
            "partition_backend_breakdown",
            "hierarchy_available",
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
        help=(
            "Named partitioning scenario defined in partitioning_workloads "
            "(e.g. dual_magic_injection)"
        ),
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
        "--workers",
        type=int,
        help=(
            "Maximum number of worker threads to use (default: auto detect)."
        ),
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

    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be a positive integer")

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
            max_workers=args.workers,
        )
        save_results(df, args.output)
    else:
        instances = iter_scenario(args.scenario)
        df = run_scenarios(
            instances,
            args.repetitions,
            memory_bytes=args.memory_bytes,
            max_workers=args.workers,
        )
        save_results(df, args.output)
        write_partitioning_tables(df, args.output)


# Import surface-code protected circuits so the CLI can discover them.
from circuits import surface_corrected_qaoa_circuit  # noqa: E402,F401


if __name__ == "__main__":
    main()

