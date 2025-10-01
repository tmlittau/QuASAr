"""Structured registry for benchmark circuit families.

The module mirrors the documentation style used across the benchmark helpers
so that new workloads follow consistent conventions.  It exposes a lightweight
registry for circuit families that provides:

* A :class:`BenchmarkCircuitFamily` dataclass describing how to synthesise the
  workload and which metadata accompanies it.
* A :class:`BenchmarkResult` container that downstream consumers can populate
  with execution artefacts (timings, subsystem descriptors, baseline runs).
* Helper functions for constructing circuits and summarising QuASAr specific
  diagnostics such as SSD traces and conversion logs.

The registry is intentionally simple—benchmarks import it alongside existing
``benchmarks.bench_utils`` modules which keeps scripting entry points stable
while allowing new circuit families to coexist with the legacy showcase suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from random import Random
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend


@dataclass(frozen=True)
class BenchmarkCircuitFamily:
    """Description of a benchmark circuit family registered with the suite."""

    name: str
    display_name: str
    description: str
    builder: Callable[..., Circuit]
    parameter_grid: Mapping[str, Sequence[int | float | str]]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metric_hooks: tuple[Callable[["BenchmarkResult"], Mapping[str, float]], ...] = ()


@dataclass(frozen=True)
class BenchmarkResult:
    """Container bundling execution artefacts for a benchmark workload."""

    family: BenchmarkCircuitFamily
    parameters: Mapping[str, Any]
    circuit: Circuit
    record: Mapping[str, Any]
    baselines: Mapping[str, Mapping[str, Any]] | None = None


BENCHMARK_FAMILIES: MutableMapping[str, BenchmarkCircuitFamily] = {}
"""Registry of available benchmark circuit families."""


def register_family(family: BenchmarkCircuitFamily) -> None:
    """Add ``family`` to :data:`BENCHMARK_FAMILIES` without overwriting entries."""

    if family.name in BENCHMARK_FAMILIES:
        raise ValueError(f"benchmark family '{family.name}' is already registered")
    BENCHMARK_FAMILIES[family.name] = family


def _grid_shape(width: int) -> tuple[int, int]:
    """Return a near-square ``(rows, cols)`` layout that covers ``width`` qubits."""

    if width <= 0:
        return 0, 0
    rows = max(1, int(math.isqrt(width)))
    cols = math.ceil(width / rows)
    while rows > 1 and (rows - 1) * cols >= width:
        rows -= 1
    return rows, cols


def build_graph_state_magic_injection(
    width: int, num_magic: int, seed: int | None = None
) -> Circuit:
    """Prepare a rectangular graph state with injected ``T`` magic rotations.

    The circuit follows three stages:

    #. Apply Hadamard gates to all qubits and entangle them via ``CZ`` edges to
       realise a rectangular grid graph state.
    #. Draw ``num_magic`` qubit indices (with replacement) from a reproducible
       random number generator and apply ``T`` rotations to those qubits.
    #. Append two Clifford readout layers that rotate qubits into alternating
       measurement bases.  The pattern mirrors existing benchmark circuits
       where Hadamard and phase gates provide deterministic basis changes.
    """

    if width <= 0:
        return Circuit([])

    rows, cols = _grid_shape(width)
    gates: list[Gate] = []

    for q in range(width):
        gates.append(Gate("H", [q]))

    for index in range(width):
        row, col = divmod(index, cols)
        right = index + 1
        if col + 1 < cols and right < width and right // cols == row:
            gates.append(Gate("CZ", [index, right]))
        down = index + cols
        if down < width:
            gates.append(Gate("CZ", [index, down]))

    rng = Random(seed)
    for _ in range(max(0, num_magic)):
        target = rng.randrange(width)
        gates.append(Gate("T", [target]))

    for q in range(width):
        gates.append(Gate("H", [q]))
    for index in range(width):
        row, col = divmod(index, cols)
        if (row + col) % 2 == 0:
            gates.append(Gate("S", [index]))
        else:
            gates.append(Gate("SDG", [index]))

    return Circuit(gates)


def _partition_gate_fractions(partitions: Iterable[Any]) -> Mapping[str, float]:
    """Return backend-labelled gate fractions derived from ``partitions``."""

    totals: dict[str, float] = {}
    gate_counts: dict[str, int] = {}
    total_gates = 0
    for part in partitions:
        backend = getattr(part, "backend", None)
        backend_name = getattr(backend, "name", str(backend)) if backend is not None else "unknown"
        history = getattr(part, "history", ()) or ()
        count = len(tuple(history))
        if count <= 0:
            continue
        gate_counts[backend_name] = gate_counts.get(backend_name, 0) + count
        total_gates += count
    if total_gates == 0:
        return {}
    for backend_name, count in gate_counts.items():
        totals[f"backend_gate_fraction_{backend_name.lower()}"] = count / total_gates
    return totals


def collect_graph_state_metrics(result: BenchmarkResult) -> dict[str, float]:
    """Extract execution diagnostics for the graph-state magic benchmark."""

    metrics: dict[str, float] = {}
    record = result.record
    run_time = float(record.get("run_time", 0.0) or 0.0)
    run_memory = float(record.get("run_peak_memory", 0) or 0)
    metrics["run_time_seconds"] = run_time
    metrics["run_peak_memory_bytes"] = run_memory

    baseline: Mapping[str, Any] | None = None
    if result.baselines is not None:
        baseline = result.baselines.get("statevector")
    if baseline is not None:
        sv_time = float(baseline.get("run_time", 0.0) or 0.0)
        sv_memory = float(baseline.get("run_peak_memory", 0) or 0)
        metrics["statevector_run_time_seconds"] = sv_time
        metrics["statevector_peak_memory_bytes"] = sv_memory
        if run_time > 0:
            metrics["speedup_vs_statevector"] = sv_time / run_time
        elif sv_time > 0:
            metrics["speedup_vs_statevector"] = float("inf")

    ssd = record.get("result")
    partitions = getattr(ssd, "partitions", None)
    if partitions:
        metrics.update(_partition_gate_fractions(partitions))

    conversions = list(getattr(ssd, "conversions", ())) if ssd is not None else []
    metrics["conversion_count"] = float(len(conversions))
    primitive_counts: dict[str, int] = {}
    for conv in conversions:
        primitive = getattr(conv, "primitive", None)
        if primitive:
            primitive_counts[primitive.lower()] = primitive_counts.get(primitive.lower(), 0) + 1
    for primitive, count in primitive_counts.items():
        metrics[f"conversion_primitive_{primitive}"] = float(count)

    trace_entries = list(getattr(ssd, "trace", ())) if ssd is not None else []
    tableau_to_sv = 0
    sv_to_tableau = 0
    applied_switches = 0
    for entry in trace_entries:
        if not getattr(entry, "applied", False):
            continue
        applied_switches += 1
        source = getattr(entry, "from_backend", None)
        target = getattr(entry, "to_backend", None)
        if source == Backend.TABLEAU and target == Backend.STATEVECTOR:
            tableau_to_sv += 1
        elif source == Backend.STATEVECTOR and target == Backend.TABLEAU:
            sv_to_tableau += 1
    metrics["backend_switch_count"] = float(applied_switches)
    if tableau_to_sv:
        metrics["tableau_to_statevector_switches"] = float(tableau_to_sv)
    if sv_to_tableau:
        metrics["statevector_to_tableau_switches"] = float(sv_to_tableau)

    return metrics


register_family(
    BenchmarkCircuitFamily(
        name="graph_state_magic_injection",
        display_name="Graph-state magic injection",
        description=(
            "Rectangular graph states with T-gate injections triggering Tableau"
            " ↔ statevector conversions."
        ),
        builder=build_graph_state_magic_injection,
        parameter_grid={
            "qubits": (64, 128, 256),
            "magic_t": (8, 16, 32),
        },
        metadata={
            "expected_conversions": (
                "Stabiliser fragments evolve under tableau simulation before"
                " injected T gates promote select regions to statevector"
                " execution, producing repeated Tableau↔SV conversions."
            ),
            "notes": (
                "The workload stresses SSD partition logging by combining a"
                " stabiliser backbone with sparse non-Clifford injections and"
                " deterministic Clifford readout layers."
            ),
        },
        metric_hooks=(collect_graph_state_metrics,),
    )
)


__all__ = [
    "BenchmarkCircuitFamily",
    "BenchmarkResult",
    "BENCHMARK_FAMILIES",
    "register_family",
    "build_graph_state_magic_injection",
    "collect_graph_state_metrics",
]

