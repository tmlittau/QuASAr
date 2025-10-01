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


def build_qec_clifford_magic_patch(
    qubits: int = 200, ring_size: int = 20
) -> Circuit:
    """Create a planar QEC patch with Clifford rounds and a T-ring injection.

    The circuit mirrors a pair of syndrome extraction Clifford layers around a
    contiguous ring of ``T`` injections.  The prelude entangles four-qubit
    plaquettes through ``CX`` and ``CZ`` scaffolding, the T-ring injects magic
    resources across ``ring_size`` qubits and the closing layer reuses the
    Clifford template to hand control back to tableau simulation.
    """

    if qubits <= 0:
        return Circuit([])

    active_ring = max(0, min(qubits, ring_size))
    start = max(0, (qubits - active_ring) // 2)
    ring_indices = list(range(start, start + active_ring))

    gates: list[Gate] = []
    stabiliser_span = 4 if qubits >= 4 else max(1, qubits)

    # Clifford syndrome extraction prelude.
    for q in range(qubits):
        gates.append(Gate("H", [q]))
    for anchor in range(0, qubits, stabiliser_span):
        ancilla = anchor
        span = min(stabiliser_span, qubits - anchor)
        gates.append(Gate("S", [ancilla]))
        for offset in range(1, span):
            target = anchor + offset
            gates.append(Gate("CX", [ancilla, target]))
            if target + 1 < anchor + span:
                gates.append(Gate("CZ", [target, target + 1]))

    # Magic ring injection.
    if ring_indices:
        for idx in ring_indices:
            gates.append(Gate("T", [idx]))
        if len(ring_indices) > 1:
            for left, right in zip(ring_indices, ring_indices[1:]):
                gates.append(Gate("CZ", [left, right]))
            gates.append(Gate("CZ", [ring_indices[-1], ring_indices[0]]))

    # Closing Clifford layer returning stabiliser structure.
    for anchor in range(0, qubits, stabiliser_span):
        ancilla = anchor
        span = min(stabiliser_span, qubits - anchor)
        gates.append(Gate("SDG", [ancilla]))
        for offset in range(span - 1, 0, -1):
            target = anchor + offset
            if target < qubits:
                gates.append(Gate("CX", [ancilla, target]))
    for q in range(qubits):
        gates.append(Gate("H", [q]))

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
    """Extract execution diagnostics shared across hybrid magic benchmarks."""

    metrics: dict[str, float] = {}
    record = result.record
    run_time = float(record.get("run_time", 0.0) or 0.0)
    run_memory = float(record.get("run_peak_memory", 0) or 0)
    metrics["run_time_seconds"] = run_time
    metrics["run_peak_memory_bytes"] = run_memory

    baseline: Mapping[str, Any] | None = None
    no_conversion: Mapping[str, Any] | None = None
    if result.baselines is not None:
        baseline = result.baselines.get("statevector")
        no_conversion = result.baselines.get("no_conversion")
    if baseline is not None:
        sv_time = float(baseline.get("run_time", 0.0) or 0.0)
        sv_memory = float(baseline.get("run_peak_memory", 0) or 0)
        metrics["statevector_run_time_seconds"] = sv_time
        metrics["statevector_peak_memory_bytes"] = sv_memory
        if run_time > 0:
            metrics["speedup_vs_statevector"] = sv_time / run_time
        elif sv_time > 0:
            metrics["speedup_vs_statevector"] = float("inf")
    if no_conversion is not None:
        nc_time = float(no_conversion.get("run_time", 0.0) or 0.0)
        nc_memory = float(no_conversion.get("run_peak_memory", 0) or 0)
        metrics["no_conversion_run_time_seconds"] = nc_time
        metrics["no_conversion_peak_memory_bytes"] = nc_memory
        if run_time > 0:
            metrics["speedup_vs_no_conversion"] = nc_time / run_time

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
            key = primitive.lower()
            primitive_counts[key] = primitive_counts.get(key, 0) + 1
    total_conversions = sum(primitive_counts.values())
    for primitive, count in primitive_counts.items():
        metrics[f"conversion_primitive_{primitive}"] = float(count)
        if total_conversions:
            metrics[f"conversion_primitive_fraction_{primitive}"] = count / total_conversions

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

    backend_costs: dict[str, float] = {}
    if partitions:
        total_cost = 0.0
        for part in partitions:
            backend = getattr(part, "backend", None)
            backend_name = (
                getattr(backend, "name", str(backend)) if backend is not None else "unknown"
            )
            cost = getattr(part, "cost", None)
            runtime = float(getattr(cost, "time", 0.0) or 0.0)
            weight = max(1, getattr(part, "multiplicity", 1))
            contribution = runtime * weight
            if contribution <= 0:
                continue
            backend_costs[backend_name] = backend_costs.get(backend_name, 0.0) + contribution
            total_cost += contribution
        if total_cost > 0:
            for backend_name, contribution in backend_costs.items():
                metrics[f"backend_dwell_fraction_{backend_name.lower()}"] = contribution / total_cost

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

register_family(
    BenchmarkCircuitFamily(
        name="qec_clifford_magic_patch",
        display_name="QEC Clifford magic patch",
        description=(
            "Planar QEC patch with Clifford syndrome rounds bracketing a ring"
            " of injected T gates to force Tableau↔SV hand-offs."
        ),
        builder=build_qec_clifford_magic_patch,
        parameter_grid={
            "qubits": (200,),
            "ring_size": (20,),
        },
        metadata={
            "expected_conversions": (
                "Initial Clifford layers dwell on tableau backends before magic",
                " injections promote the ring to dense simulation. A closing",
                " Clifford sweep returns control to stabiliser simulation.",
            ),
            "expected_conversion_path": (
                "tableau → statevector during the T-ring, then back to tableau",
                " once the Clifford cleanup completes.",
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
    "build_qec_clifford_magic_patch",
    "collect_graph_state_metrics",
]

