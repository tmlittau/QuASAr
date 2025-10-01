"""Benchmark the showcase circuits introduced most recently.

The script executes the clustered, layered and classical-control benchmark
circuits on QuASAr and the baseline backends that ship with the project.
For each workload the fastest baseline backend is compared against the
automatically selected QuASAr configuration.  Results are exported as CSV
tables (with optional Markdown mirrors) and publication-ready figures.

The module mirrors the timeout semantics used by
``benchmarks/paper_figures.py`` so that individual runs can be capped when
executing the showcase suite on workstations with limited resources.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if __package__ in {None, ""}:
    if str(PACKAGE_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(PACKAGE_ROOT))
    if str(REPO_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(REPO_ROOT))
    from plot_utils import (  # type: ignore[no-redef]
        backend_labels,
        backend_markers,
        backend_palette,
        compute_baseline_best,
        plot_quasar_vs_baseline_best,
        setup_benchmark_style,
    )
    from runner import BenchmarkRunner  # type: ignore[no-redef]
    import circuits as circuit_lib  # type: ignore[no-redef]
    import stitched_suite as stitched_suites  # type: ignore[no-redef]
    from database import BenchmarkDatabase, BenchmarkRun, open_database  # type: ignore[no-redef]
else:  # pragma: no cover - exercised when imported as a package module
    from .plot_utils import (
        backend_labels,
        backend_markers,
        backend_palette,
        compute_baseline_best,
        plot_quasar_vs_baseline_best,
        setup_benchmark_style,
    )
    from .runner import BenchmarkRunner
    from . import circuits as circuit_lib
    from . import stitched_suite as stitched_suites
    from .database import BenchmarkDatabase, BenchmarkRun, open_database

from quasar import SimulationEngine
from quasar.cost import Backend

try:  # shared utilities for both package and script execution
    from .memory_utils import max_qubits_statevector
    from .progress import ProgressReporter
    from .ssd_metrics import partition_metrics_from_result
    from .threading_utils import resolve_worker_count, thread_engine
except ImportError:  # pragma: no cover - fallback when executed as a script
    from memory_utils import max_qubits_statevector  # type: ignore
    from progress import ProgressReporter  # type: ignore
    from ssd_metrics import partition_metrics_from_result  # type: ignore
    from threading_utils import resolve_worker_count, thread_engine  # type: ignore


LOGGER = logging.getLogger(__name__)


FIGURES_DIR = PACKAGE_ROOT / "figures" / "showcase"
DATABASE_PATH = PACKAGE_ROOT / "results" / "benchmarks.sqlite"


RUN_TIMEOUT_DEFAULT_SECONDS = 1800
"""Maximum duration allowed for a single backend run (adjustable)."""


BASELINE_BACKENDS: tuple[Backend, ...] = (
    Backend.STATEVECTOR,
    Backend.TABLEAU,
    Backend.MPS,
    Backend.DECISION_DIAGRAM,
)


def _result_to_json(result: Any) -> str | None:
    if result is None:
        return None
    try:
        return json.dumps(result)
    except TypeError:
        return json.dumps({"repr": repr(result)})


def _finalise_record(
    record: Mapping[str, Any],
    *,
    spec: circuit_lib.ShowcaseCircuit,
    width: int,
    framework: str,
    backend: str | None,
    mode: str,
) -> dict[str, Any]:
    result = record.get("result")
    normalised = dict(record)
    normalised.pop("result", None)
    normalised.update(partition_metrics_from_result(result))
    normalised["result_json"] = _result_to_json(result)
    backend_name: str | None
    if isinstance(backend, Backend):
        backend_name = backend.name
    else:
        backend_name = str(backend) if backend is not None else None
    normalised.update(
        {
            "circuit": spec.name,
            "qubits": width,
            "framework": framework,
            "backend": backend_name,
            "mode": mode,
        }
    )
    return normalised


def _parse_range_expression(expr: str) -> list[int]:
    """Parse a ``start:end[:step]`` range specification."""

    parts = [int(p) for p in expr.split(":") if p.strip()]
    if not parts:
        raise ValueError("empty range expression")
    if len(parts) == 1:
        return [parts[0]]
    if len(parts) == 2:
        start, stop = parts
        step = 1
    elif len(parts) == 3:
        start, stop, step = parts
        if step <= 0:
            raise ValueError("step must be positive")
    else:
        raise ValueError("range must have the form start:end[:step]")
    if stop < start:
        raise ValueError("end must be greater than or equal to start")
    return list(range(start, stop + 1, step))


def _parse_qubit_overrides(values: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Return user-specified qubit widths for individual circuits."""

    overrides: dict[str, tuple[int, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "qubit override must use the form <circuit>=<range or comma list>"
            )
        name, spec = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("circuit name in qubit override cannot be empty")
        widths: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                widths.extend(_parse_range_expression(part))
            else:
                widths.append(int(part))
        if not widths:
            raise ValueError(f"no qubit widths specified for circuit '{name}'")
        overrides[name] = tuple(sorted(dict.fromkeys(widths)))
    return overrides


def _set_classical_simplification(circuit: object, enabled: bool) -> None:
    """Toggle the classical simplification flag on ``circuit`` when supported."""

    toggle = getattr(circuit, "enable_classical_simplification", None)
    disable = getattr(circuit, "disable_classical_simplification", None)
    if enabled:
        if callable(toggle):
            toggle()
        else:
            setattr(circuit, "use_classical_simplification", True)
    else:
        if callable(disable):
            disable()
        else:
            setattr(circuit, "use_classical_simplification", False)


@dataclass(frozen=True)
class ShowcaseCircuit:
    """Description of a showcase benchmark circuit."""

    name: str
    display_name: str
    constructor: Callable[[int], object]
    default_qubits: tuple[int, ...]
    description: str


SHOWCASE_CIRCUITS: Mapping[str, ShowcaseCircuit] = {
    # Clustered circuits previously reached 40 qubits by default, but the
    # largest width caused workstation runs to stall at the final benchmark
    # step.  Trimming the tail width to 36 keeps the suite representative
    # while ensuring the CLI completes in a reasonable time on modest hosts.
    "clustered_ghz_random": ShowcaseCircuit(
        name="clustered_ghz_random",
        display_name="Clustered GHZ + random",
        constructor=circuit_lib.clustered_ghz_random_circuit,
        default_qubits=(24, 32, 36, 40),
        description="GHZ blocks followed by deep random hybrid layers.",
    ),
    "clustered_w_random": ShowcaseCircuit(
        name="clustered_w_random",
        display_name="Clustered W + random",
        constructor=circuit_lib.clustered_w_random_circuit,
        default_qubits=(24, 32, 36),
        description="W-state clusters followed by random hybrid layers.",
    ),
    "clustered_ghz_qft": ShowcaseCircuit(
        name="clustered_ghz_qft",
        display_name="Clustered GHZ + QFT",
        constructor=circuit_lib.clustered_ghz_qft_circuit,
        default_qubits=(24, 32, 36),
        description="GHZ clusters with a global QFT tail.",
    ),
    "clustered_w_qft": ShowcaseCircuit(
        name="clustered_w_qft",
        display_name="Clustered W + QFT",
        constructor=circuit_lib.clustered_w_qft_circuit,
        default_qubits=(24, 32, 36),
        description="W-state clusters with a global QFT tail.",
    ),
    "clustered_ghz_random_qft": ShowcaseCircuit(
        name="clustered_ghz_random_qft",
        display_name="Clustered GHZ + random + QFT",
        constructor=circuit_lib.clustered_ghz_random_qft_circuit,
        default_qubits=(24, 32, 36),
        description="GHZ clusters, random evolution and a final QFT.",
    ),
    "clustered_ghz_random_globalqft_random": ShowcaseCircuit(
        name="clustered_ghz_random_globalqft_random",
        display_name="Clustered GHZ random-QFT-random",
        constructor=circuit_lib.clustered_ghz_random_globalqft_random_circuit,
        default_qubits=(24, 32, 36),
        description=(
            "GHZ clusters with a random prefix, global QFT interlude and random tail."
        ),
    ),
    "clustered_ghz_diag_globalqft_diag": ShowcaseCircuit(
        name="clustered_ghz_diag_globalqft_diag",
        display_name="Clustered GHZ diag-QFT-diag",
        constructor=circuit_lib.clustered_ghz_diag_globalqft_diag_circuit,
        default_qubits=(24, 32, 36),
        description=(
            "GHZ clusters with diagonal CRZ/CCZ slabs bracketing a global QFT."
        ),
    ),
    "clustered_w_random_xburst_random": ShowcaseCircuit(
        name="clustered_w_random_xburst_random",
        display_name="Clustered W random-X-burst-random",
        constructor=circuit_lib.clustered_w_random_xburst_random_circuit,
        default_qubits=(24, 32, 36),
        description=(
            "W-state clusters with a random prelude, cross-block burst and random tail."
        ),
    ),
    "layered_clifford_delayed_magic": ShowcaseCircuit(
        name="layered_clifford_delayed_magic",
        display_name="Layered Clifford (delayed magic)",
        constructor=circuit_lib.layered_clifford_delayed_magic_circuit,
        default_qubits=(12, 16, 20),
        description="Clifford prefix with late non-Clifford transition.",
    ),
    "layered_clifford_midpoint": ShowcaseCircuit(
        name="layered_clifford_midpoint",
        display_name="Layered Clifford (midpoint)",
        constructor=circuit_lib.layered_clifford_midpoint_circuit,
        default_qubits=(12, 16, 20),
        description="Clifford to non-Clifford switch halfway through.",
    ),
    "layered_clifford_ramp": ShowcaseCircuit(
        name="layered_clifford_ramp",
        display_name="Layered Clifford ramp",
        constructor=circuit_lib.layered_clifford_ramp_circuit,
        default_qubits=(12, 16, 20),
        description="Gradual increase of non-Clifford density.",
    ),
    "classical_controlled": ShowcaseCircuit(
        name="classical_controlled",
        display_name="Classical-controlled",
        constructor=circuit_lib.classical_controlled_circuit,
        default_qubits=(16, 22, 24),
        description="Classical control regions with moderate fan-out.",
    ),
    "dynamic_classical_control": ShowcaseCircuit(
        name="dynamic_classical_control",
        display_name="Dynamic classical control",
        constructor=circuit_lib.dynamic_classical_control_circuit,
        default_qubits=(16, 22, 24),
        description="Classical controls that toggle frequently.",
    ),
    "classical_controlled_fanout": ShowcaseCircuit(
        name="classical_controlled_fanout",
        display_name="Classical control fan-out",
        constructor=circuit_lib.classical_controlled_fanout_circuit,
        default_qubits=(16, 22, 24),
        description="Classical controls with wide fan-out.",
    ),
    "classical_controlled_dd_sandwich": ShowcaseCircuit(
        name="classical_controlled_dd_sandwich",
        display_name="Classical control DD sandwich",
        constructor=circuit_lib.classical_controlled_dd_sandwich_circuit,
        default_qubits=(16, 22, 24),
        description=(
            "Diagonal CRZ/CCZ slab sandwiched between classical-control toggles."
        ),
    ),
}


SHOWCASE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "clustered": (
        "clustered_ghz_random",
        "clustered_w_random",
        "clustered_ghz_qft",
        "clustered_w_qft",
        "clustered_ghz_random_qft",
        "clustered_ghz_random_globalqft_random",
        "clustered_ghz_diag_globalqft_diag",
        "clustered_w_random_xburst_random",
    ),
    "layered": (
        "layered_clifford_delayed_magic",
        "layered_clifford_midpoint",
        "layered_clifford_ramp",
    ),
    "classical_control": (
        "classical_controlled",
        "dynamic_classical_control",
        "classical_controlled_fanout",
        "classical_controlled_dd_sandwich",
    ),
}


def available_suite_names() -> tuple[str, ...]:
    """Return the tuple of registered showcase suite names."""

    return stitched_suites.available_suites()


class ExtendAction(argparse.Action):
    """`argparse` action that accumulates values across multiple flags."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        current = getattr(namespace, self.dest, None)
        if current is None:
            current = []
        if isinstance(values, str):
            current.append(values)
        else:
            current.extend(values)
        setattr(namespace, self.dest, current)


def _list_available_groups() -> str:
    """Return a formatted description of available circuit groups."""

    lines = ["Available groups:"]
    for name in sorted(SHOWCASE_GROUPS):
        circuits = ", ".join(SHOWCASE_GROUPS[name])
        lines.append(f"  - {name}: {circuits}")
    return "\n".join(lines)


def _resolve_selected_circuits(
    *,
    explicit: Sequence[str] | None,
    groups: Sequence[str] | None,
    catalog: Mapping[str, ShowcaseCircuit] | None = None,
) -> list[str]:
    """Return the ordered list of circuits selected for this run."""

    available = SHOWCASE_CIRCUITS if catalog is None else catalog
    order = OrderedDict((name, None) for name in available)
    selected = OrderedDict()

    def _add(name: str) -> None:
        if name not in available:
            raise SystemExit(f"unknown circuit name '{name}'")
        if name not in selected:
            selected[name] = None

    if explicit:
        for name in explicit:
            _add(name)

    if groups:
        for group in groups:
            if group not in SHOWCASE_GROUPS:
                raise SystemExit(
                    f"unknown circuit group '{group}'.\n{_list_available_groups()}"
                )
            for name in SHOWCASE_GROUPS[group]:
                _add(name)

    if not selected:
        return list(order)

    ordered_selection = [name for name in order if name in selected]
    missing = [name for name in selected if name not in order]
    return ordered_selection + missing


def _merge_results(
    path: Path,
    new_data: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    sort_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Merge ``new_data`` with existing ``path`` contents, keeping latest rows."""

    frames: list[pd.DataFrame] = []
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to read existing results from %s: %s", path, exc)
        else:
            if not existing.empty:
                frames.append(existing)

    frames.append(new_data)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    subset = [col for col in key_columns if col in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    if sort_columns:
        sortable = [col for col in sort_columns if col in combined.columns]
        if sortable:
            combined = combined.sort_values(sortable).reset_index(drop=True)
    return combined


def _configure_logging(verbosity: int) -> None:
    """Initialise logging for CLI usage."""

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _build_circuit(
    spec: ShowcaseCircuit, width: int, *, classical_simplification: bool
) -> object:
    circuit = spec.constructor(width)
    _set_classical_simplification(circuit, classical_simplification)
    return circuit


def _baseline_support_status(
    backend: Backend,
    *,
    width: int,
    circuit: object | None,
    memory_bytes: int | None,
) -> tuple[bool, str | None]:
    """Return whether ``backend`` can execute ``circuit`` and a skip reason."""

    if backend == Backend.STATEVECTOR:
        limit = max_qubits_statevector(memory_bytes)
        if width > limit:
            return (
                False,
                f"circuit width {width} exceeds statevector limit of {limit} qubits",
            )

    if backend == Backend.TABLEAU and circuit is not None:
        gates = getattr(circuit, "gates", ())
        forbidden = {"CCX", "CCZ", "MCX", "CSWAP"}
        for gate in gates:
            name = getattr(gate, "gate", "").upper()
            if name in forbidden:
                return False, f"{name} gate is unsupported by the tableau backend"
        try:
            is_clifford = circuit_lib.is_clifford(circuit)
        except Exception:  # pragma: no cover - defensive
            is_clifford = False
        if not is_clifford:
            return False, "non-Clifford gates are unsupported by the tableau backend"

    return True, None


def _run_backend_suite_for_width(
    engine: SimulationEngine,
    spec: ShowcaseCircuit,
    width: int,
    *,
    repetitions: int,
    run_timeout: float | None,
    memory_bytes: int | None,
    classical_simplification: bool,
    baseline_backends: Iterable[Backend],
    quasar_quick: bool,
    step_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    messages: list[str] = []

    LOGGER.info("Starting benchmarks for %s at %s qubits", spec.name, width)

    built_circuit: object | None = None

    def _ensure_circuit() -> object:
        nonlocal built_circuit
        if built_circuit is None:
            built_circuit = _build_circuit(
                spec, width, classical_simplification=classical_simplification
            )
        return built_circuit

    for backend in baseline_backends:
        status_msg = f"{backend.name}@{width}"
        supported, reason = _baseline_support_status(
            backend,
            width=width,
            circuit=None,
            memory_bytes=memory_bytes,
        )
        if not supported:
            LOGGER.info(
                "Skipping backend %s for %s qubits=%s: %s",
                backend.name,
                spec.name,
                width,
                reason,
            )
            record = _finalise_record(
                {
                    "unsupported": True,
                    "failed": False,
                    "comment": reason,
                    "repetitions": 0,
                    "result": None,
                },
                spec=spec,
                width=width,
                framework=backend.name,
                backend=backend.name,
                mode="forced",
            )
            records.append(record)
            messages.append(f"{status_msg} skipped: {reason}")
            continue

        circuit = _ensure_circuit()
        supported, reason = _baseline_support_status(
            backend,
            width=width,
            circuit=circuit,
            memory_bytes=memory_bytes,
        )
        if not supported:
            LOGGER.info(
                "Skipping backend %s for %s qubits=%s: %s",
                backend.name,
                spec.name,
                width,
                reason,
            )
            record = _finalise_record(
                {
                    "unsupported": True,
                    "failed": False,
                    "comment": reason,
                    "repetitions": 0,
                    "result": None,
                },
                spec=spec,
                width=width,
                framework=backend.name,
                backend=backend.name,
                mode="forced",
            )
            records.append(record)
            messages.append(f"{status_msg} skipped: {reason}")
            continue

        runner = BenchmarkRunner()
        if step_callback is not None:
            step_callback(status_msg)
        LOGGER.info(
            "Running baseline backend %s for %s qubits=%s",
            backend.name,
            spec.name,
            width,
        )
        try:
            rec = runner.run_quasar_multiple(
                circuit,
                engine,
                backend=backend,
                repetitions=repetitions,
                quick=quasar_quick,
                memory_bytes=memory_bytes,
                run_timeout=run_timeout,
            )
        except Exception as exc:  # pragma: no cover - backend implementation detail
            LOGGER.warning(
                "Backend %s failed for %s qubits=%s: %s",
                backend.name,
                spec.name,
                width,
                exc,
            )
            record = _finalise_record(
                {
                    "unsupported": True,
                    "failed": True,
                    "error": str(exc),
                    "repetitions": 0,
                },
                spec=spec,
                width=width,
                framework=backend.name,
                backend=backend.name,
                mode="forced",
            )
            records.append(record)
            messages.append(f"{status_msg} failed: {exc}")
            continue

        record = _finalise_record(
            rec,
            spec=spec,
            width=width,
            framework=backend.name,
            backend=backend.name,
            mode="forced",
        )
        records.append(record)
        messages.append(status_msg)

    circuit = _ensure_circuit()
    runner = BenchmarkRunner()
    quasar_status = f"quasar@{width}"
    if step_callback is not None:
        step_callback(quasar_status)
    LOGGER.info("Running QuASAr for %s qubits=%s", spec.name, width)
    try:
        rec = runner.run_quasar_multiple(
            circuit,
            engine,
            repetitions=repetitions,
            quick=quasar_quick,
            memory_bytes=memory_bytes,
            run_timeout=run_timeout,
        )
    except Exception as exc:  # pragma: no cover - scheduler limitations
        LOGGER.warning(
            "QuASAr failed for %s qubits=%s: %s", spec.name, width, exc
        )
        record = _finalise_record(
            {
                "unsupported": True,
                "failed": True,
                "error": str(exc),
                "repetitions": 0,
            },
            spec=spec,
            width=width,
            framework="quasar",
            backend=None,
            mode="auto",
        )
        records.append(record)
        messages.append(f"{quasar_status} failed: {exc}")
    else:
        backend_choice = rec.get("backend")
        if isinstance(backend_choice, Backend):
            rec["backend"] = backend_choice.name
        record = _finalise_record(
            rec,
            spec=spec,
            width=width,
            framework="quasar",
            backend=rec.get("backend"),
            mode="auto",
        )
        records.append(record)
        messages.append(quasar_status)

    LOGGER.info("Completed benchmarks for %s qubits=%s", spec.name, width)
    return records, messages


def _run_backend_suite_for_width_worker(
    spec: ShowcaseCircuit,
    width: int,
    *,
    repetitions: int,
    run_timeout: float | None,
    memory_bytes: int | None,
    classical_simplification: bool,
    baseline_backends: Iterable[Backend],
    quasar_quick: bool,
    step_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    engine = thread_engine()
    return _run_backend_suite_for_width(
        engine,
        spec,
        width,
        repetitions=repetitions,
        run_timeout=run_timeout,
        memory_bytes=memory_bytes,
        classical_simplification=classical_simplification,
        baseline_backends=baseline_backends,
        quasar_quick=quasar_quick,
        step_callback=step_callback,
    )


def _run_backend_suite(
    spec: ShowcaseCircuit,
    widths: Iterable[int],
    *,
    repetitions: int,
    run_timeout: float | None,
    memory_bytes: int | None,
    classical_simplification: bool,
    max_workers: int | None = None,
    include_baselines: bool = True,
    baseline_backends: Iterable[Backend] | None = None,
    quasar_quick: bool = False,
    database: BenchmarkDatabase | None = None,
    run: BenchmarkRun | None = None,
) -> pd.DataFrame:
    """Execute the benchmark for ``spec`` across the provided widths."""

    width_list = list(widths)
    if not width_list:
        return pd.DataFrame()

    baselines: tuple[Backend, ...]
    if not include_baselines:
        baselines = ()
    elif baseline_backends is not None:
        baselines = tuple(baseline_backends)
    else:
        baselines = tuple(BASELINE_BACKENDS)

    total_steps = len(width_list) * (len(baselines) + 1)
    progress = ProgressReporter(total_steps, prefix=f"{spec.name} benchmark")

    worker_count = resolve_worker_count(max_workers, len(width_list))
    LOGGER.info("Using %d worker thread(s) for showcase circuit %s", worker_count, spec.name)

    ordered: dict[int, list[dict[str, object]]] = {}
    benchmark_run = run
    if database is not None and benchmark_run is None:
        benchmark_run = database.start_run(
            description=f"showcase:{spec.name}",
            parameters={
                "circuit": spec.name,
                "repetitions": repetitions,
                "run_timeout": run_timeout,
                "memory_bytes": memory_bytes,
                "classical_simplification": classical_simplification,
                "include_baselines": include_baselines,
                "quasar_quick": quasar_quick,
            },
        )

    try:
        if worker_count <= 1:
            engine = SimulationEngine()
            for index, width in enumerate(width_list):
                benchmark_id: int | None = None
                if database is not None and benchmark_run is not None:
                    benchmark_id = database.create_benchmark(
                        benchmark_run,
                        circuit_id=spec.name,
                        circuit_display_name=spec.display_name,
                        repetitions=repetitions,
                        qubits=width,
                        run_timeout=run_timeout,
                        memory_bytes=memory_bytes,
                        classical_simplification=classical_simplification,
                        include_baselines=include_baselines,
                        quick=quasar_quick,
                        baseline_backends=[backend.name for backend in baselines],
                        workers=worker_count,
                        metadata={
                            "description": spec.description,
                        },
                    )
                recs, messages = _run_backend_suite_for_width(
                    engine,
                    spec,
                    width,
                    repetitions=repetitions,
                    run_timeout=run_timeout,
                    memory_bytes=memory_bytes,
                    classical_simplification=classical_simplification,
                    baseline_backends=baselines,
                    quasar_quick=quasar_quick,
                    step_callback=progress.announce,
                )
                ordered[index] = recs
                if database is not None and benchmark_id is not None:
                    for record in recs:
                        database.insert_simulation_run(benchmark_id, record, qubits=width)
                for message in messages:
                    progress.advance(message)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures: dict[object, int] = {}
                benchmark_ids: dict[int, int | None] = {}
                for index, width in enumerate(width_list):
                    benchmark_id: int | None = None
                    if database is not None and benchmark_run is not None:
                        benchmark_id = database.create_benchmark(
                            benchmark_run,
                            circuit_id=spec.name,
                            circuit_display_name=spec.display_name,
                            repetitions=repetitions,
                            qubits=width,
                            run_timeout=run_timeout,
                            memory_bytes=memory_bytes,
                            classical_simplification=classical_simplification,
                            include_baselines=include_baselines,
                            quick=quasar_quick,
                            baseline_backends=[backend.name for backend in baselines],
                            workers=worker_count,
                            metadata={
                                "description": spec.description,
                            },
                        )
                    benchmark_ids[index] = benchmark_id
                    future = executor.submit(
                        _run_backend_suite_for_width_worker,
                        spec,
                        width,
                        repetitions=repetitions,
                        run_timeout=run_timeout,
                        memory_bytes=memory_bytes,
                        classical_simplification=classical_simplification,
                        baseline_backends=baselines,
                        quasar_quick=quasar_quick,
                    )
                    futures[future] = index

                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        recs, messages = future.result()
                    except Exception:
                        progress.close()
                        raise
                    ordered[index] = recs
                    benchmark_id = benchmark_ids.get(index)
                    if database is not None and benchmark_id is not None:
                        for record in recs:
                            database.insert_simulation_run(benchmark_id, record, qubits=width)
                    for message in messages:
                        progress.advance(message)
    finally:
        progress.close()

    records: list[dict[str, object]] = []
    for index in range(len(width_list)):
        records.extend(ordered.get(index, []))
    return pd.DataFrame(records)

def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Failed to create Markdown table %s: %s", path, exc)
        return
    path.write_text(text + "\n", encoding="utf-8")


def _export_plot(
    df: pd.DataFrame,
    spec: ShowcaseCircuit,
    *,
    figures_dir: Path,
    metric: str = "run_time_mean",
) -> tuple[pd.DataFrame | None, Path | None]:
    if df.empty:
        return None, None

    setup_benchmark_style(palette=backend_palette(["baseline_best", "quasar"]))
    palette = backend_palette(["baseline_best", "quasar"])
    markers = backend_markers(["baseline_best", "quasar"])
    labels = backend_labels(["baseline_best", "quasar"], abbreviated=False)

    ax, speedups = plot_quasar_vs_baseline_best(
        df,
        metric=metric,
        annotate_backend=True,
        return_table=True,
        palette=palette,
        markers=markers,
    )
    ax.set_title(f"{spec.display_name} — {metric.replace('_', ' ')}")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Runtime (s)" if "time" in metric else metric)
    handles, labels_list = ax.get_legend_handles_labels()
    if handles and labels_list:
        ax.legend(handles, [labels.get(label, label) for label in labels_list])

    fig = ax.get_figure()
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{spec.name}_{metric}.png"
    pdf_path = figures_dir / f"{spec.name}_{metric}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.tight_layout()
    plt.close(fig)

    return speedups, png_path


def run_showcase_benchmarks(args: argparse.Namespace) -> None:
    suite_name = getattr(args, "suite", None)
    suite_specs: tuple[stitched_suites.StitchedCircuitSpec, ...] = ()
    suite_overrides: dict[str, tuple[int, ...]] = {}
    if suite_name:
        suite_specs = stitched_suites.resolve_suite(suite_name)
        catalog: Mapping[str, ShowcaseCircuit] = OrderedDict(
            (
                spec.name,
                ShowcaseCircuit(
                    name=spec.name,
                    display_name=spec.display_name,
                    constructor=spec.factory,
                    default_qubits=spec.widths,
                    description=spec.description,
                ),
            )
            for spec in suite_specs
        )
        suite_overrides = {spec.name: spec.widths for spec in suite_specs}
    else:
        catalog = SHOWCASE_CIRCUITS

    selected_names = _resolve_selected_circuits(
        explicit=args.circuit_names,
        groups=args.groups,
        catalog=catalog,
    )
    if suite_name:
        LOGGER.info(
            "Running showcase suite '%s' with circuits: %s",
            suite_name,
            ", ".join(selected_names),
        )

    qubit_overrides: dict[str, tuple[int, ...]] = dict(suite_overrides)
    if args.qubits:
        user_overrides = _parse_qubit_overrides(args.qubits)
        for name in user_overrides:
            if name not in catalog:
                raise SystemExit(f"unknown circuit name '{name}'")
        qubit_overrides.update(user_overrides)

    run_timeout = None if args.run_timeout <= 0 else args.run_timeout
    memory_bytes = args.memory_bytes if args.memory_bytes and args.memory_bytes > 0 else None
    classical_simplification = args.enable_classical_simplification

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    database_path = Path(getattr(args, "database", DATABASE_PATH))
    LOGGER.info("Recording benchmark results in %s", database_path)

    with open_database(database_path) as database:
        run = database.start_run(
            description="showcase_cli",
            parameters={
                "circuits": selected_names,
                "repetitions": args.repetitions,
                "workers": args.workers,
                "run_timeout": run_timeout,
                "memory_bytes": memory_bytes,
                "classical_simplification": classical_simplification,
                "metric": args.metric,
                "suite": suite_name,
            },
        )

        for name in selected_names:
            spec = catalog[name]
            widths = qubit_overrides.get(name, spec.default_qubits)
            LOGGER.info("Benchmarking %s across widths: %s", name, widths)

            raw_df = _run_backend_suite(
                spec,
                widths,
                repetitions=args.repetitions,
                run_timeout=run_timeout,
                memory_bytes=memory_bytes,
                classical_simplification=classical_simplification,
                max_workers=args.workers,
                database=database,
                run=run,
            )

            if raw_df.empty:
                LOGGER.warning("No results recorded for %s; skipping summary", name)
                continue

            try:
                baseline_best = compute_baseline_best(
                    raw_df,
                    metrics=("run_time_mean", "total_time_mean", "run_peak_memory_mean"),
                )
            except ValueError:
                LOGGER.warning("No feasible baseline measurements for %s", name)
                baseline_best = pd.DataFrame()

            quasar_df = raw_df[raw_df["framework"] == "quasar"].copy()
            if not quasar_df.empty:
                quasar_df["framework"] = "quasar"
            summary_frames = [frame for frame in (baseline_best, quasar_df) if not frame.empty]
            if not summary_frames:
                LOGGER.warning("Skipping summary export for %s due to missing data", name)
                continue

            summary_df = pd.concat(summary_frames, ignore_index=True)
            summary_df["circuit"] = spec.name
            summary_df["display_name"] = spec.display_name

            speedups, figure_path = _export_plot(
                summary_df,
                spec,
                figures_dir=FIGURES_DIR,
                metric=args.metric,
            )
            if speedups is not None and not speedups.empty:
                speedups["circuit"] = spec.name
            if figure_path is not None:
                LOGGER.info("Saved figure for %s to %s", name, figure_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the showcase circuits on QuASAr and baseline backends.",
    )
    parser.add_argument(
        "--circuit",
        "--circuits",
        dest="circuit_names",
        action=ExtendAction,
        nargs="+",
        metavar="NAME",
        default=None,
        help="Benchmark the specified circuit(s). Repeat the flag to add more.",
    )
    parser.add_argument(
        "--group",
        dest="groups",
        action=ExtendAction,
        nargs="+",
        choices=sorted(SHOWCASE_GROUPS),
        metavar="GROUP",
        default=None,
        help="Run all circuits belonging to the named group(s).",
    )
    suites = stitched_suites.available_suites()
    if suites:
        parser.add_argument(
            "--suite",
            choices=sorted(suites),
            metavar="SUITE",
            default=None,
            help="Run a preconfigured showcase suite (e.g. stitched-big).",
        )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available circuit groups and exit.",
    )
    parser.add_argument(
        "--qubits",
        action="append",
        metavar="NAME=RANGE",
        help="Override qubit widths for a circuit using start:end[:step] or comma lists.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=RUN_TIMEOUT_DEFAULT_SECONDS,
        help=(
            "Per-run timeout in seconds (default: %(default)s; set <= 0 to disable)."
        ),
    )
    parser.add_argument(
        "--memory-bytes",
        type=int,
        default=None,
        help="Optional memory cap for dense statevector backends.",
    )
    parser.add_argument(
        "--metric",
        default="run_time_mean",
        help="Metric to plot on the figures (default: run_time_mean).",
    )
    parser.add_argument(
        "--enable-classical-simplification",
        action="store_true",
        help="Enable classical control simplification in the generated circuits.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for circuit execution (default: auto).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Path to the SQLite database storing benchmark results (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use twice for debug output).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "list_groups", False):
        print(_list_available_groups())
        return
    _configure_logging(args.verbose)
    run_showcase_benchmarks(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
