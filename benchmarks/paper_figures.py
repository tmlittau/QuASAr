"""Generate reproducible benchmark figures for the QuASAr paper.

Run the module as a script with ``--verbose`` to display progress logs while
figures and result tables are generated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

try:  # Optional dependency used for memory discovery when available
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if __package__ in {None, ""}:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from plot_utils import (  # type: ignore[no-redef]
        backend_labels,
        backend_palette,
        plot_backend_timeseries,
        plot_heatmap,
        plot_quasar_vs_baseline_best,
        plot_speedup_bars,
        setup_benchmark_style,
    )
    from runner import BenchmarkRunner  # type: ignore[no-redef]
    import circuits as circuit_lib  # type: ignore[no-redef]
    from parallel_circuits import many_ghz_subsystems  # type: ignore[no-redef]
    from memory_utils import (  # type: ignore[no-redef]
        DEFAULT_MEMORY_BYTES as DEFAULT_STATEVECTOR_MEMORY_BYTES,
        ENV_VAR as STATEVECTOR_MEMORY_ENV_VAR,
        max_qubits_statevector,
    )
else:  # pragma: no cover - exercised via runtime execution
    from .plot_utils import (
        backend_labels,
        backend_palette,
        plot_backend_timeseries,
        plot_heatmap,
        plot_quasar_vs_baseline_best,
        plot_speedup_bars,
        setup_benchmark_style,
    )
    from .runner import BenchmarkRunner
    from . import circuits as circuit_lib
    from .parallel_circuits import many_ghz_subsystems
    from .memory_utils import (
        DEFAULT_MEMORY_BYTES as DEFAULT_STATEVECTOR_MEMORY_BYTES,
        ENV_VAR as STATEVECTOR_MEMORY_ENV_VAR,
        max_qubits_statevector,
    )

from quasar import SimulationEngine
from quasar.cost import Backend
from quasar.method_selector import NoFeasibleBackendError


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)


STATEVECTOR_MEMORY_HEADROOM_FRACTION = 0.25
"""Portion of detected memory made available to dense statevectors."""


def _statevector_memory_budget_bytes() -> int:
    """Return the configured memory budget for dense statevectors."""

    env_value = os.getenv(STATEVECTOR_MEMORY_ENV_VAR)
    if env_value is not None:
        try:
            budget = int(env_value)
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid %s override: %r",
                STATEVECTOR_MEMORY_ENV_VAR,
                env_value,
            )
        else:
            if budget < 0:
                LOGGER.warning(
                    "Ignoring negative %s override: %r",
                    STATEVECTOR_MEMORY_ENV_VAR,
                    env_value,
                )
            else:
                return budget

    if psutil is not None:
        try:
            available = psutil.virtual_memory().available
        except Exception as exc:  # pragma: no cover - psutil failures are rare
            LOGGER.debug("Failed to query psutil for memory availability: %s", exc)
        else:
            if available > 0:
                headroom = int(available * STATEVECTOR_MEMORY_HEADROOM_FRACTION)
                return headroom if headroom > 0 else int(available)

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        page_size = -1
        phys_pages = -1
    if page_size > 0 and phys_pages > 0:
        total = int(page_size) * int(phys_pages)
        if total > 0:
            headroom = int(total * STATEVECTOR_MEMORY_HEADROOM_FRACTION)
            return headroom if headroom > 0 else total

    return DEFAULT_STATEVECTOR_MEMORY_BYTES


STATEVECTOR_MEMORY_BYTES = _statevector_memory_budget_bytes()
"""Memory budget used when forcing statevector backends."""

STATEVECTOR_SAFE_MEMORY_BYTES = max(0, STATEVECTOR_MEMORY_BYTES - 1)
"""Budget tightened by one byte to ensure runs stay below the hard limit."""

STATEVECTOR_MAX_QUBITS = max_qubits_statevector(STATEVECTOR_SAFE_MEMORY_BYTES)
"""Largest supported dense statevector width under the configured budget."""


RUN_TIMEOUT_DEFAULT_SECONDS = 1800
"""Maximum duration allowed for a single backend run (adjustable)."""


def _filter_qubits(qubits: Sequence[int], *, name: str) -> tuple[int, ...]:
    """Return qubit widths that fit inside the statevector budget."""

    allowed = tuple(q for q in qubits if q <= STATEVECTOR_MAX_QUBITS)
    if not allowed:
        LOGGER.warning(
            "Skipping circuit family '%s'; all widths exceed %s qubits",
            name,
            STATEVECTOR_MAX_QUBITS,
        )
    elif len(allowed) != len(qubits):
        LOGGER.info(
            "Circuit family '%s' trimmed to qubits=%s (limit=%s)",
            name,
            allowed,
            STATEVECTOR_MAX_QUBITS,
        )
    return allowed


def _configure_logging(verbosity: int) -> None:
    """Initialise logging for CLI usage."""

    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _log_written(path: Path) -> None:
    """Emit a user-friendly message when ``path`` is written."""

    LOGGER.info("Wrote %s", path)


@dataclass
class CircuitSpec:
    name: str
    builder: callable
    qubits: Sequence[int]
    kwargs: dict | None = None


BACKENDS: Sequence[Backend] = (
    Backend.STATEVECTOR,
    Backend.TABLEAU,
    Backend.MPS,
    Backend.DECISION_DIAGRAM,
)


def _ghz_ladder_circuit(n_qubits: int, *, group_size: int = 4):
    """Return disjoint GHZ ladders that sum to ``n_qubits``."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if n_qubits % group_size != 0:
        raise ValueError(
            "n_qubits must be divisible by group_size for ghz ladder construction"
        )
    num_groups = n_qubits // group_size
    return many_ghz_subsystems(num_groups=num_groups, group_size=group_size)


def _random_clifford_t_circuit(
    n_qubits: int, *, depth_multiplier: int = 3, base_seed: int = 97
):
    """Return a reproducible Clifford+T hybrid circuit."""

    if depth_multiplier <= 0:
        raise ValueError("depth_multiplier must be positive")
    depth = depth_multiplier * n_qubits
    seed = base_seed + n_qubits
    return circuit_lib.random_hybrid_circuit(n_qubits, depth=depth, seed=seed)


def _large_grover_circuit(n_qubits: int, *, iterations: int = 2):
    """Return a Grover search circuit scaled to ``n_qubits``."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return circuit_lib.grover_circuit(n_qubits, n_iterations=iterations)


def _supports_backend(circuit: object, backend: Backend) -> bool:
    """Return ``True`` when ``circuit`` can run on ``backend``."""

    if backend == Backend.TABLEAU:
        if circuit is None:
            return False
        try:
            return circuit_lib.is_clifford(circuit)
        except AttributeError:  # pragma: no cover - defensive
            return False
    return True


_BASE_CIRCUITS: Sequence[CircuitSpec] = (
    CircuitSpec("qft", circuit_lib.qft_circuit, (3, 4)),
    CircuitSpec("grover", circuit_lib.grover_circuit, (3, 4), {"n_iterations": 1}),
    CircuitSpec(
        "ghz_ladder",
        lambda n, *, group_size=4: _ghz_ladder_circuit(n, group_size=group_size),
        (20, 24, 28, 32),
        {"group_size": 4},
    ),
    CircuitSpec(
        "random_clifford_t",
        lambda n, *, depth_multiplier=3, seed=97: _random_clifford_t_circuit(
            n, depth_multiplier=depth_multiplier, base_seed=seed
        ),
        (20, 24, 28, 32),
        {"depth_multiplier": 3, "seed": 97},
    ),
    CircuitSpec(
        "grover_large",
        lambda n, *, iterations=2: _large_grover_circuit(n, iterations=iterations),
        (20, 24, 28, 32),
        {"iterations": 2},
    ),
)


_FILTERED_CIRCUITS: list[CircuitSpec] = []
for spec in _BASE_CIRCUITS:
    qubits = _filter_qubits(spec.qubits, name=spec.name)
    if not qubits:
        continue
    _FILTERED_CIRCUITS.append(
        CircuitSpec(spec.name, spec.builder, qubits, spec.kwargs)
    )

CIRCUITS: Sequence[CircuitSpec] = tuple(_FILTERED_CIRCUITS)


def _build_circuit(spec: CircuitSpec, n_qubits: int, *, use_classical_simplification: bool) -> object | None:
    try:
        circuit = spec.builder(n_qubits, **(spec.kwargs or {}))
    except TypeError:
        # Builders used in notebooks rely on the ``use_classical_simplification``
        # attribute rather than accepting a keyword argument.  Align the circuit
        # behaviour manually when the call signature is rigid.
        circuit = spec.builder(n_qubits)
    enable = getattr(circuit, "enable_classical_simplification", None)
    disable = getattr(circuit, "disable_classical_simplification", None)
    if use_classical_simplification:
        if not getattr(circuit, "use_classical_simplification", False):
            if callable(enable):
                enable()
            else:
                circuit.use_classical_simplification = True
    else:
        # Forced runs should mimic a monolithic simulator, so opt into flat SSDs.
        if hasattr(circuit, "ssd_mode"):
            circuit.ssd_mode = "flat"
            if hasattr(circuit, "ssd"):
                from quasar.ssd import build_flat_ssd

                circuit.ssd = build_flat_ssd(circuit)
        if getattr(circuit, "use_classical_simplification", True):
            if callable(disable):
                disable()
            else:
                circuit.use_classical_simplification = False
    return circuit


def _circuit_qubit_width(circuit: object | None) -> int | None:
    """Return the number of qubits spanned by ``circuit`` when known."""

    if circuit is None:
        return None
    width = getattr(circuit, "num_qubits", None)
    if width is None:
        return None
    try:
        return int(width)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _statevector_memory_estimate(circuit: object | None) -> int | None:
    """Return the planner's dense memory estimate for ``circuit`` when known."""

    if circuit is None:
        return None
    estimates = getattr(circuit, "cost_estimates", None)
    if not isinstance(estimates, Mapping):
        return None
    cost = estimates.get("statevector")
    memory = getattr(cost, "memory", None)
    if memory is None:
        return None
    try:
        return int(memory)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _mps_skip_reason(
    circuit: object | None,
    requested_qubits: int,
    *,
    forced_width: int | None,
) -> str | None:
    """Return a human-readable reason for skipping MPS runs when applicable."""

    if forced_width is None or requested_qubits <= 0:
        return None
    if forced_width <= requested_qubits:
        return None

    extra = forced_width - requested_qubits
    ancilla_threshold = max(4, requested_qubits // 2)
    exceeds_limit = forced_width > STATEVECTOR_MAX_QUBITS
    ancilla_heavy = extra >= ancilla_threshold
    if not (ancilla_heavy or exceeds_limit):
        return None

    message_parts = [
        (
            "ancilla expansion increases width to "
            f"{forced_width} qubits (requested {requested_qubits})"
        )
    ]
    if exceeds_limit:
        message_parts.append(
            f"exceeds statevector limit of {STATEVECTOR_MAX_QUBITS}"
        )

    memory_estimate = _statevector_memory_estimate(circuit)
    if (
        memory_estimate is not None
        and memory_estimate > STATEVECTOR_SAFE_MEMORY_BYTES
    ):
        message_parts.append(
            "est. dense memory "
            f"{memory_estimate:,} B > budget {STATEVECTOR_SAFE_MEMORY_BYTES:,} B"
        )

    return "; ".join(message_parts)


def _automatic_failure_reason(
    circuit: object | None,
    requested_qubits: int,
    *,
    actual_qubits: int | None,
    default: str,
) -> str:
    """Return a human-readable explanation for automatic run failures."""

    reason = _mps_skip_reason(
        circuit,
        requested_qubits,
        forced_width=actual_qubits,
    )
    if reason:
        return reason

    details: list[str] = []
    if (
        actual_qubits is not None
        and requested_qubits > 0
        and actual_qubits > requested_qubits
    ):
        details.append(
            "circuit expands to "
            f"{actual_qubits} qubits (requested {requested_qubits})"
        )
    if (
        actual_qubits is not None
        and actual_qubits > STATEVECTOR_MAX_QUBITS
    ):
        details.append(
            f"exceeds statevector limit of {STATEVECTOR_MAX_QUBITS}"
        )

    memory_estimate = _statevector_memory_estimate(circuit)
    if (
        memory_estimate is not None
        and memory_estimate > STATEVECTOR_SAFE_MEMORY_BYTES
    ):
        details.append(
            "est. dense memory "
            f"{memory_estimate:,} B > budget {STATEVECTOR_SAFE_MEMORY_BYTES:,} B"
        )

    if not details:
        details.append(default)

    return "; ".join(details)


def collect_backend_data(
    specs: Iterable[CircuitSpec],
    backends: Sequence[Backend],
    *,
    repetitions: int = 3,
    run_timeout: float | None = RUN_TIMEOUT_DEFAULT_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return forced and automatic scheduler results for ``specs``."""

    spec_list = list(specs)
    effective_timeout = run_timeout if run_timeout and run_timeout > 0 else None
    LOGGER.info(
        "Collecting backend data for %d circuit family(ies)", len(spec_list)
    )
    if not spec_list:
        LOGGER.warning("No circuit specifications provided; skipping collection")
        return pd.DataFrame(), pd.DataFrame()

    engine = SimulationEngine()
    forced_records: list[dict[str, object]] = []
    auto_records: list[dict[str, object]] = []

    for spec in spec_list:
        LOGGER.info("Processing circuit family '%s'", spec.name)
        for n in spec.qubits:
            LOGGER.info("Preparing circuits for %s with %s qubits", spec.name, n)
            circuit_forced = _build_circuit(spec, n, use_classical_simplification=False)
            circuit_auto = _build_circuit(spec, n, use_classical_simplification=True)
            if circuit_forced is None or circuit_auto is None:
                LOGGER.warning(
                    "Skipping circuit %s with %s qubits because construction failed",
                    spec.name,
                    n,
                )
                continue

            forced_width = _circuit_qubit_width(circuit_forced)
            auto_width = _circuit_qubit_width(circuit_auto)

            for backend in backends:
                if (
                    backend == Backend.STATEVECTOR
                    and forced_width is not None
                    and forced_width > STATEVECTOR_MAX_QUBITS
                ):
                    message = (
                        f"circuit uses {forced_width} qubits exceeding "
                        f"statevector limit of {STATEVECTOR_MAX_QUBITS}"
                    )
                    forced_records.append(
                        {
                            "circuit": spec.name,
                            "qubits": n,
                            "actual_qubits": forced_width,
                            "framework": backend.name,
                            "backend": backend.name,
                            "unsupported": True,
                            "failed": False,
                            "error": message,
                            "comment": message,
                        }
                    )
                    LOGGER.info(
                        "Skipping forced run: circuit=%s qubits=%s backend=%s reason=%s",
                        spec.name,
                        n,
                        backend.name,
                        message,
                    )
                    continue
                if backend == Backend.MPS:
                    reason = _mps_skip_reason(
                        circuit_forced,
                        n,
                        forced_width=forced_width,
                    )
                    if reason:
                        forced_records.append(
                            {
                                "circuit": spec.name,
                                "qubits": n,
                                "actual_qubits": forced_width,
                                "framework": backend.name,
                                "backend": backend.name,
                                "unsupported": True,
                                "failed": False,
                                "error": reason,
                                "comment": reason,
                            }
                        )
                        LOGGER.info(
                            "Skipping forced run: circuit=%s qubits=%s backend=%s reason=%s",
                            spec.name,
                            n,
                            backend.name,
                            reason,
                        )
                        continue
                if not _supports_backend(circuit_forced, backend):
                    reason = "non-Clifford gates" if backend == Backend.TABLEAU else "unsupported gate set"
                    forced_records.append(
                        {
                            "circuit": spec.name,
                            "qubits": n,
                            "actual_qubits": forced_width,
                            "framework": backend.name,
                            "backend": backend.name,
                            "unsupported": True,
                            "failed": False,
                            "error": f"circuit uses {reason} unsupported by {backend.name}",
                        }
                    )
                    LOGGER.info(
                        "Skipping forced run: circuit=%s qubits=%s backend=%s reason=%s",
                        spec.name,
                        n,
                        backend.name,
                        reason,
                    )
                    continue
                runner = BenchmarkRunner()
                LOGGER.info(
                    "Executing forced run: circuit=%s qubits=%s backend=%s",
                    spec.name,
                    n,
                    backend.name,
                )
                try:
                    rec = runner.run_quasar_multiple(
                        circuit_forced,
                        engine,
                        backend=backend,
                        repetitions=repetitions,
                        quick=True,
                        memory_bytes=STATEVECTOR_SAFE_MEMORY_BYTES,
                        run_timeout=effective_timeout,
                    )
                except Exception as exc:  # pragma: no cover - backend limitations
                    forced_records.append(
                        {
                            "circuit": spec.name,
                            "qubits": n,
                            "actual_qubits": forced_width,
                            "framework": backend.name,
                            "backend": backend.name,
                            "unsupported": True,
                            "failed": False,
                            "error": str(exc),
                        }
                    )
                    LOGGER.warning(
                        "Forced run failed for circuit=%s qubits=%s backend=%s: %s",
                        spec.name,
                        n,
                        backend.name,
                        exc,
                    )
                    continue

                rec.pop("result", None)
                rec.update(
                    {
                        "circuit": spec.name,
                        "qubits": n,
                        "actual_qubits": forced_width,
                        "framework": backend.name,
                        "backend": backend.name,
                        "mode": "forced",
                        "unsupported": bool(rec.get("unsupported", False)),
                    }
                )
                forced_records.append(rec)
                LOGGER.info(
                    "Completed forced run: circuit=%s qubits=%s backend=%s",
                    spec.name,
                    n,
                    backend.name,
                )

            runner = BenchmarkRunner()
            LOGGER.info(
                "Executing automatic run: circuit=%s qubits=%s backend=quasar",
                spec.name,
                n,
            )
            auto_skip_reason = _mps_skip_reason(
                circuit_auto,
                n,
                forced_width=auto_width,
            )
            if auto_skip_reason:
                LOGGER.info(
                    "Skipping automatic run: circuit=%s qubits=%s reason=%s",
                    spec.name,
                    n,
                    auto_skip_reason,
                )
                auto_records.append(
                    {
                        "circuit": spec.name,
                        "qubits": n,
                        "actual_qubits": auto_width,
                        "framework": "quasar",
                        "backend": Backend.MPS.name,
                        "mode": "auto",
                        "unsupported": True,
                        "failed": False,
                        "error": auto_skip_reason,
                        "comment": auto_skip_reason,
                        "repetitions": 0,
                    }
                )
                continue
            try:
                rec = runner.run_quasar_multiple(
                    circuit_auto,
                    engine,
                    repetitions=repetitions,
                    quick=False,
                    memory_bytes=STATEVECTOR_SAFE_MEMORY_BYTES,
                    run_timeout=effective_timeout,
                )
            except NoFeasibleBackendError as exc:
                reason = _automatic_failure_reason(
                    circuit_auto,
                    n,
                    actual_qubits=auto_width,
                    default=str(exc),
                )
                LOGGER.info(
                    "Skipping automatic run: circuit=%s qubits=%s reason=%s",
                    spec.name,
                    n,
                    reason,
                )
                auto_records.append(
                    {
                        "circuit": spec.name,
                        "qubits": n,
                        "actual_qubits": auto_width,
                        "framework": "quasar",
                        "backend": None,
                        "mode": "auto",
                        "unsupported": True,
                        "failed": False,
                        "error": reason,
                        "comment": reason,
                        "repetitions": 0,
                    }
                )
                continue
            except Exception as exc:  # pragma: no cover - skip unsupported mixes
                LOGGER.warning(
                    "Automatic scheduling failed for circuit=%s qubits=%s: %s",
                    spec.name,
                    n,
                    exc,
                )
                continue
            rec.pop("result", None)
            rec.update(
                {
                    "circuit": spec.name,
                    "qubits": n,
                    "actual_qubits": auto_width,
                    "framework": "quasar",
                    "mode": "auto",
                    "unsupported": bool(rec.get("unsupported", False)),
                }
            )
            auto_records.append(rec)
            LOGGER.info(
                "Completed automatic run: circuit=%s qubits=%s backend=quasar",
                spec.name,
                n,
            )

    return pd.DataFrame(forced_records), pd.DataFrame(auto_records)


def generate_backend_comparison(
    *,
    repetitions: int = 3,
    run_timeout: float | None = RUN_TIMEOUT_DEFAULT_SECONDS,
    reuse_existing: bool = False,
) -> None:
    LOGGER.info("Generating backend comparison figures")
    forced_path = RESULTS_DIR / "backend_forced.csv"
    auto_path = RESULTS_DIR / "backend_auto.csv"
    allowed_qubits = {spec.name: set(spec.qubits) for spec in CIRCUITS}

    forced: pd.DataFrame
    auto: pd.DataFrame

    if reuse_existing and forced_path.exists() and auto_path.exists():
        LOGGER.info("Reusing existing backend CSVs from %s and %s", forced_path, auto_path)
        forced = pd.read_csv(forced_path)
        auto = pd.read_csv(auto_path)

        spec_lookup = {spec.name: spec for spec in CIRCUITS}
        width_cache: dict[tuple[str, int], int | None] = {}
        drop_counts = {"out_of_range": 0, "ancilla": 0}

        def _effective_width(row: pd.Series) -> int | None:
            actual = row.get("actual_qubits")
            if pd.notna(actual):
                try:
                    return int(actual)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    return None

            circuit_name = row.get("circuit")
            try:
                requested = int(row.get("qubits"))
            except (TypeError, ValueError):
                return None

            if circuit_name is None:
                return None

            key = (circuit_name, requested)
            if key in width_cache:
                return width_cache[key]

            spec = spec_lookup.get(circuit_name)
            if spec is None:
                width_cache[key] = None
                return None

            circuit = _build_circuit(spec, requested, use_classical_simplification=False)
            width = getattr(circuit, "num_qubits", None) if circuit is not None else None
            try:
                width_cache[key] = int(width) if width is not None else None
            except (TypeError, ValueError):  # pragma: no cover - defensive
                width_cache[key] = None
            return width_cache[key]

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df

            def _row_allowed(row: pd.Series) -> bool:
                circuit_name = row.get("circuit")
                allowed = allowed_qubits.get(circuit_name, set())
                if allowed and row.get("qubits") not in allowed:
                    drop_counts["out_of_range"] += 1
                    return False

                width = _effective_width(row)
                framework = str(row.get("framework", ""))
                if (
                    width is not None
                    and width > STATEVECTOR_MAX_QUBITS
                    and framework.upper() == Backend.STATEVECTOR.name
                    and not bool(row.get("unsupported"))
                ):
                    drop_counts["ancilla"] += 1
                    return False

                return True

            mask = df.apply(_row_allowed, axis=1)
            dropped = int((~mask).sum())
            if dropped:
                if drop_counts["ancilla"]:
                    LOGGER.info(
                        "Dropping %d row(s) outside the statevector limit (%d due to ancillary width)",
                        dropped,
                        drop_counts["ancilla"],
                    )
                else:
                    LOGGER.info(
                        "Dropping %d row(s) outside the statevector limit", dropped
                    )
            return df.loc[mask].copy()

        forced = _filter(forced)
        auto = _filter(auto)
    else:
        forced, auto = collect_backend_data(
            CIRCUITS,
            BACKENDS,
            repetitions=repetitions,
            run_timeout=run_timeout,
        )
    forced_path = RESULTS_DIR / "backend_forced.csv"
    auto_path = RESULTS_DIR / "backend_auto.csv"
    forced.to_csv(forced_path, index=False)
    auto.to_csv(auto_path, index=False)
    _log_written(forced_path)
    _log_written(auto_path)

    combined = pd.concat([forced, auto], ignore_index=True)
    ax, summary, fig = plot_quasar_vs_baseline_best(
        combined,
        metric="run_time_mean",
        annotate_backend=True,
        return_table=True,
        return_figure=True,
        show_speedup_table=True,
    )
    ax.set_title("Runtime comparison versus baseline best")
    png_path = FIGURES_DIR / "backend_vs_baseline.png"
    pdf_path = FIGURES_DIR / "backend_vs_baseline.pdf"
    csv_path = RESULTS_DIR / "backend_vs_baseline_speedups.csv"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    _log_written(png_path)
    _log_written(pdf_path)
    summary.to_csv(csv_path, index=False)
    _log_written(csv_path)
    plt.close(fig)

    if not forced.empty and not auto.empty:
        try:
            grid = plot_backend_timeseries(
                forced,
                auto,
                metric="run_time_mean",
                col_wrap=2,
                height=3.6,
                aspect=1.1,
                facet_kws={"sharey": False},
                annotate_offset=(0.0, 8.0),
                annotation_min_distance=24.0,
            )
        except RuntimeError as exc:
            LOGGER.warning("Skipping backend timeseries plots: %s", exc)
        else:
            runtime_png = FIGURES_DIR / "backend_timeseries_runtime.png"
            runtime_pdf = FIGURES_DIR / "backend_timeseries_runtime.pdf"
            grid.savefig(runtime_png)
            grid.savefig(runtime_pdf)
            _log_written(runtime_png)
            _log_written(runtime_pdf)
            plt.close(grid.fig)

            for frame in (forced, auto):
                frame["run_peak_memory_mib"] = frame.get("run_peak_memory_mean", 0) / (1024**2)

            grid_mem = plot_backend_timeseries(
                forced,
                auto,
                metric="run_peak_memory_mib",
                annotate_auto=False,
                col_wrap=2,
                height=3.6,
                aspect=1.1,
                facet_kws={"sharey": False},
            )
            mem_png = FIGURES_DIR / "backend_timeseries_memory.png"
            mem_pdf = FIGURES_DIR / "backend_timeseries_memory.pdf"
            grid_mem.savefig(mem_png)
            grid_mem.savefig(mem_pdf)
            _log_written(mem_png)
            _log_written(mem_pdf)
            plt.close(grid_mem.fig)
    else:
        LOGGER.info(
            "Skipping backend timeseries plots because one of the result tables is empty"
        )


def generate_heatmap() -> None:
    results_path = RESULTS_DIR / "plan_choice_heatmap_results.json"
    if not results_path.exists():
        LOGGER.info(
            "Skipping plan-choice heatmap: results file %s is missing",
            results_path,
        )
        return
    data = json.loads(results_path.read_text())
    if not data:
        LOGGER.info(
            "Skipping plan-choice heatmap: results file %s is empty",
            results_path,
        )
        return

    LOGGER.info("Generating plan-choice heatmap from %s", results_path)
    df = pd.DataFrame(data)
    df["selected_backend"] = df["steps"].apply(lambda steps: steps[-1] if steps else None)
    unique_backends = df["selected_backend"].dropna().unique()
    labels = backend_labels(unique_backends)
    short_labels = backend_labels(unique_backends, abbreviated=True)
    df["label"] = df["selected_backend"].map(
        lambda name: short_labels.get(name, labels.get(name, name))
    )

    pivot = df.pivot(index="circuit", columns="alpha", values="selected_backend")
    annot = df.pivot(index="circuit", columns="alpha", values="label")
    order = list(labels.keys())

    if not order:
        LOGGER.info("Skipping plan-choice heatmap: no backend selections present")
        return

    palette = backend_palette(order)
    missing_colours = [backend for backend in order if backend not in palette]
    if missing_colours:
        LOGGER.warning(
            "Missing palette entries for backends: %s", ", ".join(map(str, missing_colours))
        )
    colors = [palette.get(backend, "#b3b3b3") for backend in order]
    cmap = ListedColormap(colors, name="backend_palette")
    boundaries = [index - 0.5 for index in range(len(colors) + 1)]
    norm = BoundaryNorm(boundaries, cmap.N)

    pivot_numeric = (
        pivot.apply(lambda col: pd.Categorical(col, categories=order).codes)
        .astype(float)
        .replace(-1, float("nan"))
    )

    try:
        ax = plot_heatmap(
            pivot_numeric,
            annot=annot,
            fmt="",
            cmap=cmap,
            norm=norm,
            cbar=False,
        )
    except RuntimeError as exc:
        LOGGER.warning("Skipping plan-choice heatmap: %s", exc)
        return

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=labels.get(backend, str(backend)),
        )
        for backend, color in zip(order, colors)
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Backend",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

    heatmap_png = FIGURES_DIR / "plan_choice_heatmap.png"
    heatmap_pdf = FIGURES_DIR / "plan_choice_heatmap.pdf"
    table_csv = RESULTS_DIR / "plan_choice_heatmap_table.csv"
    ax.figure.savefig(heatmap_png)
    ax.figure.savefig(heatmap_pdf)
    _log_written(heatmap_png)
    _log_written(heatmap_pdf)
    annot.to_csv(table_csv)
    _log_written(table_csv)
    plt.close(ax.figure)


def generate_speedup_bars() -> None:
    csv_path = Path(__file__).resolve().parent / "quick_analysis_results.csv"
    if not csv_path.exists():
        LOGGER.info(
            "Skipping speedup summary: quick-analysis results %s not found",
            csv_path,
        )
        return
    df = pd.read_csv(csv_path)
    LOGGER.info("Generating speedup summary from %s", csv_path)
    df["label"] = df.apply(
        lambda row: f"{int(row['qubits'])}q d{int(row['depth'])}", axis=1
    )
    grouped = df.groupby("label")
    speedups = grouped["speedup"].mean().to_dict()
    ax = plot_speedup_bars(speedups)
    ax.figure.tight_layout()
    speedup_png = FIGURES_DIR / "relative_speedups.png"
    speedup_pdf = FIGURES_DIR / "relative_speedups.pdf"
    speedup_csv = RESULTS_DIR / "relative_speedups.csv"
    ax.figure.savefig(speedup_png)
    ax.figure.savefig(speedup_pdf)
    _log_written(speedup_png)
    _log_written(speedup_pdf)
    grouped["speedup"].mean().reset_index().to_csv(speedup_csv, index=False)
    _log_written(speedup_csv)
    plt.close(ax.figure)


def generate_partitioning_figures() -> None:
    try:
        from docs.utils import partitioning_analysis
    except ImportError as exc:  # pragma: no cover - optional dependency
        LOGGER.warning(
            "Skipping partitioning figures: documentation helpers unavailable (%s)",
            exc,
        )
        return

    estimator, calibration_path = partitioning_analysis.load_calibrated_estimator()
    if calibration_path is not None:
        LOGGER.info(
            "Loaded partitioning calibration coefficients from %s",
            calibration_path,
        )
    else:
        LOGGER.info(
            "Using default cost estimator coefficients for partitioning figures",
        )

    partitioning_analysis.apply_partitioning_style()

    def _save(paths: Iterable[Path]) -> None:
        for path in paths:
            _log_written(path)

    def _safe_run(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.warning("Skipping %s: %s", name, exc)

    def _clifford_crossover() -> None:
        curves = partitioning_analysis.build_clifford_fragment_curves(estimator)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(curves["num_qubits"], curves["statevector"], label="Statevector", linewidth=2.2)
        ax.plot(curves["num_qubits"], curves["tableau"], label="Tableau", linewidth=2.2)
        ax.set_xlim(curves["num_qubits"][0], curves["num_qubits"][-1])
        combined = np.concatenate(
            (
                np.asarray(curves["statevector"], dtype=float),
                np.asarray(curves["tableau"], dtype=float),
            )
        )
        ax.set_ylim(0, float(np.max(combined)) * 1.1)
        threshold = curves.get("threshold")
        if threshold is not None:
            indices = np.where(curves["num_qubits"] == threshold)[0]
            if indices.size:
                idx = int(indices[0])
                y_val = float(curves["tableau"][idx])
                ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
                x_text = min(curves["num_qubits"][-1], threshold + 1.5)
                ax.annotate(
                    f"Tableau cheaper ≥ {threshold} qubits",
                    xy=(threshold, y_val),
                    xytext=(x_text, y_val * 1.05),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0),
                    fontsize=10,
                )
                LOGGER.info(
                    "Tableau crossover occurs at %s qubits in the Clifford fragment",
                    threshold,
                )
        else:
            LOGGER.info(
                "No tableau crossover observed for Clifford fragments in sampled range",
            )
        ax.set_xlabel("Active qubits")
        ax.set_ylabel("Estimated runtime (arb. units)")
        ax.set_title("Clifford fragment crossover")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(partitioning_analysis.export_figure(fig, "clifford_crossover"))
        plt.close(fig)

    def _statevector_partition_tradeoff() -> None:
        tradeoff = partitioning_analysis.build_statevector_partition_tradeoff(estimator)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(tradeoff["num_qubits"], tradeoff["statevector"], label="Statevector only", linewidth=2.2)
        ax.plot(
            tradeoff["num_qubits"],
            tradeoff["partitioned"],
            label="Partitioned with conversions",
            linewidth=2.2,
        )
        ax.set_xlim(tradeoff["num_qubits"][0], tradeoff["num_qubits"][-1])
        combined = np.concatenate(
            (
                np.asarray(tradeoff["statevector"], dtype=float),
                np.asarray(tradeoff["partitioned"], dtype=float),
            )
        )
        ax.set_ylim(0, float(np.max(combined)) * 1.12)
        threshold = tradeoff.get("threshold")
        if threshold is not None:
            indices = np.where(tradeoff["num_qubits"] == threshold)[0]
            if indices.size:
                idx = int(indices[0])
                boundary = int(tradeoff["boundary"][idx])
                rank = int(tradeoff["rank"][idx])
                y_val = float(tradeoff["partitioned"][idx])
                ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
                x_text = min(tradeoff["num_qubits"][-1], threshold + 2)
                ax.annotate(
                    f"Switch at q={boundary} (rank≤{rank})",
                    xy=(threshold, y_val),
                    xytext=(x_text, y_val * 1.05),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0),
                    fontsize=10,
                )
                LOGGER.info(
                    "Partitioned execution overtakes dense simulation beyond %s qubits",
                    threshold,
                )
        else:
            LOGGER.info(
                "Partitioned execution never beats dense simulation in sampled range",
            )
        ax.set_xlabel("Active qubits")
        ax.set_ylabel("Estimated runtime (arb. units)")
        ax.set_title("Statevector vs. tableau with conversions")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(partitioning_analysis.export_figure(fig, "statevector_tableau_partition"))
        plt.close(fig)

    def _statevector_vs_mps() -> None:
        curves = partitioning_analysis.build_statevector_vs_mps(estimator)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(curves["num_qubits"], curves["statevector"], label="Statevector", linewidth=2.2)
        ax.plot(curves["num_qubits"], curves["mps"], label="MPS (χ=4)", linewidth=2.2)
        ax.set_xlim(curves["num_qubits"][0], curves["num_qubits"][-1])
        combined = np.concatenate(
            (
                np.asarray(curves["statevector"], dtype=float),
                np.asarray(curves["mps"], dtype=float),
            )
        )
        ax.set_ylim(0, float(np.max(combined)) * 1.12)
        threshold = curves.get("threshold")
        if threshold is not None:
            indices = np.where(curves["num_qubits"] == threshold)[0]
            if indices.size:
                idx = int(indices[0])
                y_val = float(curves["mps"][idx])
                ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
                x_text = min(curves["num_qubits"][-1], threshold + 1.5)
                ax.annotate(
                    f"MPS cheaper ≥ {threshold} qubits",
                    xy=(threshold, y_val),
                    xytext=(x_text, y_val * 1.05),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0),
                    fontsize=10,
                )
                LOGGER.info(
                    "MPS crossover occurs at %s qubits for local circuits",
                    threshold,
                )
        else:
            LOGGER.info("MPS crossover not observed for sampled qubit counts")
        ax.set_xlabel("Active qubits")
        ax.set_ylabel("Estimated runtime (arb. units)")
        ax.set_title("Local circuit cost")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(partitioning_analysis.export_figure(fig, "statevector_vs_mps"))
        plt.close(fig)

    def _conversion_aware_mps() -> None:
        data = partitioning_analysis.build_conversion_aware_mps_paths(estimator)
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        baseline_line, = ax.plot(
            data["num_qubits"], data["statevector"], label="Statevector only", linewidth=2.2
        )
        _ = baseline_line
        y_values = [np.asarray(data["statevector"], dtype=float)]
        for scenario in data["scenarios"]:
            label = f"MPS path (χ={scenario['chi']}, window={scenario['window']})"
            line, = ax.plot(
                data["num_qubits"], scenario["total"], label=label, linewidth=2.0
            )
            colour = line.get_color()
            y_values.append(np.asarray(scenario["total"], dtype=float))
            threshold = scenario.get("threshold")
            if threshold is not None:
                indices = np.where(data["num_qubits"] == threshold)[0]
                if indices.size:
                    idx = int(indices[0])
                    y_val = float(scenario["total"][idx])
                    ax.scatter([threshold], [y_val], color=colour, zorder=5)
                    x_text = min(data["num_qubits"][-1], threshold + 1.5)
                    ax.annotate(
                        f"Switch ≥ {threshold} qubits",
                        xy=(threshold, y_val),
                        xytext=(x_text, y_val * 1.05),
                        arrowprops=dict(arrowstyle="->", linewidth=1.0, color=colour),
                        fontsize=9,
                        color=colour,
                    )
            conv_time = float(scenario["sv_to_mps"] + scenario["mps_to_sv"])
            threshold_label = str(threshold) if threshold is not None else "not observed"
            LOGGER.info(
                "%s: conversions add %.2f a.u.; threshold %s",
                label,
                conv_time,
                threshold_label,
            )
        combined = np.concatenate(y_values)
        ax.set_ylim(0, float(np.max(combined)) * 1.12)
        ax.set_xlim(data["num_qubits"][0], data["num_qubits"][-1])
        ax.set_xlabel("Active qubits")
        ax.set_ylabel("Estimated runtime (arb. units)")
        ax.set_title("Conversion-aware MPS planning")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(partitioning_analysis.export_figure(fig, "conversion_aware_mps"))
        plt.close(fig)

    def _statevector_vs_decision_diagram() -> None:
        curves = partitioning_analysis.build_statevector_vs_decision_diagram(estimator)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(curves["num_qubits"], curves["statevector"], label="Statevector", linewidth=2.2)
        ax.plot(
            curves["num_qubits"],
            curves["decision_diagram"],
            label="Decision diagram",
            linewidth=2.2,
        )
        ax.set_xlim(curves["num_qubits"][0], curves["num_qubits"][-1])
        combined = np.concatenate(
            (
                np.asarray(curves["statevector"], dtype=float),
                np.asarray(curves["decision_diagram"], dtype=float),
            )
        )
        ax.set_ylim(0, float(np.max(combined)) * 1.12)
        cheaper = curves["decision_diagram"] < curves["statevector"]
        if np.any(cheaper):
            idx = int(np.where(cheaper)[0][0])
            threshold = int(curves["num_qubits"][idx])
            y_val = float(curves["decision_diagram"][idx])
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
            x_text = min(curves["num_qubits"][-1], threshold + 1.5)
            ax.annotate(
                f"DD cheaper ≥ {threshold} qubits",
                xy=(threshold, y_val),
                xytext=(x_text, y_val * 1.05),
                arrowprops=dict(arrowstyle="->", linewidth=1.0),
                fontsize=10,
            )
            LOGGER.info(
                "Decision diagrams overtake dense simulation from %s qubits",
                threshold,
            )
        else:
            LOGGER.info(
                "Decision-diagram crossover not observed for sampled sparse circuits",
            )
        ax.set_xlabel("Active qubits")
        ax.set_ylabel("Estimated runtime (arb. units)")
        ax.set_title("Sparse circuit cost")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(partitioning_analysis.export_figure(fig, "statevector_vs_decision_diagram"))
        plt.close(fig)

    def _conversion_primitive_selection() -> None:
        rows = partitioning_analysis.build_conversion_primitive_costs(estimator)
        if not rows:
            LOGGER.info(
                "No conversion primitive data returned; skipping primitive selection plot",
            )
            return
        qs = np.asarray([row["boundary"] for row in rows], dtype=int)
        times = np.asarray([row["time"] for row in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(qs, times, marker="o", linewidth=2.2)
        ax.set_xlabel("Boundary size q")
        ax.set_ylabel("Estimated conversion time (arb. units)")
        ax.set_title("Conversion primitive selection")
        ax.set_xticks(qs)
        for row in rows:
            ax.annotate(
                row["primitive"],
                xy=(row["boundary"], row["time"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        fig.tight_layout()
        _save(partitioning_analysis.export_figure(fig, "conversion_primitive_selection"))
        plt.close(fig)
        last = None
        for row in rows:
            primitive = row["primitive"]
            if primitive != last:
                LOGGER.info(
                    "Boundary q=%s prefers primitive %s (%.2f a.u.)",
                    row["boundary"],
                    primitive,
                    row["time"],
                )
                last = primitive

    def _partition_plan_breakdowns() -> None:
        scenarios = partitioning_analysis.documentation_plan_scenarios()
        for scenario in scenarios:
            plan = scenario.evaluate(estimator)
            records: list[dict[str, object]] = [
                {
                    "plan": "Single backend",
                    "component": "Full circuit (statevector)",
                    "type": "Simulation",
                    "backend": "Statevector",
                    "primitive": "",
                    "time": plan["single_backend"].time,
                    "memory": plan["single_backend"].memory,
                }
            ]

            for frag in plan["fragments"]:
                records.append(
                    {
                        "plan": "Partitioned",
                        "component": frag["name"],
                        "type": "Simulation",
                        "backend": frag["backend"].name.replace("_", " ").title(),
                        "primitive": "",
                        "time": frag["cost"].time,
                        "memory": frag["cost"].memory,
                    }
                )

            for conv in plan["conversions"]:
                records.append(
                    {
                        "plan": "Partitioned",
                        "component": conv["name"],
                        "type": "Conversion",
                        "backend": (
                            f"{conv['source'].name.replace('_', ' ').title()}"
                            f"→{conv['target'].name.replace('_', ' ').title()}"
                        ),
                        "primitive": conv["primitive"],
                        "time": conv["cost"].time,
                        "memory": conv["cost"].memory,
                    }
                )

            df = pd.DataFrame(records)
            component_order: list[str] = []
            for record in records:
                component = str(record["component"])
                if component not in component_order:
                    component_order.append(component)

            pivot = df.pivot_table(
                index="plan",
                columns="component",
                values="time",
                aggfunc="sum",
                fill_value=0,
            )
            pivot = pivot[component_order]
            ax = pivot.plot(kind="bar", stacked=True, figsize=(9, 4))
            ax.set_ylabel("Estimated time (arb. units)")
            ax.set_title(scenario.title)
            ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig = ax.figure
            fig.tight_layout()
            outputs = partitioning_analysis.export_figure(
                fig, f"partition_plan_breakdown_{scenario.key}"
            )
            if scenario.key == "balanced_handoff":
                outputs.extend(
                    partitioning_analysis.export_figure(fig, "partition_plan_breakdown")
                )
            _save(outputs)
            plt.close(fig)

            total_cost = plan["aggregate"]["total_cost"]
            LOGGER.info(
                "%s: partitioned total %.2f a.u. (peak %.2f) vs. single backend %.2f a.u. (peak %.2f)",
                scenario.title,
                total_cost.time,
                total_cost.memory,
                plan["single_backend"].time,
                plan["single_backend"].memory,
            )

    _safe_run("Clifford crossover plot", _clifford_crossover)
    _safe_run("Statevector/tableau trade-off plot", _statevector_partition_tradeoff)
    _safe_run("Statevector vs. MPS plot", _statevector_vs_mps)
    _safe_run("Conversion-aware MPS plot", _conversion_aware_mps)
    _safe_run("Statevector vs. decision-diagram plot", _statevector_vs_decision_diagram)
    _safe_run("Conversion primitive selection plot", _conversion_primitive_selection)
    _safe_run("Partition plan breakdown plots", _partition_plan_breakdowns)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible benchmark figures for the QuASAr paper",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for debug output).",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per circuit/backend combination (default: 3).",
    )
    parser.add_argument(
        "-t",
        "--run-timeout",
        type=float,
        default=RUN_TIMEOUT_DEFAULT_SECONDS,
        help=(
            "Abort individual backend runs after this many seconds "
            f"(default: {RUN_TIMEOUT_DEFAULT_SECONDS}; set to <= 0 to disable)."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Load and filter cached backend CSVs instead of running new simulations.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    setup_benchmark_style()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    run_timeout = args.run_timeout if args.run_timeout and args.run_timeout > 0 else None
    generate_backend_comparison(
        repetitions=args.repetitions,
        run_timeout=run_timeout,
        reuse_existing=args.reuse_existing,
    )
    generate_heatmap()
    generate_speedup_bars()
    generate_partitioning_figures()


if __name__ == "__main__":
    main()
