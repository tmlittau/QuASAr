from __future__ import annotations

"""Utility helpers for executing benchmarks.

This module defines :class:`BenchmarkRunner` which provides a light-weight
API for executing circuits on different backends while collecting timing data.
Two entry points are exposed:

``run``
    Execute a circuit on an arbitrary backend or callable.

``run_quasar``
    Specialised variant that measures QuASAr execution via
    :class:`~quasar.scheduler.Scheduler`.  Planning and other analysis steps
    are performed ahead of time so that only the actual ``Scheduler.run`` call
    contributes to the recorded runtime.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import time

try:  # ``pandas`` is optional; benchmarks fall back to plain records.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

# Type for objects that expose a ``run`` method
RunCallable = Callable[[Any], Any]


@dataclass
class BenchmarkRunner:
    """Execute circuits on various simulators and collect timing data."""

    results: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def _invoke(self, backend: Any, circuit: Any) -> Any:
        """Call ``backend`` with ``circuit``.

        ``backend`` may expose a ``run`` method or be directly callable.  Any
        return value is forwarded to the caller but also stored in the result
        record for completeness.
        """

        if hasattr(backend, "run"):
            return backend.run(circuit)
        if callable(backend):
            return backend(circuit)
        raise TypeError("backend must be callable or provide a 'run' method")

    # ------------------------------------------------------------------
    def run(self, circuit: Any, backend: Any) -> Dict[str, Any]:
        """Execute ``circuit`` on ``backend`` and record the runtime."""

        start = time.perf_counter()
        result = self._invoke(backend, circuit)
        elapsed = time.perf_counter() - start
        record = {
            "framework": getattr(backend, "name", backend.__class__.__name__),
            "time": elapsed,
            "result": result,
        }
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def run_quasar(self, circuit: Any, engine: Any) -> Dict[str, Any]:
        """Execute ``circuit`` using a QuASAr scheduler ``engine``.

        Planning is performed prior to measurement so that only the actual
        :meth:`quasar.scheduler.Scheduler.run` invocation is timed.
        ``engine`` may either be a :class:`~quasar.scheduler.Scheduler` or an
        object providing ``scheduler`` and ``planner`` attributes (e.g.,
        :class:`~quasar.simulation_engine.SimulationEngine`).
        """

        scheduler = getattr(engine, "scheduler", engine)
        planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
        if planner is not None:
            planner.plan(circuit)
        start = time.perf_counter()
        result = scheduler.run(circuit)
        elapsed = time.perf_counter() - start
        record = {"framework": "quasar", "time": elapsed, "result": result}
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def dataframe(self) -> "pd.DataFrame | List[Dict[str, Any]]":
        """Return collected results as a :class:`pandas.DataFrame` if available."""

        if pd is None:
            return self.results
        return pd.DataFrame(self.results)


__all__ = ["BenchmarkRunner"]
