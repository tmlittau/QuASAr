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
import tracemalloc
import statistics

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
    def _invoke(self, backend: Any, circuit: Any, **kwargs: Any) -> Any:
        """Call ``backend`` with ``circuit``.

        ``backend`` may expose a ``run`` method or be directly callable.  Any
        return value is forwarded to the caller but also stored in the result
        record for completeness.  Additional keyword arguments are forwarded to
        the backend invocation.  This allows benchmarks to control whether the
        backend should return a state representation or merely perform the
        simulation.
        """

        if hasattr(backend, "run"):
            return backend.run(circuit, **kwargs)
        if callable(backend):
            return backend(circuit, **kwargs)
        raise TypeError("backend must be callable or provide a 'run' method")

    # ------------------------------------------------------------------
    def run(self, circuit: Any, backend: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute ``circuit`` on ``backend`` and record runtime and memory."""

        prepare_time = 0.0
        prepare_peak_memory = 0
        run_peak_memory = 0

        tracemalloc.start()

        if hasattr(backend, "prepare_benchmark") and hasattr(backend, "run_benchmark"):
            start_prepare = time.perf_counter()
            if hasattr(backend, "load") and getattr(circuit, "num_qubits", None) is not None:
                backend.load(circuit.num_qubits)
            backend.prepare_benchmark(circuit)
            for g in getattr(circuit, "gates", []):
                backend.apply_gate(g.gate, g.qubits, g.params)
            prepare_time = time.perf_counter() - start_prepare
            _, prepare_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()

            start_run = time.perf_counter()
            result = backend.run_benchmark(**kwargs)
            run_time = time.perf_counter() - start_run
            _, run_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            prepared = circuit
            if hasattr(backend, "prepare"):
                start_prepare = time.perf_counter()
                prepared = backend.prepare(circuit)
                prepare_time = time.perf_counter() - start_prepare
                _, prepare_peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()

            start_run = time.perf_counter()
            result = self._invoke(backend, prepared, **kwargs)
            run_time = time.perf_counter() - start_run
            _, run_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        record = {
            "framework": getattr(backend, "name", backend.__class__.__name__),
            "prepare_time": prepare_time,
            "run_time": run_time,
            "total_time": prepare_time + run_time,
            "prepare_peak_memory": prepare_peak_memory,
            "run_peak_memory": run_peak_memory,
            "result": result,
        }
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def run_multiple(
        self,
        circuit: Any,
        backend: Any,
        *,
        repetitions: int = 1,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute ``run`` repeatedly and return aggregate statistics.

        Parameters
        ----------
        circuit, backend : Any
            Passed through to :meth:`run`.
        repetitions : int, optional
            Number of times to execute ``run``. Defaults to ``1``.
        timeout : float | None, optional
            Maximum total time in seconds to spend on executions.  When set, no
            further repetitions are started once the timeout is exceeded.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`run`.

        Returns
        -------
        Dict[str, Any]
            Mapping containing mean and standard deviation for each recorded
            metric.  Individual run results are not retained in
            :attr:`results` – only the aggregated statistics are stored.
        """

        metrics = [
            "prepare_time",
            "run_time",
            "total_time",
            "prepare_peak_memory",
            "run_peak_memory",
        ]
        records: List[Dict[str, Any]] = []
        start = time.perf_counter()
        for _ in range(repetitions):
            rec = self.run(circuit, backend, **kwargs)
            # ``run`` already appends to ``self.results``; remove the individual
            # entry so only aggregated statistics remain.
            self.results.pop()
            records.append(rec)
            if timeout is not None and (time.perf_counter() - start) > timeout:
                break

        if not records:
            raise RuntimeError("no runs executed")

        summary: Dict[str, Any] = {
            "framework": getattr(backend, "name", backend.__class__.__name__),
            "repetitions": len(records),
        }
        for m in metrics:
            values = [r[m] for r in records]
            summary[f"{m}_mean"] = statistics.fmean(values)
            summary[f"{m}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )

        self.results.append(summary)
        return summary

    # ------------------------------------------------------------------
    def run_quasar(self, circuit: Any, engine: Any) -> Dict[str, Any]:
        """Execute ``circuit`` using a QuASAr scheduler ``engine``.

        Planning is performed prior to measurement so that only the actual
        :meth:`quasar.scheduler.Scheduler.run` invocation is timed.  Both
        phases are wrapped with :mod:`tracemalloc` to also capture peak memory
        usage.  ``engine`` may either be a :class:`~quasar.scheduler.Scheduler`
        or an object providing ``scheduler`` and ``planner`` attributes (e.g.,
        :class:`~quasar.simulation_engine.SimulationEngine`).
        """

        scheduler = getattr(engine, "scheduler", engine)
        planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
        prepare_time = 0.0
        prepare_peak_memory = 0
        run_peak_memory = 0
        tracemalloc.start()
        if planner is not None:
            start_prepare = time.perf_counter()
            planner.plan(circuit)
            prepare_time = time.perf_counter() - start_prepare
            _, prepare_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        start_run = time.perf_counter()
        result = scheduler.run(circuit)
        run_time = time.perf_counter() - start_run
        _, run_peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record = {
            "framework": "quasar",
            "prepare_time": prepare_time,
            "run_time": run_time,
            "total_time": prepare_time + run_time,
            "prepare_peak_memory": prepare_peak_memory,
            "run_peak_memory": run_peak_memory,
            "result": result,
        }
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def dataframe(self) -> "pd.DataFrame | List[Dict[str, Any]]":
        """Return collected results as a :class:`pandas.DataFrame` if available.

        The returned data includes separate ``prepare_time``/``run_time`` and
        their sum ``total_time`` as well as ``prepare_peak_memory``/``run_peak_memory``
        columns for downstream analysis.
        """

        if pd is None:
            return self.results
        return pd.DataFrame(self.results)


__all__ = ["BenchmarkRunner"]
