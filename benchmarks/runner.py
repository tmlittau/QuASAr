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
from contextlib import contextmanager
import copy
import signal
import statistics
import time
import tracemalloc

try:  # ``pandas`` is optional; benchmarks fall back to plain records.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

# Type for objects that expose a ``run`` method
RunCallable = Callable[[Any], Any]

from quasar.cost import Backend


@contextmanager
def _time_limit(seconds: float) -> None:
    """Raise :class:`TimeoutError` after ``seconds`` have elapsed."""

    def handler(signum: int, frame: Any) -> None:  # pragma: no cover - external
        raise TimeoutError("operation timed out")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:  # pragma: no cover - reset alarm
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


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
        """Execute ``circuit`` on ``backend`` and record runtime and memory.

        Backends that raise :class:`NotImplementedError` are classified as
        "unsupported" and reported accordingly without counting as failures.
        """

        prepare_time = 0.0
        prepare_peak_memory = 0
        run_peak_memory = 0
        run_time = 0.0
        result: Any = None

        tracemalloc.start()

        try:
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

                kwargs.setdefault("return_state", True)
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
        except NotImplementedError as exc:
            _, run_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            record = {
                "framework": getattr(backend, "name", backend.__class__.__name__),
                "backend": getattr(backend, "name", backend.__class__.__name__),
                "prepare_time": prepare_time,
                "run_time": run_time,
                "total_time": prepare_time + run_time,
                "prepare_peak_memory": prepare_peak_memory,
                "run_peak_memory": run_peak_memory,
                "result": result,
                "failed": False,
                "unsupported": True,
                "error": str(exc),
            }
            self.results.append(record)
            return record
        except Exception as exc:  # pragma: no cover - exercised in tests
            _, run_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            record = {
                "framework": getattr(backend, "name", backend.__class__.__name__),
                "backend": getattr(backend, "name", backend.__class__.__name__),
                "prepare_time": prepare_time,
                "run_time": run_time,
                "total_time": prepare_time + run_time,
                "prepare_peak_memory": prepare_peak_memory,
                "run_peak_memory": run_peak_memory,
                "result": result,
                "failed": True,
                "error": str(exc),
            }
            self.results.append(record)
            return record

        record = {
            "framework": getattr(backend, "name", backend.__class__.__name__),
            "backend": getattr(backend, "name", backend.__class__.__name__),
            "prepare_time": prepare_time,
            "run_time": run_time,
            "total_time": prepare_time + run_time,
            "prepare_peak_memory": prepare_peak_memory,
            "run_peak_memory": run_peak_memory,
            "result": result,
            "failed": False,
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
        run_timeout: float | None = None,
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
        run_timeout : float | None, optional
            Maximum time in seconds to allow for each individual ``run`` call.
            When a run exceeds this limit, it is recorded as failed and the
            next repetition is started.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`run`.

        Returns
        -------
        Dict[str, Any]
            Mapping containing mean and standard deviation for each recorded
            metric.  Individual run results are not retained in
            :attr:`results` – only the aggregated statistics are stored.  If
            the backend reports as unsupported, the returned record contains an
            ``unsupported`` flag and no metrics.
        """

        metrics = [
            "prepare_time",
            "run_time",
            "total_time",
            "prepare_peak_memory",
            "run_peak_memory",
        ]
        records: List[Dict[str, Any]] = []
        failures: List[str] = []
        unsupported_comment: str | None = None

        def _run_once() -> Dict[str, Any]:
            rec = self.run(circuit, backend, **kwargs)
            # ``run`` already appends to ``self.results``; remove the individual
            # entry so only aggregated statistics remain.
            self.results.pop()
            return rec

        start = time.perf_counter()
        for i in range(repetitions):
            try:
                if run_timeout is not None:
                    with _time_limit(run_timeout):
                        rec = _run_once()
                else:
                    rec = _run_once()
            except TimeoutError:
                failures.append(
                    f"run {i + 1} timed out after {run_timeout} seconds"
                )
                continue
            except Exception as exc:  # pragma: no cover - safety net
                failures.append(f"run {i + 1} failed: {exc}")
                continue

            if rec.get("unsupported"):
                unsupported_comment = rec.get("error", "backend not supported")
                break
            if rec.get("failed"):
                failures.append(
                    f"run {i + 1} failed: {rec.get('error', 'unknown error')}"
                )
                continue

            records.append(rec)
            if timeout is not None and (time.perf_counter() - start) > timeout:
                break

        backend_name = getattr(backend, "name", backend.__class__.__name__)
        if unsupported_comment is not None:
            summary = {
                "framework": backend_name,
                "backend": backend_name,
                "repetitions": 0,
                "unsupported": True,
                "comment": unsupported_comment,
            }
            self.results.append(summary)
            return summary
        if not records:
            raise RuntimeError("no runs executed")

        summary: Dict[str, Any] = {
            "framework": backend_name,
            "backend": records[0].get("backend") or backend_name,
            "repetitions": len(records),
        }
        if failures:
            summary["failed_runs"] = failures
            summary["comment"] = (
                f"{len(failures)} run(s) failed and were excluded from statistics"
            )
        for m in metrics:
            values = [r[m] for r in records]
            summary[f"{m}_mean"] = statistics.fmean(values)
            summary[f"{m}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )

        summary["result"] = records[-1].get("result") if records else None

        self.results.append(summary)
        return summary

    # ------------------------------------------------------------------
    def run_quasar(
        self,
        circuit: Any,
        engine: Any,
        *,
        backend: Backend | None = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """Execute ``circuit`` using a QuASAr scheduler ``engine``.

        Planning is performed prior to measurement so that only gate execution
        reported by :meth:`quasar.scheduler.Scheduler.run` contributes to the
        recorded runtime.  Planning is profiled using :mod:`tracemalloc` while
        execution relies on the ``Cost`` object returned from a single
        ``scheduler.run(..., instrument=True)`` invocation.  ``engine`` may
        either be a
        :class:`~quasar.scheduler.Scheduler` or an object providing
        ``scheduler`` and ``planner`` attributes (e.g.,
        :class:`~quasar.simulation_engine.SimulationEngine`).  The optional
        ``backend`` argument forces both planning and execution to use a
        specific backend rather than selecting one automatically.  When
        ``quick`` is set to ``True`` the quick-path is taken regardless of the
        scheduler's heuristics.

        The returned record contains a ``backend`` field indicating the
        backend chosen by the scheduler.
        """

        scheduler = getattr(engine, "scheduler", engine)
        planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
        prepare_time = 0.0
        prepare_peak_memory = 0
        run_peak_memory = 0
        run_time = 0.0
        result: Any | None = None
        backend_choice_name: str | None = None

        try:
            backend_choice = None
            use_quick = quick
            should_quick = getattr(scheduler, "should_use_quick_path", None)
            if callable(should_quick):
                try:
                    use_quick = should_quick(circuit, backend=backend, force=quick)
                except TypeError:  # pragma: no cover - legacy signature
                    use_quick = should_quick(circuit, backend=backend)
                    if quick:
                        use_quick = True
            elif quick:
                use_quick = True

            if use_quick:
                tracemalloc.start()
                select_backend = getattr(scheduler, "select_backend", None)
                if callable(select_backend):
                    backend_choice = select_backend(circuit, backend=backend)
                else:
                    backend_choice = backend

                start_prepare = time.perf_counter()
                sim = type(scheduler.backends[backend_choice])()
                sim.load(circuit.num_qubits)
                prepare_time = time.perf_counter() - start_prepare
                _, prepare_peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                start_run = time.perf_counter()
                for g in getattr(circuit, "gates", []):
                    sim.apply_gate(g.gate, g.qubits, g.params)
                result = sim.extract_ssd()
                run_time = time.perf_counter() - start_run
                result = result if result is not None else getattr(circuit, "ssd", None)
                backend_choice_name = getattr(backend_choice, "name", str(backend_choice))
                _, run_peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            else:
                tracemalloc.start()
                if planner is not None:
                    start_prepare = time.perf_counter()
                    plan = planner.plan(circuit, backend=backend)
                    plan = scheduler.prepare_run(circuit, plan, backend=backend)
                    prepare_time = time.perf_counter() - start_prepare
                    _, prepare_peak_memory = tracemalloc.get_traced_memory()
                else:
                    start_prepare = time.perf_counter()
                    plan = scheduler.prepare_run(circuit, backend=backend)
                    prepare_time = time.perf_counter() - start_prepare
                    _, prepare_peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                result, run_cost = scheduler.run(circuit, plan, instrument=True)
                run_time = run_cost.time
                run_peak_memory = int(run_cost.memory)
                if hasattr(result, "partitions") and getattr(result, "partitions"):
                    backend_obj = result.partitions[0].backend
                    backend_choice_name = getattr(backend_obj, "name", str(backend_obj))
        except Exception as exc:  # pragma: no cover - exercised in tests
            if tracemalloc.is_tracing():
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
                "failed": True,
                "error": str(exc),
                "backend": backend_choice_name,
            }
            self.results.append(record)
            return record

        record = {
            "framework": "quasar",
            "prepare_time": prepare_time,
            "run_time": run_time,
            "total_time": prepare_time + run_time,
            "prepare_peak_memory": prepare_peak_memory,
            "run_peak_memory": run_peak_memory,
            "result": result,
            "failed": False,
            "backend": backend_choice_name,
        }
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def setup_quasar(
        self,
        circuit: Any,
        engine: Any,
        *,
        backend: Backend | None = None,
        quick: bool = False,
    ) -> None:
        """Perform any setup required before timed QuASAr measurements.

        This runs a single instrumented scheduler invocation to calibrate cost
        estimates and warm up any backend state.  The circuit's SSD as well as
        estimator coefficients are restored to their original values so that
        subsequent measurements are unaffected.
        """

        scheduler = getattr(engine, "scheduler", engine)
        planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
        use_quick = quick
        should_quick = getattr(scheduler, "should_use_quick_path", None)
        if callable(should_quick) and circuit is not None:
            try:
                use_quick = should_quick(circuit, backend=backend, force=quick)
            except TypeError:  # pragma: no cover - legacy signature
                use_quick = should_quick(circuit, backend=backend)
                if quick:
                    use_quick = True
        elif quick:
            use_quick = True

        if use_quick:
            return

        if planner is not None:
            plan = planner.plan(circuit, backend=backend)
            plan = scheduler.prepare_run(circuit, plan, backend=backend)
        else:
            plan = scheduler.prepare_run(circuit, backend=backend)

        original_ssd = (
            copy.deepcopy(getattr(circuit, "ssd", None)) if circuit is not None else None
        )
        est = getattr(planner, "estimator", None)
        coeff_backup = copy.deepcopy(getattr(est, "coeff", {})) if est else None

        scheduler.run(circuit, plan, instrument=True)

        if est is not None and coeff_backup is not None:
            est.coeff = coeff_backup
        if circuit is not None and original_ssd is not None:
            circuit.ssd = copy.deepcopy(original_ssd)

    # ------------------------------------------------------------------
    def run_quasar_multiple(
        self,
        circuit: Any,
        engine: Any,
        *,
        backend: Backend | None = None,
        repetitions: int = 1,
        timeout: float | None = None,
        run_timeout: float | None = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """Execute :meth:`run_quasar` repeatedly and aggregate statistics.

        When ``backend`` is provided it is forwarded to each
        :meth:`run_quasar` invocation to force a specific backend.  Setting
        ``quick`` to ``True`` forces quick-path execution for each run.  The
        optional ``run_timeout`` aborts individual executions after the given
        number of seconds.  Timed out runs are omitted from the final
        statistics.  The summary record includes the chosen ``backend``.
        """

        metrics = [
            "prepare_time",
            "run_time",
            "total_time",
            "prepare_peak_memory",
            "run_peak_memory",
        ]
        records: List[Dict[str, Any]] = []
        failures: List[str] = []

        scheduler = getattr(engine, "scheduler", engine)
        planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
        use_quick = quick
        should_quick = getattr(scheduler, "should_use_quick_path", None)
        if callable(should_quick) and circuit is not None:
            try:
                use_quick = should_quick(circuit, backend=backend, force=quick)
            except TypeError:  # pragma: no cover - legacy signature
                use_quick = should_quick(circuit, backend=backend)
                if quick:
                    use_quick = True
        elif quick:
            use_quick = True

        simple_backend: Backend | None = None
        simple_gates: List[Any] | None = None
        if use_quick:
            plan = None
            original_ssd = None
        else:
            if planner is not None:
                plan = planner.plan(circuit, backend=backend)
                plan = scheduler.prepare_run(circuit, plan, backend=backend)
            else:
                plan = scheduler.prepare_run(circuit, backend=backend)
            original_ssd = (
                copy.deepcopy(getattr(circuit, "ssd", None)) if circuit is not None else None
            )
            est = getattr(planner, "estimator", None)
            coeff_backup = copy.deepcopy(getattr(est, "coeff", {})) if est else None

            conv_layers = list(getattr(plan, "conversions", []))
            parts = getattr(plan, "partitions", None)
            if parts is not None and len(parts) == 1 and not conv_layers:
                backend_choice = parts[0].backend
                if backend_choice in getattr(scheduler, "backends", {}):
                    simple_backend = backend_choice
                    simple_gates = list(getattr(plan, "gates", []))
            else:
                steps = list(getattr(plan, "steps", []))
                if len(steps) == 1 and not conv_layers:
                    step = steps[0]
                    if step.backend in getattr(scheduler, "backends", {}):
                        simple_backend = step.backend
                        simple_gates = plan.gates[step.start : step.end]

            if est is not None and coeff_backup is not None and est.coeff != coeff_backup:
                est.coeff = coeff_backup
            if circuit is not None and original_ssd is not None:
                circuit.ssd = copy.deepcopy(original_ssd)

        def _run_once() -> Dict[str, Any]:
            if use_quick:
                rec = self.run_quasar(circuit, engine, backend=backend, quick=True)
                self.results.pop()
                return rec
            if simple_backend is not None and simple_gates is not None:
                if circuit is not None and original_ssd is not None:
                    circuit.ssd = copy.deepcopy(original_ssd)
                sim = type(scheduler.backends[simple_backend])()
                sim.load(circuit.num_qubits)
                tracemalloc.start()
                start_run = time.perf_counter()
                for g in simple_gates:
                    sim.apply_gate(g.gate, g.qubits, g.params)
                result = sim.extract_ssd()
                run_time = time.perf_counter() - start_run
                result = result if result is not None else getattr(circuit, "ssd", None)
                _, run_peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                backend_choice_name = getattr(simple_backend, "name", str(simple_backend))
                return {
                    "framework": "quasar",
                    "prepare_time": 0.0,
                    "run_time": run_time,
                    "total_time": run_time,
                    "prepare_peak_memory": 0,
                    "run_peak_memory": int(run_peak_memory),
                    "result": result,
                    "failed": False,
                    "backend": backend_choice_name,
                }
            assert plan is not None
            if circuit is not None and original_ssd is not None:
                circuit.ssd = copy.deepcopy(original_ssd)
            tracemalloc.start()
            start_run = time.perf_counter()
            result = scheduler.run(circuit, plan, instrument=False)
            run_time = time.perf_counter() - start_run
            _, run_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            backend_choice_name = None
            if hasattr(result, "partitions") and getattr(result, "partitions"):
                backend_obj = result.partitions[0].backend
                backend_choice_name = getattr(backend_obj, "name", str(backend_obj))
            return {
                "framework": "quasar",
                "prepare_time": 0.0,
                "run_time": run_time,
                "total_time": run_time,
                "prepare_peak_memory": 0,
                "run_peak_memory": int(run_peak_memory),
                "result": result,
                "failed": False,
                "backend": backend_choice_name,
            }

        start = time.perf_counter()
        for i in range(repetitions):
            try:
                if run_timeout is not None:
                    with _time_limit(run_timeout):
                        rec = _run_once()
                else:
                    rec = _run_once()
            except TimeoutError:
                failures.append(
                    f"run {i + 1} timed out after {run_timeout} seconds"
                )
                continue
            except Exception as exc:  # pragma: no cover - safety net
                failures.append(f"run {i + 1} failed: {exc}")
                continue

            if rec.get("failed"):
                failures.append(
                    f"run {i + 1} failed: {rec.get('error', 'unknown error')}"
                )
                continue

            records.append(rec)
            if timeout is not None and (time.perf_counter() - start) > timeout:
                break

        if not records:
            raise RuntimeError(f"no runs executed: {failures}")

        summary: Dict[str, Any] = {
            "framework": "quasar",
            "repetitions": len(records),
            "backend": records[0].get("backend"),
        }
        if failures:
            summary["failed_runs"] = failures
            summary["comment"] = (
                f"{len(failures)} run(s) failed and were excluded from statistics"
            )
        for m in metrics:
            values = [r[m] for r in records]
            summary[f"{m}_mean"] = statistics.fmean(values)
            summary[f"{m}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )

        summary["result"] = records[-1].get("result") if records else None

        self.results.append(summary)
        return summary

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
