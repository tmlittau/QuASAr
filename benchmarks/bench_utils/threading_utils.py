from __future__ import annotations

"""Shared concurrency helpers for benchmark scripts."""

import os
import threading
from typing import Any, Callable, TypeVar

from quasar import SimulationEngine

__all__ = ["resolve_worker_count", "thread_engine", "with_thread_engine"]


_THREAD_LOCAL = threading.local()

_T = TypeVar("_T")


def resolve_worker_count(max_workers: int | None, task_count: int) -> int:
    """Return an appropriate worker count bounded by ``task_count``."""

    if task_count <= 0:
        return 0
    if max_workers is not None:
        try:
            workers = int(max_workers)
        except (TypeError, ValueError):
            workers = 0
        else:
            if workers < 0:
                workers = 0
        if workers:
            return min(workers, task_count)
        return 1 if task_count else 0
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, task_count)


def thread_engine(factory: Callable[[], SimulationEngine] | None = None) -> SimulationEngine:
    """Return a thread-local :class:`~quasar.SimulationEngine` instance."""

    engine = getattr(_THREAD_LOCAL, "engine", None)
    if engine is None:
        if factory is None:
            factory = SimulationEngine
        engine = factory()
        _THREAD_LOCAL.engine = engine
    return engine


def with_thread_engine(func: Callable[[SimulationEngine, _T], Any]) -> Callable[[_T], Any]:
    """Wrap ``func`` so it receives a thread-local engine as the first argument."""

    def _wrapper(arg: _T) -> Any:
        engine = thread_engine()
        return func(engine, arg)

    return _wrapper
