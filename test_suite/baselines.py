from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import is optional
    from qiskit import QuantumCircuit


@dataclass
class RunnerResult:
    """Container for subprocess execution metadata."""

    ok: bool
    runtime: float
    peak_mem: Optional[int]
    timed_out: bool
    error: Optional[str]
    backend: str
    mode: str


def _get_peak_rss_bytes() -> Optional[int]:
    """Best-effort resident set size probe.

    Returns ``None`` if :mod:`psutil` is unavailable or if the query fails.
    """

    try:  # pragma: no cover - psutil is optional
        import os

        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss)
    except Exception:
        return None


def _subprocess_worker(
    conn: Connection, target: Any, args: Tuple[Any, ...]
) -> None:
    try:
        start = time.perf_counter()
        target(*args)
        runtime = time.perf_counter() - start
        peak = _get_peak_rss_bytes()
        conn.send(
            {
                "ok": True,
                "runtime": runtime,
                "peak_mem": peak,
                "error": None,
                "mode": "measured",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive branch
        conn.send(
            {
                "ok": False,
                "runtime": 0.0,
                "peak_mem": None,
                "error": f"{type(exc).__name__}: {exc}",
                "mode": "error",
            }
        )
    finally:
        conn.close()


def _run_in_subprocess(
    target: Any, args: Tuple[Any, ...], timeout_sec: float, backend_name: str
) -> RunnerResult:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_subprocess_worker,
        args=(child_conn, target, args),
        daemon=True,
    )
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        try:
            proc.terminate()
        finally:
            proc.join(1.0)
        return RunnerResult(
            ok=False,
            runtime=0.0,
            peak_mem=None,
            timed_out=True,
            error=None,
            backend=backend_name,
            mode="timeout",
        )
    if parent_conn.poll():
        payload = parent_conn.recv()
        return RunnerResult(
            ok=payload["ok"],
            runtime=float(payload["runtime"]),
            peak_mem=payload.get("peak_mem"),
            timed_out=False,
            error=payload.get("error"),
            backend=backend_name,
            mode=payload.get("mode", "measured"),
        )
    return RunnerResult(
        ok=False,
        runtime=0.0,
        peak_mem=None,
        timed_out=False,
        error="no result (process exited without response)",
        backend=backend_name,
        mode="error",
    )


_ES_HAS_PARAM = {
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
    "crx",
    "cry",
    "crz",
    "cp",
    "rxx",
    "ryy",
    "rzx",
}


def qiskit_circuit_has_arbitrary_rotations(qc: "QuantumCircuit") -> bool:
    """Detect gates that typically make Aer extended stabilizer bail out."""

    for instruction, _qargs, _cargs in qc.data:
        if instruction.name.lower() in _ES_HAS_PARAM:
            return True
    return False


def _dd_worker(qc: "QuantumCircuit") -> None:
    from mqt.ddsim import DDSIMProvider  # type: ignore

    backend = DDSIMProvider().get_backend("statevector_simulator")
    job = backend.run(qc, shots=0)
    job.result()


def run_dd_time_and_mem(
    qc: "QuantumCircuit", timeout_sec: float = 60.0
) -> Dict[str, Any]:
    result = _run_in_subprocess(_dd_worker, (qc,), timeout_sec, backend_name="dd")
    return {
        "backend": "dd",
        "runtime": result.runtime,
        "peak_mem": result.peak_mem,
        "timed_out": result.timed_out,
        "error": None if result.ok or result.timed_out else result.error,
        "mode": result.mode,
    }


def _extended_stabilizer_worker(qc: "QuantumCircuit") -> None:
    try:
        from qiskit_aer import Aer  # type: ignore

        backend = Aer.get_backend("aer_simulator_extended_stabilizer")
    except Exception:
        from qiskit_aer import AerSimulator  # type: ignore

        backend = AerSimulator(method="extended_stabilizer")
    job = backend.run(qc, shots=0)
    job.result()


def run_extended_stabilizer_time_and_mem(
    qc: "QuantumCircuit", timeout_sec: float = 60.0
) -> Dict[str, Any]:
    if qiskit_circuit_has_arbitrary_rotations(qc):
        return {
            "backend": "extended_stabilizer",
            "runtime": 0.0,
            "peak_mem": None,
            "timed_out": False,
            "error": "unsupported: arbitrary rotations present",
            "mode": "unsupported",
        }
    result = _run_in_subprocess(
        _extended_stabilizer_worker,
        (qc,),
        timeout_sec,
        backend_name="extended_stabilizer",
    )
    return {
        "backend": "extended_stabilizer",
        "runtime": result.runtime,
        "peak_mem": result.peak_mem,
        "timed_out": result.timed_out,
        "error": None if result.ok or result.timed_out else result.error,
        "mode": result.mode,
    }
