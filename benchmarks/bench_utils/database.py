"""SQLite storage for benchmark runs and theoretical estimates."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def _bool(value: Any | None) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _json(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


@dataclass(frozen=True)
class BenchmarkRun:
    """Metadata describing a single benchmark invocation."""

    id: str


class BenchmarkDatabase:
    """Persist benchmark results into a SQLite database."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.path), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialise()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._connection.close()

    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_run (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    parameters TEXT
                );

                CREATE TABLE IF NOT EXISTS benchmark (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    circuit_id TEXT NOT NULL,
                    circuit_display_name TEXT,
                    repetitions INTEGER NOT NULL,
                    qubits INTEGER NOT NULL,
                    run_timeout REAL,
                    memory_bytes INTEGER,
                    classical_simplification INTEGER NOT NULL,
                    include_baselines INTEGER NOT NULL,
                    quick INTEGER NOT NULL,
                    baseline_backends TEXT,
                    workers INTEGER,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES benchmark_run(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS simulation_run (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_id INTEGER NOT NULL,
                    framework TEXT NOT NULL,
                    backend TEXT,
                    mode TEXT,
                    repetitions INTEGER,
                    qubits INTEGER,
                    prepare_time_mean REAL,
                    prepare_time_std REAL,
                    run_time_mean REAL,
                    run_time_std REAL,
                    total_time_mean REAL,
                    total_time_std REAL,
                    prepare_peak_memory_mean REAL,
                    prepare_peak_memory_std REAL,
                    run_peak_memory_mean REAL,
                    run_peak_memory_std REAL,
                    unsupported INTEGER DEFAULT 0,
                    failed INTEGER DEFAULT 0,
                    timeout INTEGER,
                    comment TEXT,
                    error TEXT,
                    failed_runs TEXT,
                    partition_count INTEGER,
                    partition_total_subsystems INTEGER,
                    partition_unique_backends INTEGER,
                    partition_max_multiplicity INTEGER,
                    partition_mean_multiplicity REAL,
                    partition_backend_breakdown TEXT,
                    hierarchy_available INTEGER,
                    result_json TEXT,
                    extra TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(benchmark_id) REFERENCES benchmark(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS estimation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    circuit_id TEXT NOT NULL,
                    qubits INTEGER NOT NULL,
                    framework TEXT NOT NULL,
                    backend TEXT,
                    supported INTEGER NOT NULL,
                    time_ops REAL,
                    approx_seconds REAL,
                    memory_bytes REAL,
                    note TEXT,
                    method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    # ------------------------------------------------------------------
    def start_run(
        self, *, description: str | None = None, parameters: Mapping[str, Any] | None = None
    ) -> BenchmarkRun:
        run_id = str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO benchmark_run(id, description, parameters) VALUES (?, ?, ?)",
                (run_id, description, _json(parameters)),
            )
        return BenchmarkRun(id=run_id)

    # ------------------------------------------------------------------
    def create_benchmark(
        self,
        run: BenchmarkRun,
        *,
        circuit_id: str,
        circuit_display_name: str | None,
        repetitions: int,
        qubits: int,
        run_timeout: float | None,
        memory_bytes: int | None,
        classical_simplification: bool,
        include_baselines: bool,
        quick: bool,
        baseline_backends: Iterable[str] | None,
        workers: int | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        backends_json = _json(list(baseline_backends) if baseline_backends else None)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO benchmark (
                    run_id,
                    circuit_id,
                    circuit_display_name,
                    repetitions,
                    qubits,
                    run_timeout,
                    memory_bytes,
                    classical_simplification,
                    include_baselines,
                    quick,
                    baseline_backends,
                    workers,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    circuit_id,
                    circuit_display_name,
                    int(repetitions),
                    int(qubits),
                    float(run_timeout) if run_timeout is not None else None,
                    int(memory_bytes) if memory_bytes is not None else None,
                    _bool(classical_simplification),
                    _bool(include_baselines),
                    _bool(quick),
                    backends_json,
                    int(workers) if workers is not None else None,
                    _json(metadata),
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    def insert_simulation_run(
        self, benchmark_id: int, record: Mapping[str, Any], *, qubits: int | None = None
    ) -> None:
        known_fields = {
            "framework",
            "backend",
            "mode",
            "repetitions",
            "prepare_time_mean",
            "prepare_time_std",
            "run_time_mean",
            "run_time_std",
            "total_time_mean",
            "total_time_std",
            "prepare_peak_memory_mean",
            "prepare_peak_memory_std",
            "run_peak_memory_mean",
            "run_peak_memory_std",
            "unsupported",
            "failed",
            "timeout",
            "comment",
            "error",
            "failed_runs",
            "partition_count",
            "partition_total_subsystems",
            "partition_unique_backends",
            "partition_max_multiplicity",
            "partition_mean_multiplicity",
            "partition_backend_breakdown",
            "hierarchy_available",
            "result_json",
        }

        extra = {
            key: value
            for key, value in record.items()
            if key not in known_fields.union({"qubits"})
        }
        failed_runs = record.get("failed_runs")
        if isinstance(failed_runs, list):
            failed_runs_json = json.dumps(failed_runs)
        elif isinstance(failed_runs, str) or failed_runs is None:
            failed_runs_json = failed_runs
        else:
            failed_runs_json = json.dumps(failed_runs)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO simulation_run (
                    benchmark_id,
                    framework,
                    backend,
                    mode,
                    repetitions,
                    qubits,
                    prepare_time_mean,
                    prepare_time_std,
                    run_time_mean,
                    run_time_std,
                    total_time_mean,
                    total_time_std,
                    prepare_peak_memory_mean,
                    prepare_peak_memory_std,
                    run_peak_memory_mean,
                    run_peak_memory_std,
                    unsupported,
                    failed,
                    timeout,
                    comment,
                    error,
                    failed_runs,
                    partition_count,
                    partition_total_subsystems,
                    partition_unique_backends,
                    partition_max_multiplicity,
                    partition_mean_multiplicity,
                    partition_backend_breakdown,
                    hierarchy_available,
                    result_json,
                    extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(benchmark_id),
                    record.get("framework"),
                    record.get("backend"),
                    record.get("mode"),
                    record.get("repetitions"),
                    qubits if qubits is not None else record.get("qubits"),
                    record.get("prepare_time_mean"),
                    record.get("prepare_time_std"),
                    record.get("run_time_mean"),
                    record.get("run_time_std"),
                    record.get("total_time_mean"),
                    record.get("total_time_std"),
                    record.get("prepare_peak_memory_mean"),
                    record.get("prepare_peak_memory_std"),
                    record.get("run_peak_memory_mean"),
                    record.get("run_peak_memory_std"),
                    _bool(record.get("unsupported")),
                    _bool(record.get("failed")),
                    _bool(record.get("timeout")),
                    record.get("comment"),
                    record.get("error"),
                    failed_runs_json,
                    record.get("partition_count"),
                    record.get("partition_total_subsystems"),
                    record.get("partition_unique_backends"),
                    record.get("partition_max_multiplicity"),
                    record.get("partition_mean_multiplicity"),
                    record.get("partition_backend_breakdown"),
                    _bool(record.get("hierarchy_available")),
                    record.get("result_json"),
                    _json(extra) if extra else None,
                ),
            )

    # ------------------------------------------------------------------
    def insert_estimation(self, *, record: Mapping[str, Any], method: str | None = None) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO estimation (
                    circuit_id,
                    qubits,
                    framework,
                    backend,
                    supported,
                    time_ops,
                    approx_seconds,
                    memory_bytes,
                    note,
                    method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("circuit"),
                    record.get("qubits"),
                    record.get("framework"),
                    record.get("backend"),
                    _bool(record.get("supported")),
                    record.get("time_ops"),
                    record.get("approx_seconds"),
                    record.get("memory_bytes"),
                    record.get("note"),
                    method,
                ),
            )

    # ------------------------------------------------------------------
    def connection(self) -> sqlite3.Connection:
        return self._connection


@contextmanager
def open_database(path: Path) -> Iterator[BenchmarkDatabase]:
    db = BenchmarkDatabase(path)
    try:
        yield db
    finally:
        db.close()


__all__ = ["BenchmarkDatabase", "BenchmarkRun", "open_database"]

