from __future__ import annotations

import logging

import pytest

from quasar.cost import Backend
from quasar.method_selector import NoFeasibleBackendError

import benchmarks.paper_figures as paper_figures
from benchmarks import circuits as circuit_lib


@pytest.fixture(autouse=True)
def _reset_logger_level():
    """Ensure benchmark logging does not leak configuration between tests."""

    logger = logging.getLogger(paper_figures.__name__)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.setLevel(previous)


def test_collect_backend_data_marks_statevector_unsupported(monkeypatch, recwarn, caplog):
    """Statevector runs exceeding the width limit should be marked unsupported."""

    monkeypatch.setattr(paper_figures, "STATEVECTOR_MAX_QUBITS", 4)

    calls: list[Backend | None] = []

    def fake_run(self, circuit, engine, *, backend=None, **kwargs):
        calls.append(backend)
        backend_name = getattr(backend, "name", backend)
        return {
            "prepare_time": 0.0,
            "run_time": 0.0,
            "total_time": 0.0,
            "prepare_peak_memory": 0,
            "run_peak_memory": 0,
            "result": None,
            "failed": False,
            "backend": backend_name,
        }

    monkeypatch.setattr(paper_figures.BenchmarkRunner, "run_quasar_multiple", fake_run)

    spec = paper_figures.CircuitSpec(
        "grover_many_controls",
        circuit_lib.grover_circuit,
        (5,),
        {"n_iterations": 1},
    )

    caplog.set_level(logging.WARNING)

    forced, auto = paper_figures.collect_backend_data(
        [spec],
        [Backend.STATEVECTOR],
        repetitions=1,
    )

    assert len(forced) == 1
    row = forced.iloc[0]
    assert bool(row["unsupported"])
    assert row["framework"] == Backend.STATEVECTOR.name
    assert row["backend"] == Backend.STATEVECTOR.name
    assert row["actual_qubits"] > paper_figures.STATEVECTOR_MAX_QUBITS
    assert "exceeding statevector limit" in row["error"]
    assert Backend.STATEVECTOR not in calls
    assert calls == []

    assert auto.shape[0] == 1
    row_auto = auto.iloc[0]
    assert row_auto["mode"] == "auto"
    assert row_auto["backend"] == Backend.MPS.name
    assert row_auto["unsupported"]
    assert "ancilla expansion" in row_auto["error"]

    assert not recwarn.list
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_collect_backend_data_marks_mps_ancilla_unsupported(monkeypatch, recwarn, caplog):
    """Ancilla-heavy circuits should not invoke Aer MPS during benchmarks."""

    monkeypatch.setattr(paper_figures, "STATEVECTOR_MAX_QUBITS", 30)

    calls: list[Backend | None] = []

    def fake_run(self, circuit, engine, *, backend=None, **kwargs):
        calls.append(backend)
        backend_name = getattr(backend, "name", backend)
        return {
            "prepare_time": 0.0,
            "run_time": 0.0,
            "total_time": 0.0,
            "prepare_peak_memory": 0,
            "run_peak_memory": 0,
            "result": None,
            "failed": False,
            "backend": backend_name,
        }

    monkeypatch.setattr(paper_figures.BenchmarkRunner, "run_quasar_multiple", fake_run)

    spec = paper_figures.CircuitSpec(
        "grover_large",
        lambda n, *, iterations=2: paper_figures._large_grover_circuit(
            n, iterations=iterations
        ),
        (20,),
        {"iterations": 2},
    )

    caplog.set_level(logging.INFO)

    forced, auto = paper_figures.collect_backend_data(
        [spec],
        [Backend.MPS],
        repetitions=1,
    )

    assert len(forced) == 1
    row = forced.iloc[0]
    assert bool(row["unsupported"])
    assert row["framework"] == Backend.MPS.name
    assert row["backend"] == Backend.MPS.name
    assert "ancilla expansion" in row["error"]
    assert "exceeds statevector limit" in row["comment"]
    assert "dense memory" in row["comment"]
    assert Backend.MPS not in calls
    assert calls == []

    assert auto.shape[0] == 1
    assert auto.iloc[0]["mode"] == "auto"

    assert not recwarn.list
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_collect_backend_data_records_no_backend(monkeypatch, caplog):
    """Automatic failures without a feasible backend should be recorded."""

    def fake_run(self, circuit, engine, *, backend=None, **kwargs):
        if backend is None:
            raise NoFeasibleBackendError(
                "No simulation backend satisfies the given constraints"
            )
        backend_name = getattr(backend, "name", backend)
        return {
            "prepare_time": 0.0,
            "run_time": 0.0,
            "total_time": 0.0,
            "prepare_peak_memory": 0,
            "run_peak_memory": 0,
            "result": None,
            "failed": False,
            "backend": backend_name,
        }

    monkeypatch.setattr(paper_figures.BenchmarkRunner, "run_quasar_multiple", fake_run)

    spec = paper_figures.CircuitSpec(
        "qft_small",
        circuit_lib.qft_circuit,
        (4,),
    )

    caplog.set_level(logging.INFO)

    forced, auto = paper_figures.collect_backend_data(
        [spec],
        [Backend.STATEVECTOR],
        repetitions=1,
    )

    assert len(forced) == 1
    assert forced.iloc[0]["backend"] == Backend.STATEVECTOR.name

    assert auto.shape[0] == 1
    row_auto = auto.iloc[0]
    assert row_auto["unsupported"]
    assert row_auto["backend"] is None
    assert row_auto["mode"] == "auto"
    assert (
        row_auto["comment"]
        == "No simulation backend satisfies the given constraints"
    )

    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
