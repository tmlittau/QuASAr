from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.bench_utils.paper_figures import CircuitSpec
from benchmarks.bench_utils import theoretical_estimation_runner as runner
from benchmarks.bench_utils.theoretical_estimation_utils import EstimateRecord
from quasar.cost import CostEstimator
from quasar.planner import Planner


class FakeCircuit:
    def __init__(self, gate_count: int):
        self.gates = [SimpleNamespace(gate="H", qubits=(0,)) for _ in range(gate_count)]
        self.num_qubits = 1
        self.use_classical_simplification = False

    def enable_classical_simplification(self) -> None:
        self.use_classical_simplification = True

    def disable_classical_simplification(self) -> None:
        self.use_classical_simplification = False


class FakeAnalyzer:
    def __init__(self, circuit, estimator):  # noqa: D401 - mimic real signature
        self.circuit = circuit
        self.estimator = estimator

    def resource_estimates(self):  # noqa: D401 - mimic real method
        return {}


def _make_spec(name: str, gate_count: int) -> CircuitSpec:
    def builder(_n_qubits: int, *, gate_count: int = gate_count) -> FakeCircuit:
        return FakeCircuit(gate_count)

    return CircuitSpec(name, builder, (1,), {"gate_count": gate_count})


def _run_estimate(
    monkeypatch: pytest.MonkeyPatch,
    gate_count: int,
    *,
    enable_large_planner: bool = True,
    threshold: int = 50,
    overrides: dict[str, object] | None = None,
):
    spec = _make_spec("fake", gate_count)
    estimator = CostEstimator()
    base_planner = Planner(estimator=estimator, perf_prio="time")

    captured: dict[str, object] = {}

    def fake_quasar_record(**kwargs):
        planner = kwargs["planner"]
        captured["planner"] = planner
        return EstimateRecord(
            circuit=spec.name,
            qubits=1,
            framework="quasar",
            backend="stub",
            supported=True,
            time_ops=1.0,
            memory_bytes=1.0,
            note="base",
        )

    monkeypatch.setattr(runner, "CircuitAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(runner, "_estimate_quasar_record", fake_quasar_record)

    records = runner.estimate_circuit(
        spec,
        1,
        estimator,
        (),
        base_planner,
        enable_large_planner=enable_large_planner,
        large_gate_threshold=threshold,
        large_planner_kwargs=overrides,
    )
    return records, captured, base_planner


def test_large_circuit_uses_tuned_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {"batch_size": 16, "horizon": 256, "quick_max_gates": 500}
    records, captured, base_planner = _run_estimate(
        monkeypatch,
        gate_count=100,
        threshold=50,
        overrides=overrides,
    )
    planner = captured["planner"]
    assert planner is not base_planner
    assert planner.batch_size == overrides["batch_size"]
    assert planner.horizon == overrides["horizon"]
    assert planner.quick_max_gates == overrides["quick_max_gates"]
    quasar_note = next(rec.note for rec in records if rec.framework == "quasar")
    assert quasar_note and "tuned planner" in quasar_note


def test_small_circuit_retains_full_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    records, captured, base_planner = _run_estimate(
        monkeypatch,
        gate_count=20,
        threshold=50,
    )
    assert captured["planner"] is base_planner
    quasar_note = next(rec.note for rec in records if rec.framework == "quasar")
    assert quasar_note == "base"


def test_large_planner_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    records, captured, base_planner = _run_estimate(
        monkeypatch,
        gate_count=100,
        threshold=50,
        enable_large_planner=False,
    )
    assert captured["planner"] is base_planner
    quasar_note = next(rec.note for rec in records if rec.framework == "quasar")
    assert quasar_note == "base"
