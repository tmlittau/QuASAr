"""Tests covering diagnostic output of the method selector and planner."""

from __future__ import annotations

import pytest

import json

from quasar.circuit import Circuit, Gate
from quasar.cost import CostEstimator, Backend
from quasar.method_selector import MethodSelector
from quasar.metrics import (
    export_fragment_metrics_csv,
    export_fragment_metrics_json,
)
from quasar.planner import Planner


def build_test_circuit() -> Circuit:
    """Return a small circuit exercising multiple backend heuristics."""

    gates = [
        Gate("T", [0]),
        Gate("H", [1]),
        Gate("CX", [0, 2]),
    ]
    return Circuit(gates, use_classical_simplification=False)


def test_method_selector_populates_diagnostics() -> None:
    """``MethodSelector.select`` should annotate backend diagnostics."""

    selector = MethodSelector()
    circuit = build_test_circuit()
    diag: dict[str, object] = {}

    backend, cost = selector.select(
        circuit.gates,
        circuit.num_qubits,
        sparsity=1.0,
        phase_rotation_diversity=0,
        amplitude_rotation_diversity=0,
        diagnostics=diag,
    )

    assert diag["selected_backend"] == backend
    assert diag["selected_cost"] == cost

    backends = diag["backends"]
    assert Backend.TABLEAU in backends
    assert Backend.MPS in backends
    assert Backend.STATEVECTOR in backends

    # Tableau should be rejected because the fragment is non-Clifford.
    assert backends[Backend.TABLEAU]["feasible"] is False
    assert "non-clifford gates" in backends[Backend.TABLEAU]["reasons"]

    # The mildly non-local entangler should keep MPS in the candidate set with
    # a penalty rather than rejecting it outright.
    assert backends[Backend.MPS]["feasible"] is False
    assert "requires target fidelity" in backends[Backend.MPS]["reasons"]
    assert backends[Backend.MPS]["long_range_fraction"] == pytest.approx(1.0)
    assert backends[Backend.MPS]["long_range_extent"] == pytest.approx(0.5)
    assert backends[Backend.MPS]["max_interaction_distance"] == 2

    metrics = diag["metrics"]
    assert metrics["local"] is False
    assert metrics["mps_long_range_fraction"] == pytest.approx(1.0)
    assert metrics["mps_long_range_extent"] == pytest.approx(0.5)
    assert metrics["mps_max_interaction_distance"] == 2
    assert metrics["entanglement_entropy"] >= 0.0
    fragments = metrics["fragments"]
    assert fragments
    fragment_entry = fragments[0]
    assert fragment_entry["scope"] == "fragment"
    assert fragment_entry["local"] is False
    assert 0.0 <= fragment_entry["sparsity"] <= 1.0
    assert fragment_entry["long_range_fraction"] == pytest.approx(1.0)
    assert "rotation_density" in fragment_entry

    # The selected backend is marked accordingly in the diagnostics.
    assert backends[backend]["selected"] is True

    sv_entry = backends[Backend.STATEVECTOR]
    assert "modifiers" in sv_entry
    sv_mod = sv_entry["modifiers"]
    assert sv_mod["sparsity"] == pytest.approx(1.0)
    assert sv_mod["time_modifier"] >= sv_mod["memory_modifier"] >= 0.2


def test_planner_explain_surfaces_selection_diagnostics() -> None:
    """Planner diagnostics should include selector information."""

    estimator = CostEstimator()
    planner = Planner(
        estimator=estimator,
        quick_max_qubits=32,
        quick_max_gates=128,
        quick_max_depth=128,
    )
    circuit = build_test_circuit()

    plan = planner.plan(circuit, explain=True)
    assert plan.diagnostics is not None
    selector_diag = plan.diagnostics.backend_selection
    assert "single" in selector_diag
    assert selector_diag["single"]["selected_backend"] == plan.final_backend
    fragment_records = plan.diagnostics.fragment_metrics
    assert fragment_records
    record_dict = fragment_records[0].to_mapping()
    assert record_dict["backend"] == plan.final_backend.name
    assert record_dict["scope"] == "plan"
    assert "heuristic_sparsity" in record_dict
    dicts = plan.diagnostics.fragments_as_dicts()
    assert dicts[0]["scope"] == "plan"


def test_fragment_metric_exporters(tmp_path) -> None:
    estimator = CostEstimator()
    planner = Planner(
        estimator=estimator,
        quick_max_qubits=32,
        quick_max_gates=128,
        quick_max_depth=128,
    )
    circuit = build_test_circuit()
    plan = planner.plan(circuit, explain=True)
    assert plan.diagnostics is not None
    records = plan.diagnostics.fragment_metrics
    json_path = tmp_path / "fragments.json"
    csv_path = tmp_path / "fragments.csv"
    export_fragment_metrics_json(records, json_path)
    export_fragment_metrics_csv(records, csv_path)
    data = json.loads(json_path.read_text())
    assert data
    assert data[0]["scope"] == "plan"
    csv_lines = csv_path.read_text().strip().splitlines()
    assert csv_lines
    header = csv_lines[0].split(",")
    assert "backend" in header
