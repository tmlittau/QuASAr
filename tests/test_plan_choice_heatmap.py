import json
from pathlib import Path

from benchmarks.circuits import ghz_circuit, w_state_circuit, random_hybrid_circuit
from benchmarks.partition_circuits import mixed_backend_subsystems, hybrid_dense_to_mps_circuit
from quasar.planner import Planner


def circuits():
    return {
        "GHZ_6": ghz_circuit(6),
        "WState_6": w_state_circuit(6),
        "RandomHybrid_6": random_hybrid_circuit(num_qubits=6, depth=6, seed=2),
        "MixedSubsystems": mixed_backend_subsystems(
            ghz_width=4, qaoa_width=4, qaoa_layers=2, random_width=4, seed=11
        ),
        "HybridDenseToMPS": hybrid_dense_to_mps_circuit(
            ghz_width=4, random_width=5, qaoa_width=5, qaoa_layers=3, seed=5
        ),
    }


def load_records():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "results"
        / "plan_choice_heatmap_results.json"
    )
    with path.open() as f:
        records = json.load(f)
    return {(r["circuit"], r["alpha"]): r["steps"] for r in records}


def test_plan_choice_heatmap():
    expected = load_records()
    for (name, alpha), steps in expected.items():
        circ = circuits()[name]
        planner = Planner(conversion_cost_multiplier=alpha)
        plan = planner.plan(circ)
        observed = [s.backend.name for s in plan.steps]
        assert observed == steps
