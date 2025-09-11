import inspect
import json
from pathlib import Path

from benchmarks.circuits import ghz_circuit, grover_circuit
from quasar.planner import Planner


def circuits():
    return {
        "GHZ_6": ghz_circuit(6),
        "Grover_3": grover_circuit(3, 1),
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
        kwargs = {"conversion_cost_multiplier": alpha}
        if "compare_pre_pass_costs" in inspect.signature(Planner).parameters:
            kwargs["compare_pre_pass_costs"] = True
        planner = Planner(**kwargs)
        plan = planner.plan(circ)
        observed = [s.backend.name for s in plan.steps]
        assert observed == steps
