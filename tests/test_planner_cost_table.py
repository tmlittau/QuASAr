from functools import lru_cache

import pytest

from quasar.circuit import Circuit
from quasar.planner import Planner, _simulation_cost, _supported_backends

BASELINES = {
    "bell": {"oracle": 23.96, "dp": 23.96, "greedy": 23.96},
    "ghz3": {"oracle": 42.6, "dp": 42.6, "greedy": 42.6},
}


def compute_metrics(circuit: Circuit) -> dict[str, float]:
    planner = Planner()
    gates = circuit.gates

    res = planner.plan(circuit)
    dp_cost = res.table[-1][res.final_backend].cost.time

    @lru_cache(None)
    def dfs(i: int) -> float:
        if i >= len(gates):
            return 0.0
        best = float("inf")
        for j in range(i + 1, len(gates) + 1):
            seg = gates[i:j]
            allowed = _supported_backends(
                seg,
                allow_tableau=True,
                estimator=planner.estimator,
                sparsity=circuit.sparsity,
                phase_rotation_diversity=circuit.phase_rotation_diversity,
                amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
            )
            nq = len({q for g in seg for q in g.qubits})
            ng = len(seg)
            nm = sum(1 for g in seg if g.gate.upper() in {"MEASURE", "RESET"})
            n1 = sum(1 for g in seg if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"})
            n2 = ng - n1 - nm
            for backend in allowed:
                cost = _simulation_cost(planner.estimator, backend, nq, n1, n2, nm).time
                best = min(best, cost + dfs(j))
        return best

    oracle_cost = dfs(0)

    greedy_cost = 0.0
    for gate in gates:
        seg = [gate]
        allowed = _supported_backends(
            seg,
            allow_tableau=True,
            estimator=planner.estimator,
            sparsity=circuit.sparsity,
            phase_rotation_diversity=circuit.phase_rotation_diversity,
            amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
        )
        nq = len({q for g in seg for q in g.qubits})
        nm = sum(1 for g in seg if g.gate.upper() in {"MEASURE", "RESET"})
        n1 = sum(1 for g in seg if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"})
        n2 = len(seg) - n1 - nm
        best = min(
            _simulation_cost(planner.estimator, b, nq, n1, n2, nm).time
            for b in allowed
        )
        greedy_cost += best

    return {
        "oracle": oracle_cost,
        "dp": dp_cost,
        "greedy": greedy_cost,
    }


def circuits() -> dict[str, Circuit]:
    return {
        "bell": Circuit(
            [
                {"gate": "H", "qubits": [0]},
                {"gate": "CX", "qubits": [0, 1]},
            ],
            use_classical_simplification=False,
        ),
        "ghz3": Circuit(
            [
                {"gate": "H", "qubits": [0]},
                {"gate": "CX", "qubits": [0, 1]},
                {"gate": "CX", "qubits": [1, 2]},
            ],
            use_classical_simplification=False,
        ),
    }


@pytest.mark.parametrize("name,circuit", list(circuits().items()))
def test_planner_cost_table(name: str, circuit: Circuit) -> None:
    metrics = compute_metrics(circuit)
    for method, cost in metrics.items():
        expected_cost = BASELINES[name][method]
        assert cost == expected_cost
