import time
from functools import lru_cache

import pytest

from quasar.circuit import Circuit
from quasar.planner import Planner, _simulation_cost, _supported_backends

BASELINES = {
    "bell": {"oracle": 20.0, "dp": 20.0, "greedy": 20.0},
    "ghz3": {"oracle": 36.0, "dp": 36.0, "greedy": 36.0},
}


def compute_metrics(circuit: Circuit) -> dict[str, tuple[float, float]]:
    planner = Planner()
    gates = circuit.gates

    t0 = time.perf_counter()
    res = planner.plan(circuit)
    t1 = time.perf_counter()
    dp_cost = res.table[-1][res.final_backend].cost.time
    dp_time = t1 - t0

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

    t2 = time.perf_counter()
    oracle_cost = dfs(0)
    t3 = time.perf_counter()
    oracle_time = t3 - t2

    t4 = time.perf_counter()
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
    t5 = time.perf_counter()
    greedy_time = t5 - t4

    return {
        "oracle": (oracle_cost, oracle_time),
        "dp": (dp_cost, dp_time),
        "greedy": (greedy_cost, greedy_time),
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
    for method, (cost, runtime) in metrics.items():
        expected_cost = BASELINES[name][method]
        assert cost == expected_cost
        assert runtime > 0
    dp_time = metrics["dp"][1]
    for method, (_, runtime) in metrics.items():
        if method != "dp":
            assert dp_time > runtime
