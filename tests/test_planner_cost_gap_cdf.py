import numpy as np
import pytest
from functools import lru_cache

from quasar.circuit import Circuit
from quasar.planner import Planner, _simulation_cost, _supported_backends
from benchmarks.circuits import (
    ghz_circuit,
    _qft_spec,
    w_state_circuit,
    grover_circuit,
)


def circuits() -> dict[str, Circuit]:
    ghz = ghz_circuit(4)
    qft = Circuit(_qft_spec(4), use_classical_simplification=False)
    qft_on_ghz = Circuit(list(ghz.gates) + _qft_spec(4), use_classical_simplification=False)
    return {
        "ghz4": ghz,
        "qft4": qft,
        "qft_on_ghz4": qft_on_ghz,
        "w4": w_state_circuit(4),
        "grover4": grover_circuit(4, 1),
    }


def cost_gap(circuit: Circuit) -> float:
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
            n1 = sum(
                1
                for g in seg
                if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
            )
            n2 = ng - n1 - nm
            for backend in allowed:
                cost = _simulation_cost(planner.estimator, backend, nq, n1, n2, nm).time
                best = min(best, cost + dfs(j))
        return best

    oracle_cost = dfs(0)
    return (dp_cost - oracle_cost) / oracle_cost


def test_planner_cost_gap_cdf() -> None:
    gaps = np.array([cost_gap(c) for c in circuits().values()])
    quantiles = np.quantile(gaps, [0.25, 0.5, 0.75])
    expected = np.array([-0.37340061553843984, 0.0, 17.05337647088215])
    assert quantiles == pytest.approx(expected, rel=1e-6, abs=1e-6)
