import time
from quasar import Circuit, Planner
from quasar.planner import _simulation_cost, _add_cost
from quasar.cost import Cost


def _build_circuit(num_gates: int) -> Circuit:
    gates = []
    for i in range(num_gates):
        if i % 3 == 0:
            gates.append({"gate": "H", "qubits": [0]})
        elif i % 3 == 1:
            gates.append({"gate": "CX", "qubits": [0, 1]})
        else:
            gates.append({"gate": "T", "qubits": [1]})
    return Circuit.from_dict(gates)


def _plan_cost(planner: Planner, circuit: Circuit, steps):
    gates = circuit.gates
    n = len(gates)
    prefix = [set() for _ in range(n + 1)]
    run = set()
    for i, g in enumerate(gates, start=1):
        run |= set(g.qubits)
        prefix[i] = run.copy()
    future = [set() for _ in range(n + 1)]
    run.clear()
    for i in range(n - 1, -1, -1):
        run |= set(gates[i].qubits)
        future[i] = run.copy()
    boundaries = [prefix[i] & future[i] for i in range(n + 1)]

    total = Cost(0.0, 0.0)
    prev_backend = None
    for step in steps:
        segment = gates[step.start:step.end]
        qubits = {q for g in segment for q in g.qubits}
        sim = _simulation_cost(planner.estimator, step.backend, len(qubits), len(segment))
        conv = Cost(0.0, 0.0)
        if prev_backend is not None and prev_backend != step.backend:
            boundary = boundaries[step.start]
            if boundary:
                rank = min(2 ** len(boundary), 2 ** 8)
                frontier = len(boundary)
                conv_est = planner.estimator.conversion(
                    prev_backend,
                    step.backend,
                    num_qubits=len(boundary),
                    rank=rank,
                    frontier=frontier,
                )
                conv = conv_est.cost
        total = _add_cost(_add_cost(total, conv), sim)
        prev_backend = step.backend
    return total


def test_batch_pruning_speed_and_quality():
    circuit = _build_circuit(40)

    start = time.perf_counter()
    base = Planner(top_k=4, batch_size=1, quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None).plan(circuit)
    t_base = time.perf_counter() - start
    cost_base = _plan_cost(
        Planner(top_k=4, batch_size=1, quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None),
        circuit,
        base.steps,
    ).time

    start = time.perf_counter()
    fast = Planner(top_k=1, batch_size=5, quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None).plan(circuit)
    t_fast = time.perf_counter() - start
    cost_fast = _plan_cost(
        Planner(top_k=1, batch_size=5, quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None),
        circuit,
        fast.steps,
    ).time

    assert t_fast < t_base
    assert cost_fast <= cost_base * 1.2
