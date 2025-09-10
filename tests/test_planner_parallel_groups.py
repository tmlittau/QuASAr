from quasar.circuit import Circuit
from quasar.cost import CostEstimator
from quasar.planner import Planner, _parallel_simulation_cost
from quasar.partitioner import Partitioner
from quasar.cost import Backend


def test_parallel_group_cost(monkeypatch):
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "T", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "T", "qubits": [1]},
    ])
    planner = Planner(CostEstimator())

    plan_parallel = planner.plan(circuit, backend=Backend.STATEVECTOR)
    step_parallel = plan_parallel.steps[0]
    groups = Partitioner().parallel_groups(circuit.gates)
    cost_parallel = _parallel_simulation_cost(
        planner.estimator, Backend.STATEVECTOR, groups
    )

    def serial_groups(self, gates):
        qubits = tuple(sorted({q for g in gates for q in g.qubits}))
        return [(qubits, list(gates))]

    serial = serial_groups(None, circuit.gates)  # compute single group
    cost_serial = _parallel_simulation_cost(
        planner.estimator, Backend.STATEVECTOR, serial
    )

    assert step_parallel.parallel == ((0,), (1,))
    assert cost_parallel.time < cost_serial.time
    assert cost_parallel.memory > cost_serial.memory
