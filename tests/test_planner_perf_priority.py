from quasar.circuit import Circuit
from quasar.cost import Cost, CostEstimator, ConversionEstimate
from quasar.planner import Backend, Planner


class MemoryEstimator(CostEstimator):
    """Estimator favouring low-memory backends."""

    def __init__(self):
        super().__init__(chi_max=4)

    def statevector(self, num_qubits, num_1q_gates, num_2q_gates, num_meas):
        return Cost(time=1.0, memory=100.0)

    def mps(self, num_qubits, num_1q_gates, num_2q_gates, chi, *, svd=False):
        return Cost(time=10.0, memory=10.0)

    def decision_diagram(self, num_gates, frontier):
        return Cost(time=1000.0, memory=1000.0)

    def conversion(self, *args, **kwargs):
        return ConversionEstimate("b2b", Cost(time=0.0, memory=0.0))


def _example_circuit() -> Circuit:
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ]
    return Circuit.from_dict(gates)


def test_memory_priority_avoids_high_mem_backend():
    circ = _example_circuit()
    planner = Planner(MemoryEstimator())
    result = planner.plan(circ)
    assert result.steps[0].backend == Backend.MPS


def test_time_priority_chooses_fast_backend():
    circ = _example_circuit()
    planner = Planner(MemoryEstimator(), perf_prio="time")
    result = planner.plan(circ)
    assert result.steps[0].backend == Backend.STATEVECTOR

