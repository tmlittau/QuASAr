import math
from quasar.cost import CostEstimator, Backend
from quasar.circuit import Circuit
from quasar.method_selector import MethodSelector
from quasar.planner import Planner


def test_method_selector_memory_limit():
    est = CostEstimator()
    selector = MethodSelector(est)
    gates = [{"gate": "T", "qubits": [0]}] + [
        {"gate": "CX", "qubits": [i, i + 1]} for i in range(14)
    ]
    circ = Circuit.from_dict(gates, use_classical_simplification=False)
    sv = est.statevector(15, 0, 14, 0)
    mps = est.mps(15, 0, 14, chi=4, svd=True)
    max_mem = (sv.memory + mps.memory) / 2
    backend, _ = selector.select(
        circ.gates, 15, max_memory=max_mem, target_accuracy=0.0
    )
    assert backend == Backend.MPS


def test_planner_forwarding_constraints():
    class DummySelector(MethodSelector):
        def __init__(self, est):
            super().__init__(est)
            self.kwargs = None

        def select(self, gates, num_qubits, **kwargs):
            self.kwargs = kwargs
            cost = self.estimator.statevector(num_qubits, 0, 0, 0)
            return Backend.STATEVECTOR, cost

    est = CostEstimator()
    selector = DummySelector(est)
    planner = Planner(estimator=est, selector=selector)
    circ = Circuit.from_dict([{"gate": "H", "qubits": [0]}], use_classical_simplification=False)
    planner.plan(circ, target_accuracy=0.91, max_time=100.0)
    assert selector.kwargs is not None
    assert math.isclose(selector.kwargs["target_accuracy"], 0.91)
    assert math.isclose(selector.kwargs["max_time"], 100.0)
