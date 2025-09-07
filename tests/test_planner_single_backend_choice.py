import pytest

from quasar.circuit import Circuit
from quasar.cost import Cost, CostEstimator
from quasar.planner import Planner


class OverheadEstimator(CostEstimator):
    """Estimator adding a constant cost to each simulation fragment."""

    def statevector(self, num_qubits, num_1q_gates, num_2q_gates, num_meas):
        return Cost(time=num_1q_gates + num_2q_gates + num_meas + 1, memory=0)

    def tableau(self, num_qubits, num_gates, **kwargs):
        return Cost(time=num_gates + 1, memory=0)

    def mps(self, num_qubits, num_1q_gates, num_2q_gates, chi, *, svd=False):
        return Cost(time=num_1q_gates + num_2q_gates + 1, memory=0)

    def decision_diagram(self, num_gates, frontier):
        return Cost(time=num_gates + 1, memory=0)

    def conversion(self, *args, **kwargs):
        from quasar.cost import ConversionEstimate

        return ConversionEstimate("b2b", Cost(time=0, memory=0))


@pytest.mark.parametrize(
    "gates",
    [
        # small circuit
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
        # moderate circuit exceeding default quick path qubit limit
        [{"gate": "H", "qubits": [i]} for i in range(13)],
    ],
)
def test_single_backend_when_cheapest(gates):
    circ = Circuit.from_dict(gates)
    planner = Planner(
        OverheadEstimator(),
    )
    result = planner.plan(circ)
    assert len(result.steps) == 1
    assert result.steps[0].start == 0
    assert result.steps[0].end == len(gates)
    assert circ.ssd.conversions == []
