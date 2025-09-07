import math
import pytest

from quasar.circuit import Circuit
from quasar.cost import Backend, Cost, CostEstimator, ConversionEstimate
from quasar.planner import Planner
from quasar import config


def test_statevector_cost_regression():
    est = CostEstimator()
    cost = est.statevector(num_qubits=3, num_1q_gates=1, num_2q_gates=1, num_meas=1)
    assert cost.time == 24.0
    assert cost.memory == 160.0
    assert math.isclose(cost.log_depth, math.log2(3))


def test_tableau_cost_regression():
    est = CostEstimator()
    cost = est.tableau(
        num_qubits=2, num_gates=1, phase_bits=True, num_meas=1
    )
    assert cost.time == 16.0
    assert cost.memory == 2.625
    assert cost.log_depth == 1.0


def test_mps_cost_regression():
    est = CostEstimator()
    cost = est.mps(num_qubits=4, num_1q_gates=1, num_2q_gates=1, chi=2, svd=True)
    assert cost.time == pytest.approx(54.6666666667)
    assert cost.memory == 20.0
    assert cost.log_depth == 2.0


def test_dd_cost_regression():
    est = CostEstimator()
    cost = est.decision_diagram(num_gates=1, frontier=2)
    assert cost.time == 0.1
    assert cost.memory == pytest.approx(3.84)
    assert cost.log_depth == 1.0


@pytest.mark.parametrize(
    "target,expected",
    [
        (Backend.STATEVECTOR, 29.0),
        (Backend.TABLEAU, 21.0),
        (Backend.MPS, 25.0),
        (Backend.DECISION_DIAGRAM, 17.0),
    ],
)
def test_conversion_full_regression(target, expected):
    est = CostEstimator()
    res = est.conversion(
        Backend.STATEVECTOR,
        target,
        num_qubits=2,
        rank=4,
        frontier=2,
    )
    assert res.primitive == "Full"
    assert res.cost.time == expected
    assert res.cost.memory == 4.0


def test_conversion_lw_regression():
    est = CostEstimator()
    res = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=5,
        rank=4,
        frontier=0,
    )
    assert res.primitive == "LW"
    assert res.cost.time == 165.0
    assert res.cost.memory == 32.0


def test_conversion_st_regression():
    est = CostEstimator()
    res = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=5,
        rank=2,
        frontier=0,
    )
    assert res.primitive == "ST"
    assert res.cost.time == 157.0
    assert res.cost.memory == 32.0


def test_conversion_b2b_known_cost():
    coeff = {
        "lw_extract": 1000.0,
        "st_stage": 1000.0,
        "full_extract": 1000.0,
        "conversion_base": 0.0,
        "ingest_mps": 0.0,
        "b2b_svd": 1.0,
        "b2b_copy": 1.0,
    }
    est = CostEstimator(coeff=coeff)
    res = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=2,
        frontier=0,
    )
    assert res.primitive == "B2B"
    assert res.cost.time == 16.0
    assert res.cost.memory == 8.0


@pytest.mark.parametrize("backend", list(Backend))
def test_planner_picks_expected_backend(backend, monkeypatch):
    class Est(CostEstimator):
        def statevector(self, *args, **kwargs):
            mem = 1.0 if backend == Backend.STATEVECTOR else 1000.0
            return Cost(time=1.0, memory=mem)

        def tableau(self, *args, **kwargs):
            mem = 1.0 if backend == Backend.TABLEAU else 1000.0
            return Cost(time=1.0, memory=mem)

        def mps(self, *args, **kwargs):
            mem = 1.0 if backend == Backend.MPS else 1000.0
            return Cost(time=1.0, memory=mem)

        def decision_diagram(self, *args, **kwargs):
            mem = 1.0 if backend == Backend.DECISION_DIAGRAM else 1000.0
            return Cost(time=1.0, memory=mem)

        def conversion(self, *args, **kwargs):
            return ConversionEstimate("B2B", Cost(time=0.0, memory=0.0))

    est = Est()
    est.chi_max = 4

    if backend == Backend.TABLEAU:
        circ = Circuit([
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ])
    else:
        circ = Circuit([
            {"gate": "H", "qubits": [0]},
            {"gate": "T", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ])

    if backend == Backend.DECISION_DIAGRAM:
        circ.sparsity = 1.0
        circ.phase_rotation_diversity = 0
        circ.amplitude_rotation_diversity = 0
        monkeypatch.setattr(config.DEFAULT, "dd_sparsity_threshold", 0.0)
        monkeypatch.setattr(config.DEFAULT, "dd_nnz_threshold", 10_000_000)
        monkeypatch.setattr(config.DEFAULT, "dd_phase_rotation_diversity_threshold", 10_000_000)
        monkeypatch.setattr(config.DEFAULT, "dd_amplitude_rotation_diversity_threshold", 10_000_000)
        monkeypatch.setattr(config.DEFAULT, "dd_metric_threshold", 0.0)

    planner = Planner(estimator=est)
    result = planner.plan(circ)
    assert result.final_backend == backend
