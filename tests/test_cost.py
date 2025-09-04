from quasar import Backend, CostEstimator
import math


def test_statevector_scaling():
    est = CostEstimator()
    small = est.statevector(num_qubits=3, num_1q_gates=1, num_2q_gates=0, num_meas=0)
    large = est.statevector(num_qubits=4, num_1q_gates=1, num_2q_gates=0, num_meas=0)
    assert large.time == 2 * small.time
    assert large.memory == 2 * small.memory
    assert small.log_depth == math.log2(3)
    assert large.log_depth == math.log2(4)


def test_tableau_quadratic():
    est = CostEstimator()
    cost = est.tableau(num_qubits=5, num_gates=2)
    assert cost.time == 2 * 25
    assert cost.memory == 25


def test_mps_chi_dependence():
    est = CostEstimator()
    chi2 = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=2)
    chi4 = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=4)
    assert chi4.time == chi2.time * (4 ** 3) / (2 ** 3)
    assert chi4.memory == chi2.memory * (4 ** 2) / (2 ** 2)


def test_mps_gate_scaling():
    est = CostEstimator()
    one = est.mps(num_qubits=4, num_1q_gates=1, num_2q_gates=0, chi=4)
    two = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=4)
    assert two.time == one.time * 4


def test_mps_svd_cost():
    est = CostEstimator()
    base = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=4)
    with_svd = est.mps(
        num_qubits=4,
        num_1q_gates=0,
        num_2q_gates=1,
        chi=4,
        svd=True,
    )
    assert with_svd.time == base.time + 4 * (4 ** 3) * math.log2(4)


def test_decision_diagram_log_scaling():
    est = CostEstimator()
    c1 = est.decision_diagram(num_gates=10, frontier=5)
    c2 = est.decision_diagram(num_gates=10, frontier=10)
    ratio = (10 * math.log2(10)) / (5 * math.log2(5))
    assert c2.time == c1.time * ratio
    assert c2.memory == c1.memory * ratio
    assert c1.log_depth == math.log2(5)
    assert c2.log_depth == math.log2(10)


def test_decision_diagram_small_frontier_linear():
    est = CostEstimator()
    c1 = est.decision_diagram(num_gates=5, frontier=1)
    c2 = est.decision_diagram(num_gates=5, frontier=2)
    assert c2.time == 2 * c1.time
    assert c2.memory == 2 * c1.memory


def test_conversion_primitive_selection():
    est = CostEstimator(coeff={"lw_extract": 10.0, "full_extract": 10.0, "st_stage": 10.0})
    small = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=2,
        frontier=0,
        window=2,
    )
    assert small.primitive == "B2B"
    large = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=4,
        rank=16,
        frontier=0,
        window=2,
    )
    assert large.primitive == "LW"
    assert large.cost.time > small.cost.time
    assert small.cost.log_depth == math.log2(2)
    assert large.cost.log_depth == math.log2(16)


def test_lw_gate_counts_increase_time():
    est = CostEstimator()
    base = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=4,
        rank=16,
        frontier=0,
        window=2,
    )
    with_gates = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=4,
        rank=16,
        frontier=0,
        window=2,
        window_1q_gates=5,
    )
    assert with_gates.cost.time > base.cost.time


def test_conversion_caps():
    est = CostEstimator(q_max=2, r_max=2, s_max=4)
    res = est.conversion(
        Backend.STATEVECTOR,
        Backend.TABLEAU,
        num_qubits=3,
        rank=8,
        frontier=3,
    )
    assert res.primitive == "Full"
    assert math.isinf(res.cost.time)
    assert math.isinf(res.cost.memory)
