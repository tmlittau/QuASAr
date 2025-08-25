from quasar import Backend, CostEstimator


def test_statevector_scaling():
    est = CostEstimator()
    small = est.statevector(num_qubits=3, num_gates=1)
    large = est.statevector(num_qubits=4, num_gates=1)
    assert large.time == 2 * small.time
    assert large.memory == 2 * small.memory


def test_tableau_quadratic():
    est = CostEstimator()
    cost = est.tableau(num_qubits=5, num_gates=2)
    assert cost.time == 2 * 25
    assert cost.memory == 25


def test_mps_chi_dependence():
    est = CostEstimator()
    chi2 = est.mps(num_qubits=4, num_gates=1, chi=2)
    chi4 = est.mps(num_qubits=4, num_gates=1, chi=4)
    assert chi4.time == chi2.time * (4 ** 3) / (2 ** 3)
    assert chi4.memory == chi2.memory * (4 ** 2) / (2 ** 2)


def test_decision_diagram_linear():
    est = CostEstimator()
    c1 = est.decision_diagram(num_gates=10, frontier=5)
    c2 = est.decision_diagram(num_gates=10, frontier=10)
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
