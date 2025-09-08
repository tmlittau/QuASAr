from quasar import Backend, Circuit, CostEstimator, Gate
import math


def test_statevector_scaling():
    est = CostEstimator()
    small = est.statevector(num_qubits=3, num_1q_gates=1, num_2q_gates=0, num_meas=0)
    large = est.statevector(num_qubits=4, num_1q_gates=1, num_2q_gates=0, num_meas=0)
    assert large.time == 2 * small.time
    assert large.memory == 2 * small.memory
    assert small.log_depth == math.log2(3)
    assert large.log_depth == math.log2(4)


def test_statevector_precision_memory():
    est = CostEstimator()
    single = est.statevector(
        num_qubits=3,
        num_1q_gates=0,
        num_2q_gates=0,
        num_meas=0,
        precision="complex64",
    )
    double = est.statevector(
        num_qubits=3,
        num_1q_gates=0,
        num_2q_gates=0,
        num_meas=0,
        precision="complex128",
    )
    assert double.memory == 2 * single.memory


def test_tableau_qubit_scaling():
    est = CostEstimator()
    small = est.tableau(num_qubits=2, num_gates=1)
    large = est.tableau(num_qubits=4, num_gates=1)
    assert large.time == 4 * small.time
    assert large.memory == 4 * small.memory


def test_tableau_measurement_memory():
    est = CostEstimator()
    base = est.tableau(num_qubits=3, num_gates=0)
    meas = est.tableau(num_qubits=3, num_gates=0, num_meas=5)
    assert meas.memory == base.memory + 5 * est.coeff["tab_meas_mem"]


def test_mps_chi_dependence():
    est = CostEstimator()
    chi2 = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=2)
    chi4 = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=4)
    base_t = est.coeff["mps_base_time"]
    ratio_time = (chi4.time - base_t) / (chi2.time - base_t)
    expected_time = (2 * 4**2 + 4**3) / (2 * 2**2 + 2**3)
    assert math.isclose(ratio_time, expected_time)
    base_m = est.coeff["mps_base_mem"]
    ratio_mem = (chi4.memory - base_m) / (chi2.memory - base_m)
    expected_mem = (2 * 4 + 2 * 4**2) / (2 * 2 + 2 * 2**2)
    assert math.isclose(ratio_mem, expected_mem)


def test_mps_gate_scaling():
    est = CostEstimator()
    n = 4
    one = est.mps(num_qubits=n, num_1q_gates=1, num_2q_gates=0, chi=4)
    two = est.mps(num_qubits=n, num_1q_gates=0, num_2q_gates=1, chi=4)
    sum_site = 2 * 4 + (n - 2) * 4**2
    bond_sum = 2 * 4**2 + (n - 3) * 4**3
    ratio = n * bond_sum / (n - 1) / sum_site
    base_t = est.coeff["mps_base_time"]
    assert math.isclose(two.time - base_t, (one.time - base_t) * ratio)


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
    trunc = (16 * math.log2(4) + 64 * math.log2(4) + 16 * math.log2(4)) / 3
    expected_time = base.time + est.coeff["mps_trunc"] * 4 * trunc
    expected_mem = base.memory + est.coeff["mps_temp_mem"] * 64
    assert with_svd.time == expected_time
    assert with_svd.memory == expected_mem


def test_mps_per_bond_list():
    est = CostEstimator()
    varied = est.mps(num_qubits=3, num_1q_gates=0, num_2q_gates=1, chi=[2, 4])
    uniform = est.mps(num_qubits=3, num_1q_gates=0, num_2q_gates=1, chi=4)
    assert varied.time < uniform.time
    assert varied.memory < uniform.memory


def test_mps_derive_chi_from_gates():
    est = CostEstimator()
    gates = [Gate("CX", [0, 1]), Gate("CX", [1, 2])]
    auto = est.mps(num_qubits=3, num_1q_gates=0, num_2q_gates=2, chi=None, gates=gates)
    manual = est.mps(num_qubits=3, num_1q_gates=0, num_2q_gates=2, chi=[2, 2])
    assert auto.time == manual.time
    assert auto.memory == manual.memory


def test_mps_temp_memory():
    est = CostEstimator(coeff={"mps_temp_mem": 2.0})
    base = est.mps(num_qubits=4, num_1q_gates=0, num_2q_gates=1, chi=4)
    with_svd = est.mps(
        num_qubits=4,
        num_1q_gates=0,
        num_2q_gates=1,
        chi=4,
        svd=True,
    )
    assert with_svd.memory == base.memory + 2.0 * 64


def test_mps_qubit_scaling():
    est = CostEstimator()
    c3 = est.mps(num_qubits=3, num_1q_gates=0, num_2q_gates=1, chi=2)
    c5 = est.mps(num_qubits=5, num_1q_gates=0, num_2q_gates=1, chi=2)
    assert c5.time > c3.time
    assert c5.memory > c3.memory


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


def test_decision_diagram_gate_complexity_scaling():
    est = CostEstimator()
    c1 = est.decision_diagram(num_gates=5, frontier=8)
    c2 = est.decision_diagram(num_gates=20, frontier=8)
    ratio_mem = math.log2(21) / math.log2(6)
    ratio_time = (20 * math.log2(21)) / (5 * math.log2(6))
    assert math.isclose(c2.memory, c1.memory * ratio_mem)
    assert math.isclose(c2.time, c1.time * ratio_time)


def test_decision_diagram_memory_coefficients():
    est = CostEstimator(coeff={"dd_node_bytes": 2.0, "dd_cache_overhead": 1.0})
    cost = est.decision_diagram(num_gates=10, frontier=4)
    active = 4 * math.log2(4)
    nodes = active * math.log2(11)
    expected = nodes * 2.0
    expected += expected  # cache overhead of 1.0 doubles the table
    expected *= 0.05
    assert math.isclose(cost.memory, expected)

def test_conversion_primitive_selection():
    est = CostEstimator(
        coeff={"lw_extract": 10.0, "full_extract": 10.0, "st_stage": 10.0}
    )
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


def test_b2b_svd_memory_overhead():
    coeff = {
        "ingest_mps": 0.0,
        "conversion_base": 0.0,
        "lw_extract": 100.0,
        "full_extract": 100.0,
        "st_stage": 100.0,
        "b2b_svd_mem": 0.0,
    }
    base_est = CostEstimator(coeff=coeff)
    base = base_est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=2,
        frontier=0,
        window=2,
    )
    coeff_mem = dict(coeff)
    coeff_mem["b2b_svd_mem"] = 2.0
    est = CostEstimator(coeff=coeff_mem)
    with_mem = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=2,
        frontier=0,
        window=2,
    )
    assert with_mem.primitive == "B2B"
    assert with_mem.cost.memory == base.cost.memory + 2.0 * 4


def test_lw_temp_memory_overhead():
    coeff = {
        "b2b_svd": 100.0,
        "b2b_copy": 100.0,
        "st_stage": 100.0,
        "full_extract": 100.0,
        "ingest_mps": 0.0,
        "conversion_base": 0.0,
        "lw_temp_mem": 0.0,
    }
    base_est = CostEstimator(coeff=coeff)
    base = base_est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=4,
        frontier=0,
        window=2,
    )
    coeff_mem = dict(coeff)
    coeff_mem["lw_temp_mem"] = 2.0
    est = CostEstimator(coeff=coeff_mem)
    with_mem = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=4,
        frontier=0,
        window=2,
    )
    assert with_mem.primitive == "LW"
    assert with_mem.cost.memory == base.cost.memory + 2.0 * 4


def test_dd_ingest_memory_overhead():
    coeff = {
        "ingest_dd": 0.0,
        "b2b_svd": 1.0,
        "b2b_copy": 1.0,
        "lw_extract": 100.0,
        "st_stage": 100.0,
        "full_extract": 100.0,
        "conversion_base": 0.0,
        "ingest_dd_mem": 0.0,
    }
    base_est = CostEstimator(coeff=coeff)
    base = base_est.conversion(
        Backend.STATEVECTOR,
        Backend.DECISION_DIAGRAM,
        num_qubits=2,
        rank=2,
        frontier=1,
    )
    coeff_mem = dict(coeff)
    coeff_mem["ingest_dd_mem"] = 2.0
    est = CostEstimator(coeff=coeff_mem)
    with_mem = est.conversion(
        Backend.STATEVECTOR,
        Backend.DECISION_DIAGRAM,
        num_qubits=2,
        rank=2,
        frontier=1,
    )
    assert with_mem.cost.memory == base.cost.memory + 2.0 * 4


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


def test_b2b_uses_svd_min():
    est = CostEstimator(
        coeff={
            "b2b_copy": 0.0,
            "b2b_svd": 1.0,
            "ingest_mps": 0.0,
            "conversion_base": 0.0,
            "lw_extract": 100.0,
            "full_extract": 100.0,
        }
    )
    res = est.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=2,
        rank=4,
        frontier=0,
    )
    assert res.primitive == "B2B"
    # min(2 * 4^2, 4 * 2^2) = 16
    assert res.cost.time == 16


def test_st_chi_cap_override():
    coeff = {
        "b2b_svd": 1000.0,
        "b2b_copy": 1000.0,
        "lw_extract": 1000.0,
        "full_extract": 1000.0,
        "ingest_mps": 0.0,
        "conversion_base": 0.0,
    }
    est_default = CostEstimator(coeff=coeff)
    res_default = est_default.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=4,
        rank=32,
        frontier=0,
    )
    coeff["st_chi_cap"] = 8.0
    est_custom = CostEstimator(coeff=coeff)
    res_custom = est_custom.conversion(
        Backend.TABLEAU,
        Backend.MPS,
        num_qubits=4,
        rank=32,
        frontier=0,
    )
    assert res_default.primitive == "ST"
    assert res_custom.primitive == "ST"
    assert res_custom.cost.time < res_default.cost.time


def test_max_schmidt_rank_and_entropy():
    gates = [
        Gate("H", [0]),
        Gate("H", [1]),
        Gate("CX", [0, 2]),
        Gate("CX", [1, 3]),
    ]
    circ = Circuit(gates)
    est = CostEstimator()
    chi = est.max_schmidt_rank(circ.num_qubits, circ.gates)
    assert chi == 4
    entropy = est.entanglement_entropy(circ.num_qubits, circ.gates)
    assert entropy == math.log2(chi)
