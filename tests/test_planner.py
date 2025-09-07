from quasar import Circuit, Planner, Backend, CostEstimator


def test_tableau_for_clifford():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    result = Planner().plan(circ)
    steps = result.steps
    assert len(steps) == 1
    step = steps[0]
    assert (step.start, step.end, step.backend) == (0, 2, Backend.TABLEAU)


def test_explicit_statevector_for_clifford():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    result = Planner().plan(circ, backend=Backend.STATEVECTOR)
    steps = result.steps
    assert len(steps) == 1
    assert steps[0].backend == Backend.STATEVECTOR


def test_planner_simplifies_controls_for_clifford_detection():
    gates = [
        {"gate": "H", "qubits": [1]},
        {"gate": "CT", "qubits": [0, 2]},
    ]
    circ = Circuit.from_dict(gates, use_classical_simplification=False)
    circ.use_classical_simplification = True
    circ.classical_state = [0] * circ.num_qubits
    result = Planner().plan(circ, backend=Backend.TABLEAU)
    assert result.final_backend == Backend.TABLEAU
    assert [g.gate for g in circ.gates] == ["H"]


def test_statevector_for_non_clifford():
    gates = [
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    coeff = {
        "b2b_svd": 0.0,
        "b2b_copy": 0.0,
        "ingest_dd": 0.0,
        "ingest_tab": 0.0,
        "ingest_sv": 0.0,
        "dd_gate": 10.0,
    }
    planner = Planner(CostEstimator(coeff))
    result = planner.plan(circ)
    steps = result.steps
    assert all(s.backend == Backend.STATEVECTOR for s in steps)
    # Explicit backpointer recovery
    recovered = result.recover()
    assert all(s.backend == Backend.STATEVECTOR for s in recovered)


def test_planner_respects_caps():
    gates = [
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    coeff = {
        "b2b_svd": 0.0,
        "b2b_copy": 0.0,
        "ingest_dd": 0.0,
        "ingest_tab": 0.0,
        "ingest_sv": 0.0,
        "dd_gate": 10.0,
    }
    est = CostEstimator(coeff, q_max=0, r_max=0, s_max=1)
    planner = Planner(est)
    result = planner.plan(circ)
    steps = result.steps
    assert all(s.backend == Backend.STATEVECTOR for s in steps)


def test_conversion_cost_multiplier_discourages_switch():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    coeff = {
        "sv_gate_1q": 1.0,
        "sv_gate_2q": 1.0,
        "sv_meas": 1.0,
        "tab_gate": 0.1,
        "b2b_svd": 0.0,
        "b2b_copy": 0.0,
        "ingest_sv": 0.375,
    }
    est = CostEstimator(coeff)
    base = Planner(est, perf_prio="time")
    steps = base.plan(circ).steps
    assert all(s.backend == Backend.STATEVECTOR for s in steps)
    penalized = Planner(
        est,
        conversion_cost_multiplier=50.0,
        perf_prio="time",
    )
    steps2 = penalized.plan(circ).steps
    assert all(s.backend == Backend.STATEVECTOR for s in steps2)
