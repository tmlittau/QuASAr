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


def test_split_and_recover():
    gates = [
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    # Make conversions essentially free to encourage a cut
    coeff = {
        "b2b_svd": 0.0,
        "b2b_copy": 0.0,
        "ingest_dd": 0.0,
        "ingest_tab": 0.0,
        "ingest_sv": 0.0,
        "dd_gate": 10.0,
    }
    planner = Planner(CostEstimator(coeff), quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None)
    result = planner.plan(circ)
    steps = result.steps
    assert [(s.start, s.end, s.backend) for s in steps] == [
        (0, 2, Backend.TABLEAU),
        (2, 3, Backend.STATEVECTOR),
    ]
    # Explicit backpointer recovery
    recovered = result.recover()
    assert [(s.start, s.end, s.backend) for s in recovered] == [
        (0, 2, Backend.TABLEAU),
        (2, 3, Backend.STATEVECTOR),
    ]


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
    assert len(steps) == 1
    assert steps[0].backend == Backend.STATEVECTOR


def test_conversion_cost_multiplier_discourages_switch():
    gates = [
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ]
    circ = Circuit.from_dict(gates)
    coeff = {
        "sv_gate": 1.0,
        "tab_gate": 0.1,
        "b2b_svd": 0.0,
        "b2b_copy": 0.0,
        "ingest_sv": 0.375,
    }
    est = CostEstimator(coeff)
    base = Planner(
        est,
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )
    steps = base.plan(circ).steps
    assert [(s.start, s.end, s.backend) for s in steps] == [
        (0, 2, Backend.TABLEAU),
        (2, 3, Backend.STATEVECTOR),
    ]
    penalized = Planner(
        est,
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
        conversion_cost_multiplier=5.0,
    )
    steps2 = penalized.plan(circ).steps
    assert [(s.start, s.end, s.backend) for s in steps2] == [
        (0, 2, Backend.TABLEAU),
        (2, 3, Backend.STATEVECTOR),
    ]
