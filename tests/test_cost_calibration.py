from calibration.cost_model_sweeps import fit_all_coefficients
from docs.utils.partitioning_analysis import FragmentStats, evaluate_fragment_backends
from quasar.cost import Backend, CostEstimator


def _configured_estimator() -> CostEstimator:
    estimator = CostEstimator()
    estimator.update_coefficients(fit_all_coefficients())
    return estimator


def test_calibrated_coefficients_shift_sparsity_boundary():
    estimator = _configured_estimator()
    stats = FragmentStats(
        num_qubits=12,
        num_1q_gates=14,
        num_2q_gates=6,
        num_measurements=2,
    )
    dense_backend, _ = evaluate_fragment_backends(
        stats,
        sparsity=0.2,
        phase_rotation_diversity=6,
        amplitude_rotation_diversity=6,
        estimator=estimator,
    )
    sparse_backend, sparse_diag = evaluate_fragment_backends(
        stats,
        sparsity=0.92,
        phase_rotation_diversity=2,
        amplitude_rotation_diversity=1,
        estimator=estimator,
    )
    assert dense_backend != sparse_backend
    assert sparse_diag["backends"][Backend.DECISION_DIAGRAM]["feasible"]


def test_entanglement_changes_mps_boundary():
    estimator = _configured_estimator()
    low_ent_stats = FragmentStats(
        num_qubits=8,
        num_1q_gates=18,
        num_2q_gates=4,
        is_local=True,
    )
    # The high-entanglement fragment uses enough two-qubit gates to push the
    # estimator across the MPS/statevector boundary after the scalar ``chi``
    # expansion fix.  Lower counts still keep MPS as the winner.
    high_ent_stats = FragmentStats(
        num_qubits=8,
        num_1q_gates=18,
        num_2q_gates=50,
        is_local=True,
    )
    low_ent_backend, _ = evaluate_fragment_backends(
        low_ent_stats,
        sparsity=0.7,
        phase_rotation_diversity=3,
        amplitude_rotation_diversity=2,
        estimator=estimator,
    )
    high_ent_backend, _ = evaluate_fragment_backends(
        high_ent_stats,
        sparsity=0.1,
        phase_rotation_diversity=16,
        amplitude_rotation_diversity=16,
        estimator=estimator,
    )
    assert low_ent_backend != high_ent_backend


def test_depth_weight_affects_tableau_selection():
    estimator = _configured_estimator()
    shallow_stats = FragmentStats(
        num_qubits=5,
        num_1q_gates=6,
        num_2q_gates=3,
        num_measurements=1,
        is_clifford=True,
    )
    deep_stats = FragmentStats(
        num_qubits=5,
        num_1q_gates=90,
        num_2q_gates=60,
        num_measurements=2,
        is_clifford=True,
    )
    shallow_backend, _ = evaluate_fragment_backends(
        shallow_stats,
        sparsity=0.4,
        phase_rotation_diversity=1,
        amplitude_rotation_diversity=1,
        estimator=estimator,
        max_time=10_000,
    )
    deep_backend, _ = evaluate_fragment_backends(
        deep_stats,
        sparsity=0.4,
        phase_rotation_diversity=6,
        amplitude_rotation_diversity=4,
        estimator=estimator,
        max_time=5_000,
    )
    assert shallow_backend != deep_backend

