import math

import pandas as pd

from benchmarks.bench_utils import circuits
from benchmarks.run_benchmark import run_clifford_random_suite


def _distance_to_pi_over_4(theta: float) -> float:
    two_pi = 2.0 * math.pi
    pi_over_4 = math.pi / 4.0
    t = theta % two_pi
    min_distance = two_pi
    for k in range(8):
        target = k * pi_over_4
        direct = abs(t - target)
        min_distance = min(min_distance, direct, two_pi - direct)
    return min_distance


def test_random_clifford_with_tail_inserts_non_clifford_rotations() -> None:
    circuit = circuits.random_clifford_with_tail_circuit(
        3,
        clifford_depth=2,
        total_depth=4,
        clifford_seed=11,
        tail_seed=22,
        tail_twoq_prob=1.0,
        tail_angle_eps=1e-3,
    )

    non_clifford = [gate for gate in circuit.gates if gate.gate not in circuits.CLIFFORD_GATES]
    assert non_clifford, "tail should introduce non-Clifford rotations"
    for gate in non_clifford:
        theta = gate.params.get("theta")
        assert theta is not None
        assert _distance_to_pi_over_4(theta) > 1e-3

    metadata = getattr(circuit, "metadata", {})
    assert metadata.get("clifford_depth") == 2
    assert metadata.get("total_depth") == 4
    assert metadata.get("tail_layers") == 2


def test_run_clifford_random_suite_executes(tmp_path) -> None:
    database_path = tmp_path / "clifford_suite.sqlite"
    df = run_clifford_random_suite(
        widths=[1],
        clifford_depths=[1],
        total_depths=[2],
        repetitions=1,
        run_timeout=None,
        classical_simplification=False,
        workers=1,
        quasar_quick=True,
        reuse_existing=False,
        database_path=database_path,
        include_theoretical_sv=False,
        tail_twoq_prob=0.0,
        tail_angle_eps=1e-3,
        clifford_seed=5,
        tail_seed=7,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    frameworks = set(df["framework"].unique())
    assert {"quasar", "STATEVECTOR", "EXTENDED_STABILIZER"}.issubset(frameworks)
