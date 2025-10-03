import logging

import pytest

from benchmarks.bench_utils import circuits as circuit_lib
from benchmarks.bench_utils import stitched_suite
from quasar.cost import Backend, CostEstimator
from quasar.circuit import Gate
from quasar.planner import Planner, NoFeasibleBackendError, _parallel_simulation_cost


@pytest.fixture(autouse=True)
def _reset_disjoint_options():
    # Reset stitched suite configuration for isolation across tests.
    stitched_suite.configure_disjoint_suite(
        enforce_disjoint=True, auto_size_by_ram=True, max_ram_gb=64, block_size=None
    )
    yield
    stitched_suite.configure_disjoint_suite(
        enforce_disjoint=True, auto_size_by_ram=True, max_ram_gb=64, block_size=None
    )


def test_disjoint_runs_under_64gb_statevector_or_fallback():
    spec = stitched_suite.build_stitched_disjoint_suite()[0]
    circuit = spec.factory(16)
    planner = Planner(max_memory=64 * 1024 ** 3)
    plan = planner.plan(circuit)
    backends = {step.backend for step in plan.steps}
    allowed = {
        Backend.STATEVECTOR,
        Backend.MPS,
        Backend.DECISION_DIAGRAM,
        Backend.TABLEAU,
        Backend.EXTENDED_STABILIZER,
    }
    assert backends <= allowed


def test_disjoint_assertion_blocks_cross_links():
    circuit = circuit_lib.clustered_w_random_neighborbridge_random_circuit(
        16, block_size=4, region_blocks=2, bridge_layers=1
    )
    with pytest.raises(ValueError):
        stitched_suite.verify_disjoint_stitched_circuit(circuit, block_size=4)


def test_concurrency_downshifts_to_fit(caplog):
    estimator = CostEstimator()
    gate_a = Gate("H", [0])
    gate_b = Gate("H", [1])
    groups = [((0,), [gate_a]), ((1,), [gate_b])]
    with caplog.at_level(logging.INFO):
        cost, used_parallel = _parallel_simulation_cost(
            estimator,
            Backend.STATEVECTOR,
            groups,
            memory_limit=9.0e4,
            max_groups=None,
        )
    assert not used_parallel
    assert any("executing sequentially" in rec.message for rec in caplog.records)


def test_error_message_has_details_when_single_region_unfit():
    planner = Planner(max_memory=1.0)
    circuit = circuit_lib.clustered_ghz_random_circuit(6, block_size=3, depth=2)
    with pytest.raises(NoFeasibleBackendError) as excinfo:
        planner.plan(circuit)
    message = str(excinfo.value)
    assert "memory limit" in message.lower()
    assert "Largest region spans" in message
    assert any(name in message for name in ("STATEVECTOR", "MPS", "TABLEAU", "DECISION_DIAGRAM"))
