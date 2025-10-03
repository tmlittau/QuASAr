import logging
import math

import pytest

from benchmarks.bench_utils import circuits as circuit_lib
from quasar.circuit import Circuit, Gate
from quasar.planner import NoFeasibleBackendError, Planner, partition_into_disjoint_fragments


def _synthetic_disjoint_circuit(width: int, block_size: int) -> Circuit:
    gates: list[Gate] = []
    for start in range(0, width, block_size):
        block = list(range(start, min(start + block_size, width)))
        if not block:
            continue
        gates.append(Gate("H", [block[0]]))
        gates.append(Gate("S", [block[0]]))
        if len(block) > 1:
            gates.append(Gate("CX", [block[0], block[-1]]))
            gates.append(Gate("T", [block[-1]]))
            if len(block) > 2:
                gates.append(Gate("CX", [block[1], block[2]]))
    circuit = Circuit(gates, use_classical_simplification=False)
    num_blocks = math.ceil(width / block_size)
    setattr(circuit, "metadata", {"block_size": block_size, "num_blocks": num_blocks})
    return circuit


def test_partition_detects_blocks() -> None:
    circuit = _synthetic_disjoint_circuit(8, block_size=2)
    fragments = partition_into_disjoint_fragments(circuit)
    assert len(fragments) == 4
    for fragment in fragments:
        assert set(fragment.qubits)
        assert len(fragment.gates) == len(fragment.original_gate_indices)
        for local_gate, global_index in zip(fragment.gates, fragment.original_gate_indices):
            global_gate = circuit.gates[global_index]
            assert set(global_gate.qubits).issubset(set(fragment.qubits))
            mapped = [fragment.local_to_global[q] for q in local_gate.qubits]
            assert mapped == list(global_gate.qubits)


# Use modest widths so the test remains fast on CI; larger instances are
# exercised in the stitched-disjoint benchmark suite.
@pytest.mark.parametrize("width", [32, 48])
def test_planner_plans_fragments_under_cap(width: int, caplog: pytest.LogCaptureFixture) -> None:
    circuit = _synthetic_disjoint_circuit(width, block_size=8)
    planner = Planner(max_memory=64 * 1024 ** 3, batch_size=32)
    with caplog.at_level(logging.INFO):
        plan = planner.plan(circuit)
    assert plan.steps
    assert any("Detected" in rec.message for rec in caplog.records)
    assert any("Fragment concurrency" in rec.message for rec in caplog.records)


def test_failure_only_when_fragment_unfit() -> None:
    circuit = circuit_lib.clustered_ghz_random_circuit(32, block_size=8, depth=2)
    planner = Planner(max_memory=100.0)
    with pytest.raises(NoFeasibleBackendError) as excinfo:
        planner.plan(circuit)
    message = str(excinfo.value)
    assert "Fragment qubits" in message
    assert "STATEVECTOR=" in message or "MPS=" in message
    assert "reduce block_size" in message
