from quasar import Circuit, Gate, Backend, Partitioner
from benchmarks.circuits import ghz_circuit, qft_circuit


def large_partitioned_circuit(n: int) -> Circuit:
    half = n // 2
    ghz = ghz_circuit(half, use_classical_simplification=False)
    qft = qft_circuit(half, use_classical_simplification=False)
    gates = list(ghz.gates)
    gates += [Gate(g.gate, [q + half for q in g.qubits], g.params) for g in qft.gates]
    gates += [
        Gate("CX", [0, half]),
        Gate("CX", [half - 1, n - 1]),
    ]
    return Circuit(gates, use_classical_simplification=False)


def test_large_partitioned_circuit_routing():
    circuit = large_partitioned_circuit(16)
    partitioner = Partitioner()
    ssd = partitioner.partition(circuit)

    assert len(ssd.partitions) >= 2
    assert ssd.partitions[0].backend == Backend.TABLEAU
    assert any(p.backend == Backend.DECISION_DIAGRAM for p in ssd.partitions)
    assert any(
        c.source == Backend.TABLEAU and c.target == Backend.DECISION_DIAGRAM
        for c in ssd.conversions
    )
