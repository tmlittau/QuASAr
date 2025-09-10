from quasar import Circuit
from quasar.partitioner import Partitioner

def test_ssd_metadata_enriched():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ]
    circ = Circuit(gates, use_classical_simplification=False)
    part = Partitioner()
    ssd = part.partition(circ)
    assert ssd.partitions[0].resources["memory"] == ssd.partitions[0].cost.memory
    assert ssd.partitions[1].dependencies == (0,)
    assert 0 in ssd.partitions[1].entangled_with
    assert ssd.partitions[1].backend in ssd.partitions[1].compatible_methods
