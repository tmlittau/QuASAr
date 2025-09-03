from benchmarks.backends import StimAdapter
from quasar.circuit import Circuit
from quasar.ssd import SSD


def test_stim_adapter_run_defaults_to_ssd_and_supports_statevector():
    circ = Circuit.from_dict([{"gate": "H", "qubits": [0]}])
    backend = StimAdapter()
    ssd = backend.run(circ)
    assert isinstance(ssd, SSD)
    vec = backend.run(circ, statevector=True)
    assert len(vec) == 2
