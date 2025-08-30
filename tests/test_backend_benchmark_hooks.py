from benchmarks.runner import BenchmarkRunner
from quasar.backends.base import Backend
from quasar.circuit import Circuit

class DummyBackend(Backend):
    def __init__(self):
        self._benchmark_mode = False
        self._benchmark_ops = []
        self.executed = []

    def load(self, num_qubits: int, **kwargs):
        pass

    def ingest(self, state):
        pass

    def apply_gate(self, name, qubits, params=None):
        if self._benchmark_mode:
            self._benchmark_ops.append((name, qubits, params))
        else:
            self.executed.append(name)

    def extract_ssd(self):
        return self.executed

    def statevector(self):
        return self.executed

def test_benchmark_runner_uses_hooks():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    backend = DummyBackend()
    runner = BenchmarkRunner()
    record = runner.run(circuit, backend)
    assert backend.executed == ["H"]
    assert record["result"] == ["H"]
