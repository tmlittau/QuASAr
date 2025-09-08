import time
from types import SimpleNamespace

import pytest

from benchmarks.runner import BenchmarkRunner
from quasar.cost import Backend


class SlowSim:
    name = "slow"

    def load(self, n: int) -> None:
        pass

    def apply_gate(self, gate, qubits, params) -> None:  # pragma: no cover
        pass

    def extract_ssd(self):
        time.sleep(0.05)
        return None


class DummyScheduler:
    def __init__(self) -> None:
        self.backends = {Backend.STATEVECTOR: SlowSim()}

    def select_backend(self, circuit, backend=None):
        return backend or Backend.STATEVECTOR


def test_run_quasar_multiple_timeout() -> None:
    runner = BenchmarkRunner()
    engine = SimpleNamespace(scheduler=DummyScheduler())
    circuit = SimpleNamespace(num_qubits=1, gates=[], ssd=None)

    with pytest.raises(RuntimeError):
        runner.run_quasar_multiple(
            circuit,
            engine,
            backend=Backend.STATEVECTOR,
            repetitions=1,
            run_timeout=0.01,
            quick=True,
        )

    res = runner.run_quasar_multiple(
        circuit,
        engine,
        backend=Backend.STATEVECTOR,
        repetitions=1,
        run_timeout=0.2,
        quick=True,
    )
    assert res["backend"] == Backend.STATEVECTOR.name

