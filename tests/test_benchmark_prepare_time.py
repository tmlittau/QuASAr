import time
from benchmarks.runner import BenchmarkRunner
from quasar.circuit import Circuit


class SlowPrepareBackend:
    """Backend with an expensive preparation step.

    ``prepare`` simulates a costly compilation by sleeping while ``run`` is
    essentially free.  This allows verifying that the benchmark runner measures
    preparation time separately from execution time.
    """

    name = "slow_prepare"

    def prepare(self, circuit: Circuit):
        time.sleep(0.05)
        return circuit

    def run(self, circuit: Circuit, **_):
        return 0


def test_prepare_time_dominates_for_heavy_compilation():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    backend = SlowPrepareBackend()
    runner = BenchmarkRunner()
    record = runner.run(circuit, backend)
    assert record["prepare_time"] > record["run_time"]
