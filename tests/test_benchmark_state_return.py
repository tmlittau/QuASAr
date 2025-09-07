import math
from unittest.mock import patch

from benchmarks.runner import BenchmarkRunner
from quasar.cost import Backend
from quasar.planner import PlanResult
from quasar.ssd import SSD, SSDPartition


class DummyScheduler:
    def __init__(self):
        class Planner:
            def plan(self, circuit, *, backend=None):
                return PlanResult(
                    table=[],
                    final_backend=backend or Backend.STATEVECTOR,
                    gates=[],
                    explicit_steps=[],
                    explicit_conversions=[],
                    step_costs=[],
                )

        self.planner = Planner()

    def should_use_quick_path(self, circuit, *, backend=None):
        return False

    def prepare_run(self, circuit, plan=None, *, backend=None):
        return plan

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        return SSD(
            [
                SSDPartition(
                    subsystems=((0,),),
                    backend=plan.final_backend or Backend.STATEVECTOR,
                )
            ]
        )


def test_fixed_and_auto_runs_return_state_and_comparable_metrics(monkeypatch):
    runner = BenchmarkRunner()
    engine = DummyScheduler()

    monkeypatch.setattr("benchmarks.runner.tracemalloc.start", lambda: None)
    monkeypatch.setattr("benchmarks.runner.tracemalloc.stop", lambda: None)
    monkeypatch.setattr(
        "benchmarks.runner.tracemalloc.get_traced_memory", lambda: (0, 100)
    )

    side_effect = [0.0, 0.0, 1.0]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        fixed = runner.run_quasar_multiple(
            None, engine, backend=Backend.STATEVECTOR, repetitions=1
        )

    side_effect = [0.0, 0.0, 1.0]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        auto = runner.run_quasar_multiple(None, engine, repetitions=1)

    assert fixed["result"] is not None
    assert auto["result"] is not None
    assert fixed["run_time_mean"] == auto["run_time_mean"]
    assert fixed["run_peak_memory_mean"] == auto["run_peak_memory_mean"]
