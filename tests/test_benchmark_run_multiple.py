import math
import time
from unittest.mock import patch

from benchmarks.runner import BenchmarkRunner


class DummyBackend:
    name = "dummy"

    def run(self, circuit, **_):
        return None


def test_run_multiple_aggregates_statistics():
    runner = BenchmarkRunner()
    # perf_counter is called once for start_time and twice per run
    side_effect = [
        0.0,  # start of run_multiple
        # run 1
        0.0, 1.0,
        # run 2
        1.0, 3.0,
        # run 3
        3.0, 6.0,
    ]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        record = runner.run_multiple(None, DummyBackend(), repetitions=3)
    assert len(runner.results) == 1
    assert runner.results[0] == record
    assert record["repetitions"] == 3
    assert record["run_time_mean"] == 2.0
    assert math.isclose(record["run_time_std"], math.sqrt(2 / 3))


class SleepBackend:
    name = "sleep"

    def __init__(self, durations):
        self.durations = durations

    def run(self, circuit, **_):
        t = self.durations.pop(0)
        time.sleep(t)
        return None


def test_run_multiple_timeout_records_failure():
    runner = BenchmarkRunner()
    backend = SleepBackend([0.01, 0.2])
    record = runner.run_multiple(
        None, backend, repetitions=2, run_timeout=0.05
    )
    assert record["repetitions"] == 1
    assert "failed_runs" in record and len(record["failed_runs"]) == 1
    assert "timed out" in record["failed_runs"][0]
