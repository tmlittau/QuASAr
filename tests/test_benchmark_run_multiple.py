import math
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
