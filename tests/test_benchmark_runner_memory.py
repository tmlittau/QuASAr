from benchmarks.runner import BenchmarkRunner


class DummyBackend:
    name = "dummy"

    def prepare(self, circuit):
        # allocate some memory to ensure tracemalloc captures it
        self._prep = [0] * 10000
        return circuit

    def run(self, circuit, **kwargs):
        self._run = [0] * 10000
        return "done"


def test_run_records_memory():
    runner = BenchmarkRunner()
    record = runner.run(None, DummyBackend())
    assert record["prepare_memory"] > 0
    assert record["run_memory"] > 0

    df = runner.dataframe()
    if isinstance(df, list):
        assert "prepare_memory" in df[0]
        assert "run_memory" in df[0]
    else:  # pragma: no cover - requires pandas
        assert set(["prepare_memory", "run_memory"]).issubset(df.columns)


class DummyPlanner:
    def plan(self, circuit):
        self._data = [0] * 10000


class DummyScheduler:
    def __init__(self):
        self.planner = DummyPlanner()

    def run(self, circuit):
        self._data = [0] * 10000
        return "done"


def test_run_quasar_records_memory():
    runner = BenchmarkRunner()
    record = runner.run_quasar(None, DummyScheduler())
    assert record["prepare_memory"] > 0
    assert record["run_memory"] > 0
