from benchmarks.runner import BenchmarkRunner


class ErrorBackend:
    name = "error"

    def run(self, circuit, **_):
        raise ValueError("boom")


def test_run_returns_failure_record_on_exception():
    runner = BenchmarkRunner()
    record = runner.run("circ", ErrorBackend())
    assert record["failed"] is True
    assert "boom" in record["error"]
    assert record["framework"] == "error"
    assert runner.results[0] == record
