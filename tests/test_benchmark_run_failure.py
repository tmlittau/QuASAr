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


class NotImplementedBackend:
    name = "unsupported"

    def run(self, circuit, **_):
        raise NotImplementedError("nyi")


def test_run_returns_unsupported_record_on_not_implemented():
    runner = BenchmarkRunner()
    record = runner.run("circ", NotImplementedBackend())
    assert record["failed"] is False
    assert record["unsupported"] is True
    assert "nyi" in record["error"]
    assert record["framework"] == "unsupported"
