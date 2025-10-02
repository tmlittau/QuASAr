import pytest

from benchmarks.run_benchmark import parse_args


def test_parse_args_accepts_stitched_suite() -> None:
    args = parse_args(["--suite", "stitched-big"])
    assert args.suite == "stitched-big"
    assert args.circuit_names is None


def test_parse_args_suite_conflicts_with_circuit() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            ["--suite", "stitched-big", "--circuit", "clustered_ghz_random"]
        )


def test_parse_args_suite_conflicts_with_group() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--suite", "stitched-big", "--group", "clustered"])


def test_parse_args_accepts_reuse_flag() -> None:
    args = parse_args(["--reuse-existing"])
    assert args.reuse_existing is True
