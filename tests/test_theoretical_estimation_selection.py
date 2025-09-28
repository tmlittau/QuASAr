from __future__ import annotations

import pytest

from benchmarks.bench_utils import paper_figures
from benchmarks.bench_utils import theoretical_estimation_selection as selection


def test_defaults_align_with_showcase_group() -> None:
    specs = selection.resolve_requested_specs(None, None)
    assert specs == selection.GROUPS["showcase"]


def test_explicit_paper_default_is_preserved() -> None:
    specs = selection.resolve_requested_specs(None, None, default_group="paper")
    assert specs == paper_figures.CIRCUITS


def test_group_selection_includes_showcase_circuits() -> None:
    specs = selection.resolve_requested_specs(None, ["showcase"])
    names = {spec.name for spec in specs}
    assert "layered_clifford_ramp" in names


def test_custom_circuit_with_alias_and_widths() -> None:
    specs = selection.resolve_requested_specs(["qft_circuit:4,6"], None)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "qft_circuit"
    assert spec.qubits == (4, 6)
    assert spec.kwargs is None


def test_custom_circuit_with_parameters() -> None:
    specs = selection.resolve_requested_specs(
        ["grover_circuit[n_iterations=2]:5"], None
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name.startswith("grover_circuit[")
    assert spec.qubits == (5,)
    assert spec.kwargs == {"n_iterations": 2}


def test_keyword_only_builder_supports_required_parameters() -> None:
    specs = selection.resolve_requested_specs(
        [
            "large_scale_circuits.alternating_ladder_circuit"
            "[dense_gadgets=2,gadget_width=3,ladder_layers=2,gadget_layers=2]:6"
        ],
        None,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.qubits == (6,)
    assert spec.kwargs is not None
    assert spec.kwargs["dense_gadgets"] == 2


def test_missing_widths_raise_value_error() -> None:
    with pytest.raises(ValueError):
        selection.resolve_requested_specs(["qft_circuit"], None)


def test_unknown_group_raises_value_error() -> None:
    with pytest.raises(ValueError):
        selection.resolve_requested_specs(None, ["unknown"])  # type: ignore[list-item]
