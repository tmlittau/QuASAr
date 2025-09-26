"""Tests for the high-impact benchmark circuits."""

from __future__ import annotations

from typing import List

import pandas as pd
import pytest

from benchmarks.bench_utils import showcase_benchmarks as sb
from benchmarks.circuits import (
    CLIFFORD_GATES,
    classical_controlled_circuit,
    clustered_entanglement_circuit,
    layered_clifford_nonclifford_circuit,
    layered_clifford_ramp_circuit,
)


def _layer_gates(circuit, layer_offsets: List[int], layer: int):
    start = layer_offsets[layer]
    end = layer_offsets[layer + 1] if layer + 1 < len(layer_offsets) else len(circuit.gates)
    return circuit.gates[start:end]


def test_clustered_entanglement_prep_blocks():
    circuit = clustered_entanglement_circuit(
        10, block_size=5, state="ghz", entangler="qft", depth=0
    )
    metadata = circuit.metadata
    assert metadata["num_blocks"] == 2
    prep_gates = circuit.gates[: metadata["prep_gate_count"]]
    first_block = {0, 1, 2, 3, 4}
    second_block = {5, 6, 7, 8, 9}
    fb_gates = [g for g in prep_gates if set(g.qubits) <= first_block]
    sb_gates = [g for g in prep_gates if set(g.qubits) <= second_block]
    assert any(g.gate == "H" and g.qubits == [0] for g in fb_gates)
    assert any(g.gate == "H" and g.qubits == [5] for g in sb_gates)
    for idx in range(1, 5):
        assert any(g.gate == "CX" and g.qubits == [idx - 1, idx] for g in fb_gates)
        offset = idx + 5 - 1
        assert any(g.gate == "CX" and g.qubits == [offset, offset + 1] for g in sb_gates)


def test_clustered_entanglement_random_layer_metadata():
    circuit = clustered_entanglement_circuit(
        10,
        block_size=5,
        state="w",
        entangler="random",
        depth=3,
        seed=123,
    )
    metadata = circuit.metadata
    assert metadata["state"] == "w"
    assert len(metadata["layer_offsets"]) == 3
    random_section = circuit.gates[metadata["prep_gate_count"] :]
    assert any(g.gate not in CLIFFORD_GATES for g in random_section)
    # Each W preparation ends with an X gate on the first qubit of the block.
    for block in range(metadata["num_blocks"]):
        qubit = block * metadata["block_size"]
        assert any(g.gate == "X" and g.qubits == [qubit] for g in circuit.gates)


def test_layered_clifford_transition_delays_magic():
    circuit = layered_clifford_nonclifford_circuit(
        6, depth=12, fraction_clifford=0.5, seed=1
    )
    metadata = circuit.metadata
    offsets = metadata["layer_offsets"]
    assert len(offsets) == 12
    for layer in range(metadata["clifford_layers"]):
        gates = _layer_gates(circuit, offsets, layer)
        assert all(g.gate in CLIFFORD_GATES for g in gates)
    for layer in range(metadata["clifford_layers"], metadata["depth"]):
        gates = _layer_gates(circuit, offsets, layer)
        assert any(g.gate not in CLIFFORD_GATES for g in gates)


def test_layered_clifford_ramp_metadata():
    circuit = layered_clifford_ramp_circuit(
        5, depth=10, ramp_start_fraction=0.3, ramp_end_fraction=0.6, seed=2
    )
    metadata = circuit.metadata
    flags = metadata["non_clifford_layer_flags"]
    assert len(flags) == metadata["depth"]
    assert not any(flags[: metadata["ramp_start_layer"]])
    assert any(flags[metadata["ramp_end_layer"] :])


def test_classical_controlled_circuit_enables_simplification():
    circuit = classical_controlled_circuit(
        12, depth=5, classical_qubits=4, toggle_period=2, fanout=2, seed=7
    )
    metadata = circuit.metadata
    assert circuit.use_classical_simplification is False
    assert metadata["classical_qubits"] == [0, 1, 2, 3]
    assert metadata["prep_gate_count"] == len(metadata["classical_qubits"]) // 2
    # Ensure classical qubits only appear as controls or are flipped by X gates.
    classical_set = set(metadata["classical_qubits"])
    for gate in circuit.gates:
        control_gates = {"CX", "CZ", "CRZ"}
        if gate.qubits[0] in classical_set and gate.gate in control_gates:
            continue
        if gate.gate == "X" and gate.qubits[0] in classical_set:
            continue
        assert not classical_set.intersection(gate.qubits)
    before = len(circuit.gates)
    circuit.enable_classical_simplification()
    after = len(circuit.gates)
    assert after <= before


def test_resolve_selected_defaults_to_all() -> None:
    """When no selection is provided all showcase circuits are returned."""

    expected = list(sb.SHOWCASE_CIRCUITS)
    assert sb._resolve_selected_circuits(explicit=None, groups=None) == expected


def test_resolve_selected_combines_groups_and_explicit() -> None:
    """Explicit circuits and groups are merged without duplicates."""

    explicit = ["classical_controlled_fanout"]
    group = ["clustered"]
    result = sb._resolve_selected_circuits(explicit=explicit, groups=group)
    for name in sb.SHOWCASE_GROUPS["clustered"]:
        assert name in result
    assert result.count("classical_controlled_fanout") == 1


def test_resolve_selected_unknown_group() -> None:
    """An unknown group name raises ``SystemExit`` with a helpful message."""

    with pytest.raises(SystemExit):
        sb._resolve_selected_circuits(explicit=None, groups=["missing"])


def test_merge_results_updates_existing(tmp_path) -> None:
    """New measurements replace matching rows while preserving existing data."""

    path = tmp_path / "data.csv"
    existing = pd.DataFrame(
        {
            "circuit": ["clustered_ghz_random", "layered_clifford_ramp"],
            "framework": ["quasar", "quasar"],
            "qubits": [30, 40],
            "run_time_mean": [1.0, 2.0],
        }
    )
    existing.to_csv(path, index=False)

    new_rows = pd.DataFrame(
        {
            "circuit": ["layered_clifford_ramp"],
            "framework": ["quasar"],
            "qubits": [40],
            "run_time_mean": [3.5],
        }
    )

    merged = sb._merge_results(
        path,
        new_rows,
        key_columns=("circuit", "framework", "qubits"),
        sort_columns=("circuit", "qubits"),
    )

    assert len(merged) == 2
    assert merged.loc[merged["circuit"] == "layered_clifford_ramp", "run_time_mean"].item() == 3.5
