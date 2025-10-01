"""Tests for the high-impact benchmark circuits."""

from __future__ import annotations

import math
import contextlib
from functools import partial
from types import SimpleNamespace
from typing import List

import pandas as pd
import pytest

from benchmarks.bench_utils import showcase_benchmarks as sb
from benchmarks.bench_utils import stitched_suite
from benchmarks.circuits import (
    CLIFFORD_GATES,
    classical_controlled_circuit,
    clustered_entanglement_circuit,
    clustered_ghz_qft_circuit,
    clustered_ghz_random_circuit,
    clustered_ghz_random_qft_circuit,
    clustered_w_qft_circuit,
    clustered_w_random_circuit,
    layered_clifford_nonclifford_circuit,
    layered_clifford_magic_islands_circuit,
    layered_clifford_ramp_circuit,
)
from quasar.circuit import Circuit, Gate
from quasar.cost import Backend
from quasar.partitioner import Partitioner


SMALL_BLOCK_SIZE = 3
SMALL_RANDOM_DEPTH = 8


def _layer_gates(circuit, layer_offsets: List[int], layer: int):
    start = layer_offsets[layer]
    end = layer_offsets[layer + 1] if layer + 1 < len(layer_offsets) else len(circuit.gates)
    return circuit.gates[start:end]


def _cluster_blocks(num_qubits: int, block_size: int) -> list[tuple[int, ...]]:
    return [
        tuple(range(start, min(start + block_size, num_qubits)))
        for start in range(0, num_qubits, block_size)
        if start < num_qubits
    ]


def _expected_island_flags(
    post_layers: int, islands: int, island_len: int, island_gap: int
) -> list[bool]:
    flags: list[bool] = []
    for idx in range(islands):
        if len(flags) >= post_layers:
            break
        remaining = post_layers - len(flags)
        flags.extend([True] * min(island_len, remaining))
        if len(flags) >= post_layers:
            break
        if idx < islands - 1 and island_gap > 0:
            remaining = post_layers - len(flags)
            flags.extend([False] * min(island_gap, remaining))
    if len(flags) < post_layers:
        flags.extend([False] * (post_layers - len(flags)))
    return flags


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


@pytest.mark.parametrize(
    "constructor",
    [
        partial(
            clustered_ghz_random_circuit,
            block_size=SMALL_BLOCK_SIZE,
            depth=SMALL_RANDOM_DEPTH,
        ),
        partial(clustered_ghz_qft_circuit, block_size=SMALL_BLOCK_SIZE),
        partial(
            clustered_w_random_circuit,
            block_size=SMALL_BLOCK_SIZE,
            depth=SMALL_RANDOM_DEPTH,
        ),
        partial(clustered_w_qft_circuit, block_size=SMALL_BLOCK_SIZE),
        partial(
            clustered_ghz_random_qft_circuit,
            block_size=SMALL_BLOCK_SIZE,
            depth=SMALL_RANDOM_DEPTH,
        ),
    ],
)
def test_clustered_circuits_stay_within_blocks(constructor):
    circuit = constructor(17)
    metadata = circuit.metadata
    blocks = _cluster_blocks(circuit.num_qubits, metadata["block_size"])
    qubit_to_block = {q: idx for idx, block in enumerate(blocks) for q in block}

    for gate in circuit.gates:
        clusters = {qubit_to_block[q] for q in gate.qubits}
        assert len(clusters) == 1


def test_clustered_ghz_random_parallel_groups_match_blocks():
    circuit = clustered_ghz_random_circuit(
        15,
        block_size=SMALL_BLOCK_SIZE,
        depth=SMALL_RANDOM_DEPTH,
    )
    metadata = circuit.metadata
    blocks = [block for block in _cluster_blocks(circuit.num_qubits, metadata["block_size"]) if block]

    partitioner = Partitioner()
    groups = partitioner.parallel_groups(list(circuit.gates))

    observed = sorted(tuple(sorted(qubits)) for qubits, _ in groups if qubits)
    expected = sorted(tuple(block) for block in blocks)
    assert observed == expected


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


def test_layered_clifford_nonclifford_islands_spacing():
    circuit = layered_clifford_nonclifford_circuit(
        6,
        depth=20,
        fraction_clifford=0.5,
        seed=11,
        islands=3,
        island_len=2,
        island_gap=1,
    )
    metadata = circuit.metadata
    clifford_layers = metadata["clifford_layers"]
    assert clifford_layers == 10
    flags = metadata["non_clifford_layer_flags"]
    assert len(flags) == metadata["depth"]
    assert not any(flags[:clifford_layers])
    expected_post = _expected_island_flags(10, 3, 2, 1)
    assert flags[clifford_layers:] == expected_post
    assert metadata["non_clifford_layers"] == sum(flags)


def test_layered_clifford_magic_islands_helper_overrides_defaults():
    circuit = layered_clifford_magic_islands_circuit(
        5,
        depth=24,
        fraction_clifford=0.5,
        islands=2,
        island_len=3,
        island_gap=4,
        seed=7,
    )
    metadata = circuit.metadata
    clifford_layers = metadata["clifford_layers"]
    assert clifford_layers == 12
    flags = metadata["non_clifford_layer_flags"]
    expected_post = _expected_island_flags(12, 2, 3, 4)
    assert flags[clifford_layers:] == expected_post
    assert metadata["non_clifford_layers"] == sum(flags)


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


def test_classical_controlled_circuit_supports_diag_and_cz_windows() -> None:
    diag_phi = math.pi / 11
    circuit = classical_controlled_circuit(
        7,
        depth=6,
        classical_qubits=3,
        toggle_period=3,
        fanout=2,
        seed=5,
        diag_fixed_phi=diag_phi,
        diag_period=2,
        cz_window_period=3,
    )
    metadata = circuit.metadata
    assert metadata["diag_fixed_phi"] == diag_phi
    assert metadata["diag_period"] == 2
    assert metadata["cz_window_period"] == 3

    offsets = metadata["layer_offsets"]
    diag_period = metadata["diag_period"]
    cz_window_period = metadata["cz_window_period"]

    def _diag_active(layer: int) -> bool:
        return (layer % (2 * diag_period)) < diag_period

    def _cz_active(layer: int) -> bool:
        return (layer % (2 * cz_window_period)) < cz_window_period

    forced_layers = set()
    for layer in range(metadata["depth"]):
        diag_active = _diag_active(layer)
        if diag_active:
            gates = _layer_gates(circuit, offsets, layer)
            rz_gates = [g for g in gates if g.gate == "RZ"]
            assert rz_gates
            for gate in rz_gates:
                assert math.isclose(gate.params["phi"], diag_phi)
        if diag_active or _cz_active(layer):
            forced_layers.add(layer)
            gates = _layer_gates(circuit, offsets, layer)
            assert all(g.gate != "CRZ" for g in gates)

    sample_layer = None
    sample_idx = None
    for layer in range(metadata["depth"]):
        if layer in forced_layers:
            continue
        for idx, _ in enumerate(metadata["classical_qubits"]):
            if (layer + idx) % 3 == 2:
                sample_layer = layer
                sample_idx = idx
                break
        if sample_layer is not None:
            break

    assert sample_layer is not None and sample_idx is not None

    gates = _layer_gates(circuit, offsets, sample_layer)
    quantum_qubits = metadata["quantum_qubits"]
    first_target = quantum_qubits[0]
    rz_gate = next(
        gate for gate in gates if gate.gate == "RZ" and gate.qubits == [first_target]
    )
    expected_phi = ((sample_layer + 1) * (first_target + 2)) % 37 / 37 * math.pi
    assert math.isclose(rz_gate.params["phi"], expected_phi)

    crz_gate = next(gate for gate in gates if gate.gate == "CRZ")
    idx = metadata["classical_qubits"].index(crz_gate.qubits[0])
    assert idx == sample_idx
    expected_crz = ((sample_layer + 1) * (idx + 1)) % 23 / 23 * math.pi
    assert math.isclose(crz_gate.params["phi"], expected_crz)


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


def test_statevector_support_respects_memory_budget() -> None:
    supported, reason = sb._baseline_support_status(
        Backend.STATEVECTOR,
        width=3,
        circuit=None,
        memory_bytes=64,
    )
    assert not supported
    assert reason and "statevector" in reason


def test_tableau_support_rejects_non_clifford() -> None:
    circuit = Circuit([Gate("H", [0]), Gate("T", [0])])
    supported, reason = sb._baseline_support_status(
        Backend.TABLEAU,
        width=circuit.num_qubits,
        circuit=circuit,
        memory_bytes=None,
    )
    assert not supported
    assert reason and "non-Clifford" in reason


def test_tableau_support_rejects_forbidden_gates() -> None:
    circuit = Circuit([Gate("H", [0]), Gate("CCX", [0, 1, 2])])
    supported, reason = sb._baseline_support_status(
        Backend.TABLEAU,
        width=circuit.num_qubits,
        circuit=circuit,
        memory_bytes=None,
    )
    assert not supported
    assert reason and "unsupported" in reason


def test_backend_suite_reuses_single_built_circuit(monkeypatch) -> None:
    spec = sb.ShowcaseCircuit(
        name="dummy",
        display_name="Dummy",
        constructor=lambda width: SimpleNamespace(num_qubits=width, gates=()),
        default_qubits=(4,),
        description="",
    )

    builds: list[SimpleNamespace] = []
    support_calls: list[tuple[Backend, object | None]] = []
    runner_calls: list[tuple[Backend | None, object]] = []

    def fake_build(
        spec_arg: sb.ShowcaseCircuit,
        width: int,
        *,
        classical_simplification: bool,
    ) -> SimpleNamespace:
        assert spec_arg == spec
        circuit = SimpleNamespace(num_qubits=width, gates=())
        builds.append(circuit)
        return circuit

    def fake_support(
        backend: Backend,
        *,
        width: int,
        circuit: object | None,
        memory_bytes: int | None,
    ) -> tuple[bool, str | None]:
        support_calls.append((backend, circuit))
        return True, None

    class DummyRunner:
        def run_quasar_multiple(
            self,
            circuit: object,
            engine: object,
            *,
            backend: Backend | None = None,
            repetitions: int,
            quick: bool,
            memory_bytes: int | None,
            run_timeout: float | None,
        ) -> dict[str, object]:
            runner_calls.append((backend, circuit))
            backend_name = backend.name if backend is not None else "quasar"
            return {
                "framework": backend_name,
                "backend": backend_name,
                "prepare_time_mean": 0.0,
                "prepare_time_std": 0.0,
                "run_time_mean": 0.0,
                "run_time_std": 0.0,
                "total_time_mean": 0.0,
                "total_time_std": 0.0,
                "prepare_peak_memory_mean": 0,
                "prepare_peak_memory_std": 0.0,
                "run_peak_memory_mean": 0,
                "run_peak_memory_std": 0.0,
                "repetitions": repetitions,
                "failed": False,
                "unsupported": False,
                "result": None,
            }

    monkeypatch.setattr(sb, "_build_circuit", fake_build)
    monkeypatch.setattr(sb, "_baseline_support_status", fake_support)
    monkeypatch.setattr(sb, "BenchmarkRunner", lambda: DummyRunner())

    records, messages = sb._run_backend_suite_for_width(
        engine=object(),
        spec=spec,
        width=4,
        repetitions=1,
        run_timeout=None,
        memory_bytes=None,
        classical_simplification=False,
        baseline_backends=(Backend.STATEVECTOR, Backend.MPS),
        quasar_quick=False,
    )

    assert len(builds) == 1
    built = builds[0]
    assert support_calls == [
        (Backend.STATEVECTOR, None),
        (Backend.STATEVECTOR, built),
        (Backend.MPS, None),
        (Backend.MPS, built),
    ]
    assert {id(call[1]) for call in runner_calls} == {id(built)}
    assert len(records) == 3
    assert len(messages) == 3

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


def test_stitched_suite_factories_metadata() -> None:
    specs = stitched_suite.resolve_suite("stitched-big")
    expected_seeds = {
        "stitched_clustered_hybrid": 1337,
        "stitched_layered_magic_islands": 2025,
        "stitched_classical_diag_windows": 424242,
    }
    for spec in specs:
        circuit = spec.factory(12)
        metadata = getattr(circuit, "metadata", {})
        assert metadata.get("seed") == expected_seeds[spec.name]
        assert "layer_offsets" in metadata
        if spec.name == "stitched_clustered_hybrid":
            assert "stage_summaries" in metadata


def test_run_showcase_benchmarks_suite_invokes_factories(monkeypatch, tmp_path) -> None:
    suite_specs = stitched_suite.resolve_suite("stitched-big")
    expected_widths = {spec.name: spec.widths for spec in suite_specs}
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_run_backend_suite(spec, widths, **kwargs):
        calls.append((spec.name, tuple(widths)))
        width = widths[0]
        data = pd.DataFrame(
            {
                "framework": ["baseline", "quasar"],
                "backend": ["baseline_backend", "quasar"],
                "run_time_mean": [2.0, 1.0],
                "total_time_mean": [2.4, 1.2],
                "run_peak_memory_mean": [1.0, 1.0],
                "prepare_peak_memory_mean": [0.5, 0.5],
                "run_time_std": [0.1, 0.1],
                "total_time_std": [0.1, 0.1],
                "run_peak_memory_std": [0.1, 0.1],
                "prepare_peak_memory_std": [0.1, 0.1],
                "qubits": [width, width],
                "result": [None, None],
            }
        )
        return data

    monkeypatch.setattr(sb, "_run_backend_suite", fake_run_backend_suite)
    monkeypatch.setattr(
        sb,
        "compute_baseline_best",
        lambda df, metrics: df[df["framework"] == "baseline"].copy(),
    )
    monkeypatch.setattr(sb, "_export_plot", lambda df, spec, figures_dir, metric: (None, None))

    class DummyDatabase:
        def __init__(self) -> None:
            self.start_kwargs: dict | None = None

        def start_run(self, **kwargs):
            self.start_kwargs = kwargs
            return SimpleNamespace(id=1)

    dummy_db = DummyDatabase()
    monkeypatch.setattr(
        sb, "open_database", lambda path: contextlib.nullcontext(dummy_db)
    )
    monkeypatch.setattr(sb, "FIGURES_DIR", tmp_path)

    args = SimpleNamespace(
        suite="stitched-big",
        circuit_names=None,
        groups=None,
        qubits=None,
        repetitions=1,
        run_timeout=0.0,
        memory_bytes=None,
        enable_classical_simplification=False,
        workers=None,
        metric="run_time_mean",
        database=tmp_path / "suite.sqlite",
    )

    sb.run_showcase_benchmarks(args)

    assert {name for name, _ in calls} == set(expected_widths)
    for name, widths in calls:
        assert widths == expected_widths[name]
    assert dummy_db.start_kwargs is not None
    assert dummy_db.start_kwargs["parameters"]["suite"] == "stitched-big"
