"""Partitioning benchmark scenarios derived from the theory notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from quasar.circuit import Circuit

try:  # Allow execution via `python benchmarks/run_benchmark.py`
    from .large_scale_circuits import (
        alternating_ladder_circuit,
        dense_to_clifford_partition_circuit,
        dual_magic_injection_circuit,
        staged_partition_circuit,
        w_state_phase_oracle_circuit,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from large_scale_circuits import (  # type: ignore
        alternating_ladder_circuit,
        dense_to_clifford_partition_circuit,
        dual_magic_injection_circuit,
        staged_partition_circuit,
        w_state_phase_oracle_circuit,
    )


@dataclass(frozen=True)
class WorkloadInstance:
    """Concrete workload instance with metadata about the swept parameters."""

    scenario: str
    variant: str
    builder: Callable[..., Circuit]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    enable_classical_simplification: bool = False

    def build(self) -> Circuit:
        circuit = self.builder(**self.parameters)
        meta = dict(self.metadata)
        meta.setdefault("scenario", self.scenario)
        meta.setdefault("variant", self.variant)
        setattr(circuit, "metadata", meta)
        return circuit


def _tableau_boundary_sweep() -> List[WorkloadInstance]:
    dense = 4
    clifford = 3
    combos = [
        {"boundary": 2, "schmidt_layers": 1},
        {"boundary": 2, "schmidt_layers": 2},
        {"boundary": 3, "schmidt_layers": 2},
    ]
    instances: List[WorkloadInstance] = []
    for idx, cfg in enumerate(combos, start=1):
        params = dict(cfg)
        params.update(
            {
                "dense_qubits": dense,
                "clifford_qubits": clifford,
                "prefix_layers": 1,
                "clifford_layers": 3,
            }
        )
        metadata = {
            "dense_qubits": dense,
            "clifford_qubits": clifford,
            "boundary": cfg["boundary"],
            "schmidt_layers": cfg["schmidt_layers"],
            "total_qubits": dense + clifford,
        }
        instances.append(
            WorkloadInstance(
                scenario="tableau_boundary",
                variant=f"tableau_boundary_{idx}",
                builder=dense_to_clifford_partition_circuit,
                parameters=params,
                metadata=metadata,
                enable_classical_simplification=False,
            )
        )
    return instances


def _staged_rank_sweep() -> List[WorkloadInstance]:
    clifford = 4
    core = 3
    suffix = 2
    base = {
        "clifford_qubits": clifford,
        "core_qubits": core,
        "suffix_qubits": suffix,
        "prefix_core_boundary": 2,
        "core_suffix_boundary": 1,
        "suffix_sparsity": 0.6,
        "prefix_depth": 2,
        "core_layers": 2,
        "suffix_layers": 2,
    }
    layers = [1, 2, 3]
    instances: List[WorkloadInstance] = []
    for idx, cross_layers in enumerate(layers, start=1):
        params = dict(base, cross_layers=cross_layers)
        metadata = {
            "clifford_qubits": clifford,
            "core_qubits": core,
            "suffix_qubits": suffix,
            "prefix_core_boundary": base["prefix_core_boundary"],
            "core_suffix_boundary": base["core_suffix_boundary"],
            "cross_layers": cross_layers,
            "suffix_sparsity": base["suffix_sparsity"],
            "total_qubits": clifford + core + suffix,
        }
        instances.append(
            WorkloadInstance(
                scenario="staged_rank",
                variant=f"staged_rank_{idx}",
                builder=staged_partition_circuit,
                parameters=params,
                metadata=metadata,
                enable_classical_simplification=False,
            )
        )
    return instances


def _staged_sparsity_sweep() -> List[WorkloadInstance]:
    clifford = 5
    core = 3
    suffix = 2
    base = {
        "clifford_qubits": clifford,
        "core_qubits": core,
        "suffix_qubits": suffix,
        "prefix_core_boundary": 2,
        "core_suffix_boundary": 2,
        "cross_layers": 2,
    }
    sparsities = [0.2, 0.5, 0.8]
    instances: List[WorkloadInstance] = []
    for idx, sparsity in enumerate(sparsities, start=1):
        params = dict(base, suffix_sparsity=sparsity)
        metadata = {
            "clifford_qubits": clifford,
            "core_qubits": core,
            "suffix_qubits": suffix,
            "prefix_core_boundary": base["prefix_core_boundary"],
            "core_suffix_boundary": base["core_suffix_boundary"],
            "cross_layers": base["cross_layers"],
            "suffix_sparsity": sparsity,
            "total_qubits": clifford + core + suffix,
        }
        instances.append(
            WorkloadInstance(
                scenario="staged_sparsity",
                variant=f"staged_sparsity_{idx}",
                builder=staged_partition_circuit,
                parameters=params,
                metadata=metadata,
                enable_classical_simplification=False,
            )
        )
    return instances


def _dual_magic_injection_sweep() -> List[WorkloadInstance]:
    patch_distance = 8
    gadget = "t_bridge"
    scheme = "repetition"
    boundary_widths = [6, 7, 8]
    stabilizer_depths = [2, 4]
    instances: List[WorkloadInstance] = []
    variant = 1
    for boundary_width in boundary_widths:
        for stabilizer_rounds in stabilizer_depths:
            params = {
                "patch_distance": patch_distance,
                "stabilizer_rounds": stabilizer_rounds,
                "gadget_width": boundary_width,
                "gadget": gadget,
                "scheme": scheme,
            }
            preview = dual_magic_injection_circuit(**params)
            total_qubits = getattr(preview, "num_qubits", None)
            if total_qubits is None:
                total_qubits = preview.num_qubits if hasattr(preview, "num_qubits") else 0
            data_qubits = patch_distance if scheme == "repetition" else patch_distance * patch_distance
            gadget_qubits = min(boundary_width, data_qubits)
            metadata = {
                "patch_distance": patch_distance,
                "boundary_width": boundary_width,
                "gadget_width": gadget_qubits,
                "stabilizer_rounds": stabilizer_rounds,
                "gadget": gadget,
                "scheme": scheme,
                "total_qubits": total_qubits,
            }
            instances.append(
                WorkloadInstance(
                    scenario="dual_magic_injection",
                    variant=f"dual_magic_injection_{variant}",
                    builder=dual_magic_injection_circuit,
                    parameters=params,
                    metadata=metadata,
                    enable_classical_simplification=False,
                )
            )
            variant += 1
    return instances


def _ladder_dense_gadgets_sweep() -> List[WorkloadInstance]:
    chain_length = 18
    gadget_width = 4
    ladder_layers = 3
    dense_gadget_counts = [2, 3, 5]
    gadget_depths = [1, 2, 3]
    instances: List[WorkloadInstance] = []
    variant = 1
    for dense_count in dense_gadget_counts:
        for gadget_layers in gadget_depths:
            params = {
                "chain_length": chain_length,
                "dense_gadgets": dense_count,
                "gadget_width": gadget_width,
                "ladder_layers": ladder_layers,
                "gadget_layers": gadget_layers,
                "seed": variant,
            }
            boundary_width = min(gadget_width, chain_length)
            chi_cap = 2 ** boundary_width
            chi_depth = 2 ** gadget_layers
            chi_target = min(chi_cap, chi_depth)
            spacing = chain_length / dense_count if dense_count else None
            metadata = {
                "chain_length": chain_length,
                "total_qubits": chain_length * 2,
                "dense_gadgets": dense_count,
                "gadget_spacing": spacing,
                "ladder_layers": ladder_layers,
                "gadget_layers": gadget_layers,
                "gadget_width": min(gadget_width, chain_length),
                "gadget_size": min(gadget_width, chain_length),
                "boundary_width": boundary_width,
                "chi_target": chi_target,
            }
            instances.append(
                WorkloadInstance(
                    scenario="ladder_dense_gadgets",
                    variant=f"ladder_dense_gadgets_{variant}",
                    builder=alternating_ladder_circuit,
                    parameters=params,
                    metadata=metadata,
                    enable_classical_simplification=False,
                )
            )
            variant += 1
    return instances


def _w_state_oracle_sweep() -> List[WorkloadInstance]:
    width = 18
    oracle_depths = [1, 2, 4]
    rotation_choices = [
        ("rz",),
        ("rz", "ry"),
        ("rz", "ry", "rx", "t"),
    ]

    instances: List[WorkloadInstance] = []
    variant = 1
    for oracle_layers in oracle_depths:
        for rotation_set in rotation_choices:
            params = {
                "width": width,
                "oracle_layers": oracle_layers,
                "rotation_set": rotation_set,
                "seed": variant,
            }
            preview = w_state_phase_oracle_circuit(**params)
            base = w_state_phase_oracle_circuit(
                width=width, oracle_layers=0, rotation_set=rotation_set, seed=variant
            )
            prefix_len = len(base.gates)
            oracle_gates = list(preview.gates[prefix_len:])
            rotation_names = {
                "RZ",
                "RY",
                "RX",
                "P",
                "PHASE",
                "T",
                "S",
                "SDG",
            }
            rotation_gate_count = sum(
                1
                for gate in oracle_gates
                if len(gate.qubits) == 1 and gate.gate in rotation_names
            )
            parameterised_rotations = sum(1 for gate in oracle_gates if gate.params)
            entangling_gates = [
                gate
                for gate in oracle_gates
                if len(gate.qubits) > 1 and gate.gate in {"CZ", "CRZ"}
            ]
            entangling_count = len(entangling_gates)
            rotation_per_layer = (
                rotation_gate_count / oracle_layers if oracle_layers > 0 else 0.0
            )
            entangling_per_layer = (
                entangling_count / oracle_layers if oracle_layers > 0 else 0.0
            )
            rotation_density = (
                rotation_per_layer / width if width > 0 else 0.0
            )
            oracle_sparsity = max(0.0, 1.0 - min(1.0, rotation_density))
            rotation_unique = sorted(
                {
                    gate.gate
                    for gate in oracle_gates
                    if len(gate.qubits) == 1 and gate.gate in rotation_names
                }
            )
            metadata = {
                "total_qubits": width,
                "w_state_width": width,
                "oracle_layers": oracle_layers,
                "oracle_rotation_gate_count": rotation_gate_count,
                "oracle_rotation_unique": ",".join(rotation_unique),
                "oracle_rotation_density": rotation_density,
                "oracle_sparsity": oracle_sparsity,
                "oracle_entangling_count": entangling_count,
                "oracle_entangling_per_layer": entangling_per_layer,
                "oracle_rotation_per_layer": rotation_per_layer,
                "oracle_parameterised_rotations": parameterised_rotations,
                "rotation_set": ",".join(rotation_set),
            }
            instances.append(
                WorkloadInstance(
                    scenario="w_state_oracle",
                    variant=f"w_state_oracle_{variant}",
                    builder=w_state_phase_oracle_circuit,
                    parameters=params,
                    metadata=metadata,
                    enable_classical_simplification=False,
                )
            )
            variant += 1
    return instances


SCENARIOS: Dict[str, Callable[[], List[WorkloadInstance]]] = {
    "tableau_boundary": _tableau_boundary_sweep,
    "staged_rank": _staged_rank_sweep,
    "staged_sparsity": _staged_sparsity_sweep,
    "dual_magic_injection": _dual_magic_injection_sweep,
    "ladder_dense_gadgets": _ladder_dense_gadgets_sweep,
    "w_state_oracle": _w_state_oracle_sweep,
}


def iter_scenario(name: str) -> Iterable[WorkloadInstance]:
    try:
        builder = SCENARIOS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown scenario '{name}'") from exc
    return builder()


__all__ = ["WorkloadInstance", "SCENARIOS", "iter_scenario"]
