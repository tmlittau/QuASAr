"""Partitioning benchmark scenarios derived from the theory notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from quasar.circuit import Circuit

try:  # Allow execution via `python benchmarks/run_benchmarks.py`
    from .large_scale_circuits import (
        dense_to_clifford_partition_circuit,
        dual_magic_injection_circuit,
        staged_partition_circuit,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from large_scale_circuits import (  # type: ignore
        dense_to_clifford_partition_circuit,
        dual_magic_injection_circuit,
        staged_partition_circuit,
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


SCENARIOS: Dict[str, Callable[[], List[WorkloadInstance]]] = {
    "tableau_boundary": _tableau_boundary_sweep,
    "staged_rank": _staged_rank_sweep,
    "staged_sparsity": _staged_sparsity_sweep,
    "dual_magic_injection": _dual_magic_injection_sweep,
}


def iter_scenario(name: str) -> Iterable[WorkloadInstance]:
    try:
        builder = SCENARIOS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown scenario '{name}'") from exc
    return builder()


__all__ = ["WorkloadInstance", "SCENARIOS", "iter_scenario"]
