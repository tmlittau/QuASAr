"""Preconfigured showcase suites combining stitched circuit factories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Tuple

from . import circuits as circuit_lib


@dataclass(frozen=True)
class StitchedCircuitSpec:
    """Description of a stitched showcase circuit entry."""

    name: str
    display_name: str
    description: str
    factory: Callable[[int], object]
    widths: Tuple[int, ...]


def _stitched_clustered_factory(seed: int) -> Callable[[int], object]:
    """Return a GHZ-cluster factory with stitched hybrid stages."""

    def factory(width: int) -> object:
        return circuit_lib.clustered_entanglement_circuit(
            width,
            block_size=6,
            state="ghz",
            entangler="random+diag+global_qft+random",
            depth=(480, 180, 240),
            seed=seed,
        )

    return factory


def _stitched_layered_factory(seed: int) -> Callable[[int], object]:
    """Return a layered Clifford factory with magic-state islands."""

    def factory(width: int) -> object:
        return circuit_lib.layered_clifford_magic_islands_circuit(
            width,
            depth=1800,
            fraction_clifford=0.78,
            islands=18,
            island_len=5,
            island_gap=12,
            seed=seed,
        )

    return factory


def _stitched_classical_factory(seed: int) -> Callable[[int], object]:
    """Return a classical-control factory with stitched diagonal windows."""

    def factory(width: int) -> object:
        classical_qubits = max(10, width // 2)
        if classical_qubits >= width:
            classical_qubits = max(1, width - 2)
        return circuit_lib.classical_controlled_circuit(
            width,
            depth=1800,
            classical_qubits=classical_qubits,
            toggle_period=80,
            fanout=3,
            seed=seed,
            diag_fixed_phi=math.pi / 3,
            diag_period=96,
            cz_window_period=72,
        )

    return factory


def build_stitched_big_suite() -> Tuple[StitchedCircuitSpec, ...]:
    """Return the stitched-big showcase suite specification."""

    return (
        StitchedCircuitSpec(
            name="stitched_clustered_hybrid",
            display_name="Stitched clustered hybrid",
            description="GHZ clusters with stitched random/diag/QFT/random stages.",
            factory=_stitched_clustered_factory(seed=1337),
            widths=(40, 48, 56),
        ),
        StitchedCircuitSpec(
            name="stitched_layered_magic_islands",
            display_name="Stitched layered magic islands",
            description="Layered Clifford transition interleaving spaced magic windows.",
            factory=_stitched_layered_factory(seed=2025),
            widths=(28, 36, 44),
        ),
        StitchedCircuitSpec(
            name="stitched_classical_diag_windows",
            display_name="Stitched classical control windows",
            description="Classical-control ladder stitched with diagonal and CZ windows.",
            factory=_stitched_classical_factory(seed=424242),
            widths=(32, 40, 48),
        ),
    )


def build_stitched_2x_suite() -> Tuple[StitchedCircuitSpec, ...]:
    """Return the stitched-2x showcase suite specification."""

    def _magic_islands_factory(width: int) -> object:
        return circuit_lib.layered_clifford_magic_islands_circuit(width, depth=2000)

    def _diag_qft_diag_factory(width: int) -> object:
        return circuit_lib.clustered_ghz_diag_globalqft_diag_circuit(
            width, block_size=8
        )

    def _rand_qft_rand_factory(width: int) -> object:
        return circuit_lib.clustered_ghz_random_globalqft_random_circuit(
            width, block_size=8
        )

    def _classical_dd_factory(width: int) -> object:
        return circuit_lib.classical_controlled_dd_sandwich_circuit(width)

    def _rand_xburst_rand_factory(width: int) -> object:
        return circuit_lib.clustered_w_random_xburst_random_circuit(
            width, block_size=8
        )

    return (
        StitchedCircuitSpec(
            name="stitched_magic_islands",
            display_name="Magic islands",
            description="Layered Clifford transition with spaced magic windows.",
            factory=_magic_islands_factory,
            widths=(128, 192),
        ),
        StitchedCircuitSpec(
            name="stitched_diag_qft_diag",
            display_name="Diag – QFT – Diag",
            description="Diagonal slabs surrounding a global QFT landing zone.",
            factory=_diag_qft_diag_factory,
            widths=(128, 160, 192),
        ),
        StitchedCircuitSpec(
            name="stitched_rand_qft_rand",
            display_name="Random – QFT – Random",
            description="Random layers with a global QFT spike.",
            factory=_rand_qft_rand_factory,
            widths=(128, 160),
        ),
        StitchedCircuitSpec(
            name="stitched_classical_dd_sandwich",
            display_name="Classical DD sandwich",
            description="Classical controls with DD/Tableau stitched windows.",
            factory=_classical_dd_factory,
            widths=(128, 160),
        ),
        StitchedCircuitSpec(
            name="stitched_rand_xburst_rand",
            display_name="Random – X-burst – Random",
            description="W-state clusters with a cross-block burst landing zone.",
            factory=_rand_xburst_rand_factory,
            widths=(192, 256),
        ),
    )


def build_stitched_disjoint_suite() -> Tuple[StitchedCircuitSpec, ...]:
    """Return the stitched-disjoint showcase suite specification."""

    return (
        StitchedCircuitSpec(
            name="stitched_rand_bandedqft_rand",
            display_name="Random – Banded QFT – Random",
            description="Random layers stitched with bounded QFT regions per cluster.",
            factory=lambda width: circuit_lib.clustered_ghz_random_bandedqft_random_circuit(
                width, block_size=8, region_blocks=3
            ),
            widths=(128, 160, 192),
        ),
        StitchedCircuitSpec(
            name="stitched_diag_bandedqft_diag",
            display_name="Diag – Banded QFT – Diag",
            description="Diagonal slabs surrounding bounded banded QFT windows.",
            factory=lambda width: circuit_lib.clustered_ghz_diag_bandedqft_diag_circuit(
                width, block_size=8, region_blocks=3
            ),
            widths=(128, 160),
        ),
        StitchedCircuitSpec(
            name="stitched_rand_bridge_rand",
            display_name="Random – Bridge – Random",
            description="W clusters with neighbour bridges confined to local regions.",
            factory=lambda width: circuit_lib.clustered_w_random_neighborbridge_random_circuit(
                width, block_size=8, region_blocks=3, bridge_layers=2
            ),
            widths=(192, 256),
        ),
    )


SUITES: Mapping[str, Callable[[], Tuple[StitchedCircuitSpec, ...]]] = {
    "stitched-big": build_stitched_big_suite,
    "stitched-2x": build_stitched_2x_suite,
    "stitched-disjoint": build_stitched_disjoint_suite,
}


def available_suites() -> Tuple[str, ...]:
    """Return the sorted tuple of available stitched suite names."""

    return tuple(sorted(SUITES))


def resolve_suite(name: str) -> Tuple[StitchedCircuitSpec, ...]:
    """Return the stitched suite specification for ``name``."""

    try:
        builder = SUITES[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"unknown suite '{name}'") from exc
    return builder()


__all__ = [
    "StitchedCircuitSpec",
    "available_suites",
    "build_stitched_big_suite",
    "resolve_suite",
]

