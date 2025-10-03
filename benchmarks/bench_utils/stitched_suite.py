"""Preconfigured showcase suites combining stitched circuit factories."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Tuple

from . import circuits as circuit_lib
from quasar.cost import CostEstimator


LOGGER = logging.getLogger(__name__)


@dataclass
class DisjointSuiteOptions:
    """Configuration controlling stitched-disjoint circuit generation."""

    enforce_disjoint: bool = True
    auto_size_by_ram: bool = True
    max_ram_gb: float = 64.0
    block_size: int | None = None


_DISJOINT_OPTIONS = DisjointSuiteOptions()
_DEFAULT_DISJOINT_BLOCK = 8
_UNSET = object()


def configure_disjoint_suite(
    *,
    enforce_disjoint: bool | None = None,
    auto_size_by_ram: bool | None = None,
    max_ram_gb: float | None = None,
    block_size: int | None | object = _UNSET,
) -> None:
    """Mutate the stitched-disjoint configuration for subsequent factories."""

    if enforce_disjoint is not None:
        _DISJOINT_OPTIONS.enforce_disjoint = bool(enforce_disjoint)
    if auto_size_by_ram is not None:
        _DISJOINT_OPTIONS.auto_size_by_ram = bool(auto_size_by_ram)
    if max_ram_gb is not None:
        _DISJOINT_OPTIONS.max_ram_gb = float(max_ram_gb)
    if block_size is not _UNSET:
        if block_size is None:
            _DISJOINT_OPTIONS.block_size = None
        else:
            if block_size <= 0:
                raise ValueError(
                    "block_size must be positive when overriding disjoint suite"
                )
            _DISJOINT_OPTIONS.block_size = int(block_size)


def current_disjoint_options() -> DisjointSuiteOptions:
    """Return a copy of the active stitched-disjoint configuration."""

    return DisjointSuiteOptions(
        enforce_disjoint=_DISJOINT_OPTIONS.enforce_disjoint,
        auto_size_by_ram=_DISJOINT_OPTIONS.auto_size_by_ram,
        max_ram_gb=_DISJOINT_OPTIONS.max_ram_gb,
        block_size=_DISJOINT_OPTIONS.block_size,
    )


def _max_statevector_qubits(max_ram_gb: float) -> int:
    """Return the largest SV region size that fits in ``max_ram_gb``."""

    if max_ram_gb <= 0:
        return 0
    estimator = CostEstimator()
    max_bytes = float(max_ram_gb) * (1024**3)
    base_mem = float(estimator.coeff.get("sv_base_mem", 0.0))
    avail = max(0.0, max_bytes - base_mem)
    if avail <= 0.0:
        return 0
    bytes_per_amp = float(estimator.coeff.get("sv_bytes_per_amp", 1.0)) * 16.0
    if bytes_per_amp <= 0.0:
        return 0
    amps = avail / bytes_per_amp
    if amps < 1.0:
        return 0
    return max(0, int(math.floor(math.log2(amps))))


def _resolve_disjoint_block_size(num_qubits: int) -> tuple[int, int, int | None]:
    """Return ``(block_size, region_blocks, max_sv_qubits)`` for disjoint builds."""

    options = _DISJOINT_OPTIONS
    block_size = options.block_size or _DEFAULT_DISJOINT_BLOCK
    region_blocks = 1 if options.enforce_disjoint else 3
    max_sv_qubits: int | None = None
    if options.auto_size_by_ram:
        max_sv_qubits = _max_statevector_qubits(options.max_ram_gb)
        if max_sv_qubits <= 0:
            adjusted = 1
        else:
            adjusted = max(1, min(block_size, max_sv_qubits))
        if adjusted < block_size and options.block_size is not None:
            LOGGER.warning(
                "Requested block_size=%d exceeds SV limit (%d qubits); downscaling for stitched-disjoint.",
                block_size,
                max_sv_qubits,
            )
        block_size = adjusted
    block_size = max(1, block_size)
    # Ensure the final block does not exceed the circuit width.
    block_size = min(block_size, max(1, num_qubits))
    return block_size, region_blocks, max_sv_qubits


def verify_disjoint_stitched_circuit(circuit: object, block_size: int) -> None:
    """Assert that ``circuit`` contains no cross-block entangling gates."""

    if block_size <= 0:
        raise ValueError("block_size must be positive for disjoint verification")
    gate_list = getattr(circuit, "gates", None)
    if gate_list is None:
        return
    block_map = {}
    for qubit in range(getattr(circuit, "num_qubits", 0)):
        block_map[qubit] = qubit // block_size
    for gate in gate_list:
        qubits = getattr(gate, "qubits", ())
        if not qubits or len(qubits) <= 1:
            continue
        blocks = {block_map.get(q, q // block_size) for q in qubits}
        if len(blocks) > 1:
            raise ValueError(
                "stitched-disjoint: circuit contains cross-block entangling gates"
            )


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


def _disjoint_factory(
    builder: Callable[..., object],
    *,
    extra_kwargs: Mapping[str, object] | None = None,
    disable_bridges: bool = False,
) -> Callable[[int], object]:
    """Wrap ``builder`` with stitched-disjoint configuration handling."""

    kwargs_template = dict(extra_kwargs or {})

    def factory(width: int) -> object:
        block_size, region_blocks, max_sv_qubits = _resolve_disjoint_block_size(width)
        kwargs = dict(kwargs_template)
        kwargs.setdefault("block_size", block_size)
        if "region_blocks" in builder.__code__.co_varnames:
            kwargs["region_blocks"] = region_blocks
        if disable_bridges and _DISJOINT_OPTIONS.enforce_disjoint:
            kwargs["bridge_layers"] = 0
        circuit = builder(width, **kwargs)
        if _DISJOINT_OPTIONS.enforce_disjoint:
            verify_disjoint_stitched_circuit(circuit, block_size)
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "stitched-disjoint configured block_size=%d region_blocks=%d sv_cap=%s for width=%d",
                block_size,
                region_blocks,
                "none" if max_sv_qubits is None else str(max_sv_qubits),
                width,
            )
        return circuit

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
            factory=_disjoint_factory(
                circuit_lib.clustered_ghz_random_bandedqft_random_circuit,
                extra_kwargs={"region_blocks": 3},
            ),
            widths=(128, 160, 192),
        ),
        StitchedCircuitSpec(
            name="stitched_diag_bandedqft_diag",
            display_name="Diag – Banded QFT – Diag",
            description="Diagonal slabs surrounding bounded banded QFT windows.",
            factory=_disjoint_factory(
                circuit_lib.clustered_ghz_diag_bandedqft_diag_circuit,
                extra_kwargs={"region_blocks": 3},
            ),
            widths=(128, 160),
        ),
        StitchedCircuitSpec(
            name="stitched_rand_bridge_rand",
            display_name="Random – Bridge – Random",
            description="W clusters with neighbour bridges confined to local regions.",
            factory=_disjoint_factory(
                circuit_lib.clustered_w_random_neighborbridge_random_circuit,
                extra_kwargs={"region_blocks": 3, "bridge_layers": 2},
                disable_bridges=True,
            ),
            widths=(192, 256),
        ),
    )


def build_stitched_workqft_suite() -> Tuple[StitchedCircuitSpec, ...]:
    """Return the stitched-workQFT showcase suite specification."""

    return (
        StitchedCircuitSpec(
            name="stitched_rand_workqft_rand",
            display_name="Random – Work-QFT – Random",
            description="Random layers stitched with a bounded work-register QFT window.",
            factory=lambda width: circuit_lib.clustered_ghz_random_workqft_random_circuit(
                width, block_size=8, work_qubits=24
            ),
            widths=(128, 160, 192),
        ),
    )


SUITES: Mapping[str, Callable[[], Tuple[StitchedCircuitSpec, ...]]] = {
    "stitched-big": build_stitched_big_suite,
    "stitched-2x": build_stitched_2x_suite,
    "stitched-disjoint": build_stitched_disjoint_suite,
    "stitched-workqft": build_stitched_workqft_suite,
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
    "configure_disjoint_suite",
    "current_disjoint_options",
    "verify_disjoint_stitched_circuit",
    "resolve_suite",
]

