"""Render SSD visualisations for showcase benchmarks and a schematic example.

This helper demonstrates how to obtain the subsystem descriptor (SSD) for a
showcase benchmark circuit and how to display it using the layout utilities in
:mod:`tools.ssd_visualisation`.  It also constructs a compact synthetic SSD with
explicit dependencies to illustrate the partitioning mechanics at a glance.

Run the script either as ``python -m benchmarks.visualise_ssd`` or directly via
``python benchmarks/visualise_ssd.py``.  Both figures can be written to disk or
shown interactively.  Matplotlib and NetworkX are required; install Plotly if
an interactive view is desired.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from quasar import SimulationEngine
from quasar.cost import Backend, Cost
from quasar.ssd import SSD, SSDPartition
from tools.ssd_visualisation import HighlightOptions, compute_layout, draw_ssd_matplotlib

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


if __package__ in {None, ""}:  # pragma: no cover - manual execution helper
    for path in (PACKAGE_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from bench_utils.showcase_benchmarks import SHOWCASE_CIRCUITS, ShowcaseCircuit
else:  # pragma: no cover - exercised when used as a package module
    from .bench_utils.showcase_benchmarks import SHOWCASE_CIRCUITS, ShowcaseCircuit


@dataclass
class RenderTarget:
    """Container describing how a figure should be emitted."""

    output: Optional[Path]
    title: str
    include_conversions: bool
    include_backends: bool
    highlight: HighlightOptions
    partition_gap: float
    figsize: tuple[float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="layered_clifford_midpoint",
        choices=sorted(SHOWCASE_CIRCUITS.keys()),
        help="Name of the showcase circuit to simulate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help=(
            "Number of qubits for the benchmark circuit. Defaults to the"
            " narrowest width advertised by the showcase specification."
        ),
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        help=(
            "Optional path for the benchmark SSD figure (format inferred from "
            "extension)."
        ),
    )
    parser.add_argument(
        "--example-output",
        type=Path,
        help=(
            "Optional path for the schematic SSD figure (format inferred from "
            "extension)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures after rendering even when outputs are provided.",
    )
    parser.add_argument(
        "--benchmark-partition-gap",
        type=float,
        default=2.8,
        help="Horizontal spacing between partitions in the benchmark layout.",
    )
    parser.add_argument(
        "--example-partition-gap",
        type=float,
        default=2.5,
        help="Horizontal spacing between partitions in the schematic layout.",
    )
    parser.add_argument(
        "--long-range-threshold",
        type=int,
        default=6,
        help=(
            "Minimum qubit distance highlighted as long-range entanglement in "
            "the benchmark figure."
        ),
    )
    parser.add_argument(
        "--boundary-threshold",
        type=int,
        default=6,
        help=(
            "Boundary qubit threshold used to highlight wide interfaces in the "
            "benchmark figure."
        ),
    )
    return parser.parse_args()


def _resolve_showcase(name: str) -> ShowcaseCircuit:
    try:
        return SHOWCASE_CIRCUITS[name]
    except KeyError as exc:  # pragma: no cover - guarded by argparse
        available = ", ".join(sorted(SHOWCASE_CIRCUITS))
        raise SystemExit(f"Unknown showcase circuit '{name}'. Available: {available}") from exc


def _simulate_benchmark(spec: ShowcaseCircuit, width: Optional[int]) -> SSD:
    """Run the selected showcase circuit and return its SSD."""

    qubits = width or spec.default_qubits[0]
    circuit = spec.constructor(qubits)
    engine = SimulationEngine()
    result = engine.simulate(circuit)
    return result.ssd


def _build_schematic_ssd() -> SSD:
    """Return a hand-crafted SSD emphasising partition dependencies."""

    partitions: list[SSDPartition] = [
        SSDPartition(
            subsystems=((0,),),
            history=("H",),
            backend=Backend.TABLEAU,
            cost=Cost(time=0.1, memory=0.5),
            dependencies=(),
            entangled_with=(2,),
            boundary_qubits=(0,),
        ),
        SSDPartition(
            subsystems=((3,),),
            history=("RX",),
            backend=Backend.STATEVECTOR,
            cost=Cost(time=0.2, memory=1.0),
            dependencies=(),
            entangled_with=(3,),
            boundary_qubits=(3,),
        ),
        SSDPartition(
            subsystems=((0, 1),),
            history=("CX",),
            backend=Backend.MPS,
            cost=Cost(time=0.3, memory=0.8),
            dependencies=(0,),
            entangled_with=(0, 3),
            boundary_qubits=(0, 1),
            rank=2,
        ),
        SSDPartition(
            subsystems=((1, 3),),
            history=("CZ",),
            backend=Backend.DECISION_DIAGRAM,
            cost=Cost(time=0.25, memory=0.6),
            dependencies=(1, 2),
            entangled_with=(1, 2),
            boundary_qubits=(1, 3),
            rank=2,
        ),
    ]
    schematic = SSD(partitions)
    schematic.build_metadata()
    return schematic


def _render_ssd(ssd: SSD, target: RenderTarget) -> None:
    """Render ``ssd`` according to ``target`` and optionally save the figure."""

    import matplotlib.pyplot as plt

    graph = ssd.to_networkx(
        include_dependencies=True,
        include_entanglement=True,
        include_conversions=target.include_conversions,
        include_backends=target.include_backends,
    )
    layout = compute_layout(graph, partition_gap=target.partition_gap)

    fig, ax = plt.subplots(figsize=target.figsize)
    draw_ssd_matplotlib(
        graph,
        layout=layout,
        highlight=target.highlight,
        ax=ax,
        node_size=800,
    )
    ax.set_title(target.title)
    fig.tight_layout()

    if target.output is not None:
        target.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target.output, bbox_inches="tight")
        print(f"Saved {target.title} figure to {target.output}")
        plt.close(fig)


def main() -> None:  # pragma: no cover - invoked manually
    args = _parse_args()
    spec = _resolve_showcase(args.benchmark)

    benchmark_ssd = _simulate_benchmark(spec, args.width)
    schematic_ssd = _build_schematic_ssd()

    benchmark_target = RenderTarget(
        output=args.benchmark_output,
        title=f"SSD for {spec.display_name}",
        include_conversions=True,
        include_backends=True,
        highlight=HighlightOptions(
            long_range_threshold=args.long_range_threshold,
            boundary_qubit_threshold=args.boundary_threshold,
        ),
        partition_gap=args.benchmark_partition_gap,
        figsize=(12.0, 6.5),
    )

    schematic_target = RenderTarget(
        output=args.example_output,
        title="Schematic partition layout",
        include_conversions=False,
        include_backends=False,
        highlight=HighlightOptions(boundary_qubit_threshold=2),
        partition_gap=args.example_partition_gap,
        figsize=(7.0, 4.0),
    )

    _render_ssd(benchmark_ssd, benchmark_target)
    _render_ssd(schematic_ssd, schematic_target)

    if args.show or not (args.benchmark_output and args.example_output):
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":  # pragma: no cover - module intended for manual execution
    main()
