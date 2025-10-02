"""Minimal SSD partition visualisation focusing on partition structure."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from quasar.circuit import Circuit
from quasar.ssd import SSD

from tools.ssd_visualisation import HighlightOptions, compute_layout, draw_ssd_matplotlib


def build_minimal_ssd() -> SSD:
    """Return a compact SSD with a few clearly separated partitions."""

    circuit = Circuit(
        [
            {"gate": "H", "qubits": [0]},  # isolated single-qubit fragment
            {"gate": "RX", "qubits": [2], "params": {"theta": 0.5}},  # non-Clifford fragment
            {"gate": "CX", "qubits": [0, 1]},  # entangling bridge between partitions
        ],
        use_classical_simplification=False,
    )
    return circuit.ssd


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the rendered figure (format inferred from extension).",
    )
    return parser.parse_args()


def render(output: Optional[Path]) -> None:  # pragma: no cover - exercised manually
    """Render the SSD diagram, optionally writing it to disk."""

    import matplotlib.pyplot as plt

    ssd = build_minimal_ssd()
    graph = ssd.to_networkx(include_conversions=False, include_backends=False)
    layout = compute_layout(graph)

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_ssd_matplotlib(
        graph,
        layout=layout,
        highlight=HighlightOptions(long_range_threshold=None, boundary_qubit_threshold=None),
        ax=ax,
        node_size=700,
    )
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:  # pragma: no cover - exercised manually
    args = _parse_args()
    render(args.output)


if __name__ == "__main__":  # pragma: no cover - module intended for manual execution
    main()
