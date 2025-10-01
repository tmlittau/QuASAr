"""Example usage of the SSD visualisation helpers."""

from __future__ import annotations

from quasar.circuit import Circuit
from quasar.ssd import SSD

from tools.ssd_visualisation import HighlightOptions, compute_layout, draw_ssd_matplotlib


def build_sample_ssd() -> SSD:
    """Return a compact SSD showcasing conversions and entanglement."""

    circuit = Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "H", "qubits": [5]},
            {"gate": "CX", "qubits": [0, 5]},
            {"gate": "SWAP", "qubits": [1, 4]},
            {"gate": "CX", "qubits": [2, 3]},
            {"gate": "SWAP", "qubits": [3, 6]},
        ],
        use_classical_simplification=False,
    )
    return circuit.ssd


def render() -> None:  # pragma: no cover - exercised manually in docs
    """Plot the SSD with long-range entanglement emphasised."""

    import matplotlib.pyplot as plt

    ssd = build_sample_ssd()
    graph = ssd.to_networkx()
    layout = compute_layout(graph)
    options = HighlightOptions(long_range_threshold=3, boundary_qubit_threshold=2)
    draw_ssd_matplotlib(graph, layout=layout, highlight=options)
    plt.show()


if __name__ == "__main__":
    render()
