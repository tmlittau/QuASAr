"""Visualise the SSD of a stitched benchmark circuit."""

from __future__ import annotations

import importlib
from typing import Any, Sequence, cast

from benchmarks.bench_utils import stitched_suite
from quasar.circuit import Circuit


def _require_module(module: str, package: str) -> Any:
    """Import *module* or terminate with a helpful error message."""

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise SystemExit(
            f"This example requires the '{package}' package. Install it to run the SSD visualisation."
        ) from exc


# Ensure the SSD helpers can be imported even when optional dependencies are missing.
_require_module("networkx", "networkx")

from tools.ssd_visualisation import HighlightOptions, compute_layout, draw_ssd_matplotlib, draw_ssd_plotly


def render(width_index: int = 0) -> None:  # pragma: no cover - exercised manually in docs
    """Render stitched SSD layouts with Matplotlib and Plotly."""

    specs = stitched_suite.resolve_suite("stitched-big")
    spec = specs[0]
    widths: Sequence[int] = spec.widths
    try:
        width = widths[width_index]
    except IndexError as exc:
        raise ValueError(f"invalid width index {width_index!r}; available indices: 0..{len(widths) - 1}") from exc

    circuit = cast(Circuit, spec.factory(width))
    graph = circuit.to_networkx_ssd(include_conversions=True, include_backends=True)
    layout = compute_layout(graph)
    highlight = HighlightOptions(boundary_qubit_threshold=32, long_range_threshold=8, only_problematic=True)

    plt = _require_module("matplotlib.pyplot", "matplotlib")
    ax = draw_ssd_matplotlib(graph, layout=layout, highlight=highlight)
    ax.set_title(f"{spec.display_name} (width={width})")
    plt.tight_layout()
    plt.show()

    _require_module("plotly.graph_objects", "plotly")
    fig = draw_ssd_plotly(graph, layout=layout, highlight=highlight)
    fig.update_layout(title=f"{spec.display_name} (width={width})")
    fig.show()


if __name__ == "__main__":
    render()
