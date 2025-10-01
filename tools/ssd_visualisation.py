"""Visualisation helpers for :meth:`SSD.to_networkx` graphs.

These utilities position partitions, conversions and backends on
predictable rows so that entanglement and backend transitions can be
inspected visually.  Matplotlib and Plotly frontends are provided for
quick exploration inside notebooks and scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import importlib

import networkx as nx

Node = Tuple[str, Any]
Position = Tuple[float, float]
EdgeKey = Tuple[Node, Node, int]


@dataclass(frozen=True)
class HighlightOptions:
    """Configuration flags for rendering SSD graphs.

    Attributes
    ----------
    long_range_threshold:
        Minimum Manhattan distance between qubits of two partitions for
        an entanglement edge to be considered *long range*.  Distances
        are computed from the minimum absolute difference between qubit
        indices.  When ``None`` all entanglement edges are treated
        uniformly.
    boundary_qubit_threshold:
        Threshold used to mark partitions or conversions with a large
        boundary.  When provided, nodes exceeding the threshold are
        highlighted and, if ``only_problematic`` is set, are the only
        boundary elements rendered.
    only_problematic:
        When ``True`` only elements marked as long-range entanglement or
        boundary hotspots are drawn.  Useful when working with dense SSDs
        where the full graph becomes cluttered.
    """

    long_range_threshold: Optional[int] = None
    boundary_qubit_threshold: Optional[int] = None
    only_problematic: bool = False


def _require_matplotlib() -> Any:
    """Return the :mod:`matplotlib.pyplot` module if available."""

    return importlib.import_module("matplotlib.pyplot")


def _require_plotly() -> Any:
    """Return the :mod:`plotly.graph_objects` module if available."""

    return importlib.import_module("plotly.graph_objects")


def compute_layout(
    graph: nx.MultiDiGraph,
    *,
    partition_gap: float = 2.5,
    backend_y: float = 3.0,
    conversion_y: float = 1.5,
) -> Dict[Node, Position]:
    """Compute a deterministic layout for an SSD graph.

    Partitions are arranged along the x-axis ordered by their ``index``
    attribute.  Conversion layers are positioned slightly above the
    partitions while backend nodes reside at the top of the diagram.

    Parameters
    ----------
    graph:
        Graph returned by :meth:`SSD.to_networkx`.
    partition_gap:
        Horizontal spacing between partitions.
    backend_y:
        Vertical position for backend nodes.
    conversion_y:
        Vertical position for conversion layer nodes.

    Returns
    -------
    dict
        Mapping of node identifiers to ``(x, y)`` coordinates.
    """

    positions: Dict[Node, Position] = {}

    partitions: List[Tuple[Node, Mapping[str, Any]]] = [
        (node, data)
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "partition"
    ]
    partitions.sort(key=lambda item: item[1].get("index", 0))

    for x_idx, (node, _) in enumerate(partitions):
        positions[node] = (x_idx * partition_gap, 0.0)

    conversions: List[Tuple[Node, Mapping[str, Any]]] = [
        (node, data)
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "conversion"
    ]
    conversions.sort(key=lambda item: item[1].get("index", 0))

    for node, _ in conversions:
        neighbours = [
            positions[nb]
            for nb in graph.predecessors(node)
            if nb in positions
        ]
        if neighbours:
            x_pos = sum(x for x, _ in neighbours) / len(neighbours)
        else:
            successors = [
                positions[nb]
                for nb in graph.successors(node)
                if nb in positions
            ]
            if successors:
                x_pos = sum(x for x, _ in successors) / len(successors)
            else:
                x_pos = len(positions) * partition_gap
        positions[node] = (x_pos, conversion_y)

    backends: List[Tuple[Node, Mapping[str, Any]]] = [
        (node, data)
        for node, data in graph.nodes(data=True)
        if data.get("kind") == "backend"
    ]
    backends.sort(key=lambda item: item[1].get("backend", ""))

    if partitions:
        min_x = min(x for x, _ in positions.values())
        max_x = max(x for x, _ in positions.values())
    else:
        min_x = 0.0
        max_x = partition_gap * max(len(backends) - 1, 0)

    for idx, (node, _) in enumerate(backends):
        if backends:
            x_pos = min_x + (idx / max(len(backends) - 1, 1)) * (max_x - min_x)
        else:
            x_pos = 0.0
        positions[node] = (x_pos, backend_y)

    # Include any remaining nodes (e.g. synthetic metadata nodes) at the
    # origin to avoid networkx draw errors.
    for node in graph.nodes:
        positions.setdefault(node, (0.0, 0.0))

    return positions


def _qubit_distance(part_a: Sequence[int], part_b: Sequence[int]) -> int:
    if not part_a or not part_b:
        return 0
    return min(abs(a - b) for a in part_a for b in part_b)


def _long_range_edges(
    graph: nx.MultiDiGraph, threshold: Optional[int]
) -> set[EdgeKey]:
    highlighted: set[EdgeKey] = set()
    if threshold is None:
        return highlighted

    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("kind") != "entanglement":
            continue
        part_a = graph.nodes[u].get("qubits", ())
        part_b = graph.nodes[v].get("qubits", ())
        distance = _qubit_distance(tuple(part_a), tuple(part_b))
        if distance >= threshold:
            highlighted.add((u, v, key))
            highlighted.add((v, u, key))
    return highlighted


def _boundary_hotspots(
    graph: nx.MultiDiGraph, threshold: Optional[int]
) -> Tuple[set[Node], set[Node]]:
    if threshold is None:
        return set(), set()

    partition_nodes: set[Node] = set()
    conversion_nodes: set[Node] = set()
    for node, data in graph.nodes(data=True):
        if data.get("kind") == "partition":
            boundary = data.get("boundary_qubits", ())
            if boundary and len(boundary) >= threshold:
                partition_nodes.add(node)
        elif data.get("kind") == "conversion":
            boundary = data.get("boundary", ())
            if boundary and len(boundary) >= threshold:
                conversion_nodes.add(node)
    return partition_nodes, conversion_nodes


def draw_ssd_matplotlib(
    graph: nx.MultiDiGraph,
    *,
    layout: Optional[Mapping[Node, Position]] = None,
    highlight: HighlightOptions | None = None,
    ax: Any | None = None,
    node_size: int = 900,
) -> Any:
    """Render an SSD graph using :mod:`matplotlib`.

    Parameters
    ----------
    graph:
        Graph produced by :meth:`SSD.to_networkx`.
    layout:
        Optional precomputed layout mapping.
    highlight:
        Highlight configuration.  When ``None`` no special filtering is
        applied.
    ax:
        Optional Matplotlib axes to draw on.  When ``None`` a new figure
        and axes are created.
    node_size:
        Size of rendered nodes in points^2.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the rendered diagram.
    """

    plt = _require_matplotlib()

    highlight = highlight or HighlightOptions()
    layout = dict(layout or compute_layout(graph))

    long_range = _long_range_edges(graph, highlight.long_range_threshold)
    boundary_partitions, boundary_conversions = _boundary_hotspots(
        graph, highlight.boundary_qubit_threshold
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    ax.set_axis_off()

    def _filter_edge(u: Node, v: Node, key: int, data: Mapping[str, Any]) -> bool:
        if not highlight.only_problematic:
            return True
        if (u, v, key) in long_range or (v, u, key) in long_range:
            return True
        if u in boundary_partitions or v in boundary_partitions:
            return True
        if u in boundary_conversions or v in boundary_conversions:
            return True
        return False

    node_colors: Dict[Node, str] = {}
    node_labels: Dict[Node, str] = {}
    partitions: List[Node] = []
    conversions: List[Node] = []
    backends: List[Node] = []

    for node, data in graph.nodes(data=True):
        kind = data.get("kind")
        if kind == "partition":
            partitions.append(node)
            node_colors[node] = "#4C78A8"
            node_labels[node] = f"P{data.get('index', '?')}"
            if node in boundary_partitions:
                node_colors[node] = "#E45756"
        elif kind == "conversion":
            conversions.append(node)
            node_colors[node] = "#F58518"
            node_labels[node] = data.get("primitive", "conv")
            if node in boundary_conversions:
                node_colors[node] = "#E45756"
        elif kind == "backend":
            backends.append(node)
            node_colors[node] = "#72B7B2"
            node_labels[node] = data.get("label", data.get("backend", "backend"))
        else:
            node_colors[node] = "#B0BEC5"
            node_labels[node] = str(node)

    def _draw_nodes(nodes: Iterable[Node]) -> None:
        if not nodes:
            return
        nx.draw_networkx_nodes(
            graph,
            layout,
            nodelist=list(nodes),
            node_color=[node_colors[node] for node in nodes],
            node_size=node_size,
            ax=ax,
        )

    _draw_nodes(partitions)
    _draw_nodes(conversions)
    _draw_nodes(backends)

    nx.draw_networkx_labels(graph, layout, labels=node_labels, font_size=10, ax=ax)

    dependency_edges: List[Tuple[Node, Node]] = []
    backend_edges: List[Tuple[Node, Node]] = []
    conversion_edges: List[Tuple[Node, Node]] = []
    entanglement_edges: List[Tuple[Node, Node]] = []
    long_range_edges: List[Tuple[Node, Node]] = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        if not _filter_edge(u, v, key, data):
            continue
        kind = data.get("kind")
        if kind == "dependency":
            dependency_edges.append((u, v))
        elif kind == "backend_assignment" or kind == "conversion_source" or kind == "conversion_target":
            backend_edges.append((u, v))
        elif kind == "conversion_boundary":
            conversion_edges.append((u, v))
        elif kind == "entanglement":
            if (u, v, key) in long_range or (v, u, key) in long_range:
                long_range_edges.append((u, v))
            else:
                entanglement_edges.append((u, v))
        else:
            conversion_edges.append((u, v))

    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=dependency_edges,
        edge_color="#9E9E9E",
        width=1.5,
        arrows=True,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=backend_edges,
        edge_color="#54A24B",
        width=1.5,
        arrows=True,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=conversion_edges,
        edge_color="#EECA3B",
        width=1.5,
        arrows=True,
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=entanglement_edges,
        edge_color="#B279A2",
        width=2.0,
        arrows=False,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=long_range_edges,
        edge_color="#FF7F0E",
        width=2.5,
        arrows=False,
        style="dotted",
        ax=ax,
    )

    ax.set_title("SSD visualisation")
    return ax


def draw_ssd_plotly(
    graph: nx.MultiDiGraph,
    *,
    layout: Optional[Mapping[Node, Position]] = None,
    highlight: HighlightOptions | None = None,
) -> Any:
    """Render an SSD graph as an interactive Plotly figure."""

    go = _require_plotly()

    highlight = highlight or HighlightOptions()
    layout = dict(layout or compute_layout(graph))

    long_range = _long_range_edges(graph, highlight.long_range_threshold)
    boundary_partitions, boundary_conversions = _boundary_hotspots(
        graph, highlight.boundary_qubit_threshold
    )

    def include_node(node: Node) -> bool:
        if not highlight.only_problematic:
            return True
        if node in boundary_partitions or node in boundary_conversions:
            return True
        for _, nb, key, _ in graph.out_edges(node, keys=True, data=True):
            if (node, nb, key) in long_range:
                return True
        for nb, _, key, _ in graph.in_edges(node, keys=True, data=True):
            if (nb, node, key) in long_range:
                return True
        return False

    def include_edge(u: Node, v: Node, key: int, data: Mapping[str, Any]) -> bool:
        if not highlight.only_problematic:
            return True
        if (u, v, key) in long_range or (v, u, key) in long_range:
            return True
        if u in boundary_partitions or v in boundary_partitions:
            return True
        if u in boundary_conversions or v in boundary_conversions:
            return True
        return False

    nodes_by_kind: MutableMapping[str, List[Node]] = {"partition": [], "conversion": [], "backend": [], "other": []}
    node_labels: Dict[Node, str] = {}

    for node, data in graph.nodes(data=True):
        if not include_node(node):
            continue
        kind = data.get("kind", "other")
        kind = kind if kind in nodes_by_kind else "other"
        nodes_by_kind[kind].append(node)
        if kind == "partition":
            node_labels[node] = f"P{data.get('index', '?')}"\
                + ("\nBoundary=" + str(len(data.get("boundary_qubits", ()))) if data.get("boundary_qubits") else "")
        elif kind == "conversion":
            node_labels[node] = data.get("primitive", "conversion") + "\n" + data.get("source", "") + "→" + data.get("target", "")
        elif kind == "backend":
            node_labels[node] = data.get("label", data.get("backend", "backend"))
        else:
            node_labels[node] = str(node)

    color_map = {
        "partition": "#4C78A8",
        "conversion": "#F58518",
        "backend": "#72B7B2",
        "other": "#B0BEC5",
    }

    fig = go.Figure()

    for kind, nodes in nodes_by_kind.items():
        if not nodes:
            continue
        xs = [layout[node][0] for node in nodes]
        ys = [layout[node][1] for node in nodes]
        labels = [node_labels[node] for node in nodes]
        base_color = color_map[kind]
        colors = [base_color for _ in nodes]
        sizes = [18 for _ in nodes]
        if kind == "partition":
            for idx, node in enumerate(nodes):
                if node in boundary_partitions:
                    colors[idx] = "#E45756"
                    sizes[idx] = 22
        elif kind == "conversion":
            for idx, node in enumerate(nodes):
                if node in boundary_conversions:
                    colors[idx] = "#E45756"
                    sizes[idx] = 22
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(size=sizes, color=colors, line=dict(width=1.2, color="#2F2F2F")),
                name=kind,
                hovertext=[str(graph.nodes[node]) for node in nodes],
                hoverinfo="text",
            )
        )

    def add_edge_trace(edges: List[Tuple[Node, Node]], *, color: str, dash: str = "solid", width: float = 1.5, name: str = "") -> None:
        if not edges:
            return
        x_values: List[float] = []
        y_values: List[float] = []
        for u, v in edges:
            x_values.extend([layout[u][0], layout[v][0], None])
            y_values.extend([layout[u][1], layout[v][1], None])
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=color, dash=dash, width=width),
                hoverinfo="none",
                name=name,
            )
        )

    dependency_edges: List[Tuple[Node, Node]] = []
    backend_edges: List[Tuple[Node, Node]] = []
    conversion_edges: List[Tuple[Node, Node]] = []
    entanglement_edges: List[Tuple[Node, Node]] = []
    long_range_edges: List[Tuple[Node, Node]] = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        if not include_edge(u, v, key, data):
            continue
        kind = data.get("kind")
        if kind == "dependency":
            dependency_edges.append((u, v))
        elif kind in {"backend_assignment", "conversion_source", "conversion_target"}:
            backend_edges.append((u, v))
        elif kind == "conversion_boundary":
            conversion_edges.append((u, v))
        elif kind == "entanglement":
            if (u, v, key) in long_range or (v, u, key) in long_range:
                long_range_edges.append((u, v))
            else:
                entanglement_edges.append((u, v))
        else:
            conversion_edges.append((u, v))

    add_edge_trace(dependency_edges, color="#9E9E9E", width=1.5, name="dependencies")
    add_edge_trace(backend_edges, color="#54A24B", width=1.5, name="backend transitions")
    add_edge_trace(conversion_edges, color="#EECA3B", dash="dash", width=1.5, name="conversions")
    add_edge_trace(entanglement_edges, color="#B279A2", width=2.0, name="entanglement")
    add_edge_trace(long_range_edges, color="#FF7F0E", dash="dot", width=3.0, name="long-range entanglement")

    fig.update_layout(
        title="SSD visualisation",
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
    )

    return fig
