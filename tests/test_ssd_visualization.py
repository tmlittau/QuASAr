"""Tests for networkx visualisation helpers on SSD objects."""

from __future__ import annotations

import pytest

from quasar.circuit import Circuit
from quasar.cost import Backend, Cost
from quasar.ssd import ConversionLayer, SSD, SSDPartition


def test_to_networkx_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempting to visualise without networkx should raise a helpful error."""

    partition = SSDPartition(subsystems=((0,),))
    ssd = SSD([partition])

    def _missing_package(name: str):  # pragma: no cover - behaviour verified in test
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("quasar.ssd.importlib.import_module", _missing_package)

    with pytest.raises(RuntimeError) as excinfo:
        ssd.to_networkx()

    assert "networkx" in str(excinfo.value)


def test_ssd_to_networkx_structure() -> None:
    """The generated graph should expose partitions, backends and conversions."""

    nx = pytest.importorskip("networkx")

    partitions = [
        SSDPartition(subsystems=((0,),), backend=Backend.STATEVECTOR),
        SSDPartition(subsystems=((1,),), backend=Backend.MPS),
    ]
    conversion = ConversionLayer(
        boundary=(0, 1),
        source=Backend.STATEVECTOR,
        target=Backend.MPS,
        rank=4,
        frontier=2,
        primitive="B2B",
        cost=Cost(time=1.0, memory=2.0),
    )
    ssd = SSD(partitions, conversions=[conversion])

    graph = ssd.to_networkx()

    assert isinstance(graph, nx.MultiDiGraph)

    part0 = ("partition", 0)
    part1 = ("partition", 1)
    conv_node = ("conversion", 0)
    backend_sv = ("backend", Backend.STATEVECTOR.name)
    backend_mps = ("backend", Backend.MPS.name)

    assert part0 in graph.nodes
    assert graph.nodes[part0]["backend"] == Backend.STATEVECTOR.name
    assert conv_node in graph.nodes
    assert graph.nodes[conv_node]["source"] == Backend.STATEVECTOR.name

    # Partition 0 should depend on partition 1 via conversion metadata.
    edge_data = graph.get_edge_data(part0, part1)
    assert edge_data is not None
    assert any(data.get("kind") == "dependency" for data in edge_data.values())
    assert any(data.get("kind") == "entanglement" for data in edge_data.values())

    # Backend assignment and conversion edges should be present.
    assert graph.has_edge(part0, backend_sv)
    assert graph.has_edge(part1, backend_mps)
    assert graph.has_edge(part0, conv_node)
    assert graph.has_edge(backend_sv, conv_node)
    assert graph.has_edge(conv_node, backend_mps)


def test_circuit_to_networkx_ssd_proxy() -> None:
    """Circuit helper should delegate to the underlying SSD instance."""

    nx = pytest.importorskip("networkx")

    circuit = Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
        use_classical_simplification=False,
    )

    graph = circuit.to_networkx_ssd()

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.graph["fingerprint"] == circuit.ssd.fingerprint
    assert ("partition", 0) in graph.nodes
