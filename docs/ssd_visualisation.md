# SSD visualisation helpers

The utilities in [`tools/ssd_visualisation.py`](../tools/ssd_visualisation.py)
turn the graph returned by :meth:`SSD.to_networkx` into publication-ready
figures.  They position partitions, conversions and backends on
dedicated rows and provide highlight options for long-range entanglement
and large conversion boundaries.

## Rendering with Matplotlib

```python
import matplotlib.pyplot as plt
from quasar.circuit import Circuit
from quasar.ssd import SSD
from tools.ssd_visualisation import HighlightOptions, compute_layout, draw_ssd_matplotlib

circuit = Circuit(
    [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 3]},
        {"gate": "CX", "qubits": [1, 2]},
        {"gate": "SWAP", "qubits": [2, 5]},
    ],
    use_classical_simplification=False,
)
ssd: SSD = circuit.ssd

graph = ssd.to_networkx(include_backends=True)
layout = compute_layout(graph)
options = HighlightOptions(long_range_threshold=2, boundary_qubit_threshold=2)

draw_ssd_matplotlib(graph, layout=layout, highlight=options)
plt.show()
```

The example above highlights the entanglement edge between partitions that
operate on distant qubits as well as partitions with a wide boundary.

## Interactive Plotly figure

```python
from tools.ssd_visualisation import HighlightOptions, draw_ssd_plotly

fig = draw_ssd_plotly(graph, layout=layout, highlight=HighlightOptions(
    long_range_threshold=2,
    boundary_qubit_threshold=2,
))
fig.show()
```

The Plotly backend retains node metadata as hover tooltips and can be
restricted to problematic regions by setting
``HighlightOptions(only_problematic=True)``.

## Example script

An executable demonstration is provided in
[`docs/examples/ssd_visualisation_example.py`](examples/ssd_visualisation_example.py).
Running the script will open a Matplotlib window highlighting the
long-range entanglement between distant qubits.

## Stitched benchmark workflow

Stitched benchmark circuits bundle several heterogeneous stages and
produce substantially larger SSDs.  The helper script in
[`docs/examples/ssd_visualisation_stitched.py`](examples/ssd_visualisation_stitched.py)
illustrates the recommended workflow:

1. Resolve the showcase definition with
   ``stitched_suite.resolve_suite("stitched-big")`` and pick the desired
   entry and width (e.g. ``spec.factory(spec.widths[0])``).
2. Convert the resulting :class:`~quasar.circuit.Circuit` to a graph via
   :meth:`Circuit.to_networkx_ssd`, keeping conversions and backend nodes
   so that the stitched transitions remain visible.
3. Derive a layout using :func:`compute_layout` and render either with
   :func:`draw_ssd_matplotlib` for static figures or
   :func:`draw_ssd_plotly` for interactive exploration.

The stitched example guards all optional dependencies: ``networkx`` is
required for the visualisation helpers, ``matplotlib`` powers the static
plot and ``plotly`` enables the interactive view.  Install any missing
packages before running the script.

Large stitched graphs can overwhelm dense views.  Start with
``HighlightOptions(only_problematic=True)`` and a generous
``boundary_qubit_threshold`` to focus on bottlenecks, or increase the
``partition_gap`` in :func:`compute_layout` to spread out the layout.
When the figure still becomes cluttered, consider toggling
``include_entanglement=False`` on :meth:`Circuit.to_networkx_ssd` to hide
less critical edges and rely on the Plotly backend for targeted zooming.

## Minimal partition-only layout

For a compact example that focuses purely on partition structure, the
[`docs/examples/ssd_visualisation_minimal.py`](examples/ssd_visualisation_minimal.py)
script constructs a three-gate circuit consisting of two disjoint
single-qubit fragments (`H` on qubit 0 and `RX` on qubit 2) and a single
entangling `CX` acting on qubits 0 and 1.  Invoking
``Circuit.ssd.to_networkx(include_conversions=False, include_backends=False)``
removes conversion layers and backend nodes so the resulting SSD clearly
shows how partitions connect.

![Minimal SSD partition layout](data/ssd_visualisation_minimal.svg)

In the rendered graph, partition ``P0`` represents the `H` gate on qubit 0
and feeds into partition ``P2`` that contains the `CX` gate on qubits 0 and
1, encoding the execution dependency introduced by the entangling
operation.  Partition ``P1`` captures the `RX` rotation on qubit 2 and
remains isolated because it manipulates a qubit that does not interact
with the other fragments.

Recreate the diagram with::

    PYTHONPATH=. python docs/examples/ssd_visualisation_minimal.py --output docs/data/ssd_visualisation_minimal.svg

The command emits the SVG used above, letting readers modify the circuit
or plotting parameters to explore other partitioning outcomes.


## Filtering problematic regions

Both rendering helpers accept :class:`HighlightOptions`.  Use the
``boundary_qubit_threshold`` field to highlight large interfaces between
partitions and conversions, and ``only_problematic`` to hide benign
regions.  Long-range entanglement (measured by the minimum distance
between qubit indices) is emphasised using the ``long_range_threshold``
field.

For larger SSDs consider serialising the graph to disk and loading it in a
notebook together with these utilities for interactive exploration.
