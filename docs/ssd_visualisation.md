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

## Filtering problematic regions

Both rendering helpers accept :class:`HighlightOptions`.  Use the
``boundary_qubit_threshold`` field to highlight large interfaces between
partitions and conversions, and ``only_problematic`` to hide benign
regions.  Long-range entanglement (measured by the minimum distance
between qubit indices) is emphasised using the ``long_range_threshold``
field.

For larger SSDs consider serialising the graph to disk and loading it in a
notebook together with these utilities for interactive exploration.
