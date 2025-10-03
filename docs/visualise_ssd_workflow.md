# Visualising SSDs with `benchmarks/visualise_ssd.py`

The `benchmarks/visualise_ssd.py` helper renders two subsystem descriptor
(SSD) figures in a single invocation: one generated from a configurable
showcase benchmark and one constructed from a compact, hand-crafted example.
Both rely on the visualisation utilities in
[`tools/ssd_visualisation.py`](../tools/ssd_visualisation.py) and require
`matplotlib` and `networkx` to be installed.  Install `plotly` if you want to
inspect the graphs interactively after saving them to disk.

The script can be launched either as a module (`python -m
benchmarks.visualise_ssd`) or as a regular file (`python
benchmarks/visualise_ssd.py`).  The command-line options described below
control which circuits are simulated, how wide the figures are spaced and
where the rendered images are stored.【F:benchmarks/visualise_ssd.py†L1-L135】【F:benchmarks/visualise_ssd.py†L188-L253】

## 1. Rendering an SSD for a benchmark circuit

1. **Choose the benchmark.** List the available showcase names from the
   ``SHOWCASE_CIRCUITS`` registry and pick one (for example
   `layered_clifford_midpoint`).  The script resolves the matching
   specification and builds the circuit at its narrowest advertised width
   unless you override it with ``--width``.【F:benchmarks/visualise_ssd.py†L35-L84】【F:benchmarks/visualise_ssd.py†L137-L172】
2. **Run the simulation and render the layout.** Execute:

   ```bash
   PYTHONPATH=. python benchmarks/visualise_ssd.py \
       --benchmark layered_clifford_midpoint \
       --benchmark-output docs/data/layered_clifford_midpoint_ssd.svg
   ```

   Replace the ``--benchmark`` value with your chosen circuit and adjust the
   ``--benchmark-output`` path and format as desired.  The script simulates the
   circuit with `SimulationEngine`, converts the resulting SSD to a graph with
   backend and conversion nodes, computes a layout and saves the Matplotlib
   figure.  The command prints a confirmation once the file has been written.
   Use ``--show`` to open the figure in a window after saving it.【F:benchmarks/visualise_ssd.py†L86-L191】【F:benchmarks/visualise_ssd.py†L198-L247】
3. **Tweak the highlights if required.** Increase or decrease
   ``--long-range-threshold`` to highlight long-distance entanglement edges and
   ``--boundary-threshold`` to emphasise interfaces with many boundary qubits in
   the benchmark figure.  Adjust ``--benchmark-partition-gap`` when partitions
   appear crowded in the rendered layout.【F:benchmarks/visualise_ssd.py†L55-L115】【F:benchmarks/visualise_ssd.py†L213-L227】

## 2. Rendering and customising the schematic SSD example

The second figure produced by the script is a synthetic SSD assembled in
``_build_schematic_ssd``.  It illustrates how QuASAr's planner combines
single-qubit and entangling partitions and annotates them with backend
assignments and coarse cost estimates.  To tailor the example to your own
partitioning constraints or cost assumptions:

1. **Inspect the predefined partitions.** Each ``SSDPartition`` in
   ``_build_schematic_ssd`` specifies the qubits it covers, the gate history,
   the backend choice and an approximate ``Cost`` tuple.  Dependencies and
   ``entangled_with`` links control the edges in the visualisation.  Start by
   reviewing these defaults to understand how the schematic layout is derived.【F:benchmarks/visualise_ssd.py†L117-L187】
2. **Adjust the cost or backend assignments.** Modify the ``Cost`` values or
   switch the ``backend`` enum to reflect the configuration you want to
   communicate.  Keep dependencies consistent so that downstream partitions
   still point to the appropriate inputs; the SSD metadata is rebuilt with
   ``schematic.build_metadata()`` to validate the structure after your edits.【F:benchmarks/visualise_ssd.py†L156-L187】
3. **Render the updated schematic.** Run the helper again, this time pointing
   the ``--example-output`` flag to a target file:

   ```bash
   PYTHONPATH=. python benchmarks/visualise_ssd.py \
       --example-output docs/data/schematic_ssd.svg
   ```

   Combine the command with the benchmark options if you wish to regenerate
   both figures in one go.  The schematic rendering omits conversions and
   backend nodes by default so that the partition structure remains uncluttered;
   adjust ``--example-partition-gap`` if you change the number of partitions and
   need extra horizontal spacing.【F:benchmarks/visualise_ssd.py†L188-L247】
4. **Verify the partitioning outcome.** Open the saved figure to ensure the
   dependencies still produce the desired graph.  When further tweaking is
   required, iterate on the partition definitions and rerun the command until
   the layout communicates the intended constraints and costs.

With these steps you can reproduce the showcase SSDs used throughout the
QuASAr documentation and craft bespoke examples that highlight particular
partitioning scenarios.
