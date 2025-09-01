# Stabilizer Tensor Network Prototype

This document sketches the data structures and algorithms required to support
stabilizer tensor networks (STNs) within QuASAr's C++ code base.

## Contraction rules

* STN tensors represent stabilizer fragments and are contracted using
  log-depth Clifford circuits.
* Bridges between fragments only fuse when both sides represent stabilizer
  states.
* Contraction proceeds by repeatedly merging compatible stabilizer nodes while
  respecting tensor indices and preserving overall parity.

## Providing STN tensors

`conversion_engine.hpp` exposes `convert_boundary_to_statevector`, which
produces a phase-factorable stabilizer state for the boundary qubits.  The
helper `convert_boundary_to_stn` wraps this dense vector together with an
optional stabilizer tableau, yielding an explicit `StnTensor` that downstream
contraction routines can consume directly.

## Detecting stabilizer-only bridges

`learn_stabilizer` attempts to recognise simple stabilizer states from raw
amplitudes.  A bridge tensor is STN-compatible if this routine succeeds on the
state produced by `convert_boundary_to_statevector`.  In that case the resulting
Tableau is attached to the `StnTensor` emitted by `convert_boundary_to_stn`
instead of materialising a dense tensor.

## Core data structures

* **`StnTensor`** – holds a stabilizer tableau alongside index labels.
* **`StnEdge`** – lightweight reference connecting two tensor indices.
* **`StnNetwork`** – container managing tensors and performing log-depth
  contractions.

These structures enable a future C++ prototype to interoperate with existing
conversion paths while capturing STN-specific complexity metrics.
