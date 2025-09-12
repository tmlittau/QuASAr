# Large Circuit Benchmarks

This document outlines composite circuits designed to stress QuASAr's
partitioning engine. Each circuit combines independent subroutines followed by
operations that entangle previously separate qubit groups.

## Dual GHZ with Global QFT

Two disjoint registers of equal width are prepared in GHZ states. A global
quantum Fourier transform then ties both registers together. The initial
separable structure allows QuASAr to identify independent fragments before the
QFT introduces all-to-all connectivity across the combined register.

## Adder–GHZ–QAOA Hybrid

A ripple-carry adder acts on two ``n``-bit operands while an ``n``-qubit GHZ
state is generated on a separate register. Once the arithmetic and GHZ
preparations finish, all qubits undergo several rounds of QAOA on a ring
graph. The mix of classical reversible logic, entangled state preparation and
variational layers highlights QuASAr's ability to switch between simulation
methods.
