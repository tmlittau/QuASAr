# Sparsity heuristic

The `Circuit` class exposes a `sparsity` attribute estimating how many zero
amplitudes remain after running the circuit on `|0…0>`.  It returns a value
between 0 and 1: `0` denotes a fully dense state with all `2**n` amplitudes
non-zero, while `1` indicates maximal sparsity where only a single basis state
is populated.

The estimator performs a single pass over the circuit, tracking an
approximate count `nnz` of non-zero amplitudes and computing `1 - nnz / 2**n`,
where `n` is the number of qubits.

## Branching-gate heuristic

Only a small set of single-qubit gates are treated as *branching* operations:
`H`, `RX`, `RY`, `U`, `U2`, and `U3`.  An uncontrolled branching gate doubles
`nnz`.  A controlled version adds just one extra amplitude, reflecting that the
new branch appears only when the control qubit is `|1⟩`.  The count is clamped
to `2**n`, yielding an overall complexity of `O(num_gates)`.

## Examples

*W-state:* an `n`-qubit W-state contains exactly `n` non-zero amplitudes, so
the heuristic reports a sparsity of `1 - n / 2**n`, approaching 1 as `n`
grows.

*QFT:* the quantum Fourier transform applies repeated Hadamards and
controlled-phase rotations that drive `nnz` to `2**n`, producing a sparsity of
`0`—a maximally dense state.

The planner combines the sparsity score with an estimate of the number of
non-zero amplitudes to form part of the decision-diagram metric.  Weights
``dd_sparsity_weight`` and ``dd_nnz_weight``—together with
``dd_phase_rotation_weight``, ``dd_amplitude_rotation_weight`` and
``dd_metric_threshold``—determine when the decision-diagram backend is
considered.  These values may be overridden via
the ``QUASAR_DD_*`` environment variables.

