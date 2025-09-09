"""Convenience wrapper for the benchmark command line interface.

The CLI executes circuits through QuASAr's scheduler and can force
individual backends via :mod:`quasar.cost.Backend`. Circuit families that
consist solely of Clifford gates are skipped automatically by the CLI to
avoid measuring trivial stabiliser workloads.
"""

from benchmark_cli import main

# Import surface-code protected circuits so the CLI can discover them.
from circuits import surface_corrected_qaoa_circuit  # noqa: F401


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

