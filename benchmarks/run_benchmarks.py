"""Convenience wrapper for the benchmark command line interface.

The CLI executes circuits through QuASAr's scheduler and can force
individual backends via :mod:`quasar.cost.Backend`. Circuit families that
consist solely of Clifford gates are skipped automatically by the CLI to
avoid measuring trivial stabiliser workloads.
"""

from benchmark_cli import main


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

