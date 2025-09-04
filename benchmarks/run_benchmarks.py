"""Convenience wrapper for the benchmark command line interface.

The CLI executes circuits through QuASAr's scheduler and can force
individual backends via :mod:`quasar.cost.Backend`.
"""
from benchmark_cli import main

if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
