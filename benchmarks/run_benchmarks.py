"""Convenience wrapper for the benchmark command line interface.

The CLI instantiates simulators via :class:`benchmarks.backends.BackendAdapter`
which delegates to the actual QuASAr backend implementations.
"""
from benchmark_cli import main

if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
