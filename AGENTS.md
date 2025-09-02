# AGENTS

## Project Overview
- **QuASAr (Quantum Adaptive Simulation Architecture)** provides a declarative interface for quantum circuit simulation.
- It analyses circuits, partitions them and chooses simulation methods per partition using a cost model and a compact state descriptor (SSD).
- Components: `CircuitAnalyzer`, `Planner`, `Scheduler`, and a C++ `ConversionEngine` for state extraction and conversion.
- See `Littau_QuASAr.pdf` for the full paper describing architecture, optimization strategy and benchmarks.

## Development Guidelines
- Code is primarily Python 3.8+ with optional C++17 extensions.
- Follow existing style: PEP8 formatting, descriptive docstrings and type annotations.
- Document new user‑facing behaviour in `docs/` or relevant notebooks.

## Testing
- Run the full test suite with:
  ```bash
  pytest
  ```
- Add tests for new features or bug fixes.

## Benchmarking Notes
- When benchmarking, disable quick‑path heuristics (`QUASAR_QUICK_MAX_QUBITS`, `QUASAR_QUICK_MAX_GATES`, `QUASAR_QUICK_MAX_DEPTH`) or set them via `Planner`/`Scheduler` to `None` for fair comparisons.
- Explicitly pass a `Backend` when comparing against other simulators so both run on the same method, and record the selected backend in results.
- Benchmarks and example notebooks live under `benchmarks/`; raw results are stored in `benchmarks/results/`.

## Paper Insights
- The paper highlights:
  - A declarative API that separates method selection from circuit specification.
  - Two‑level optimisation combining method‑based partitioning with parallelism heuristics.
  - Conversion primitives (boundary extraction, local windows, stabilizer learning) to switch between backends.
  - Benchmark metrics: runtime, peak memory, backend switches, conversion time, plan cache behaviour and state fidelity.
- Align contributions and experiments with these goals and metrics.

