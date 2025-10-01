# Issue: Review quick-path thresholds

**Location:** `quasar/scheduler.py` and configuration defaults.

**Context:** Quick-path heuristics determine when planning is bypassed. The repository includes `benchmarks/bench_utils/quick_analysis_benchmark.py` to guide tuning, but defaults may drift from optimal values.

**Impact:** Does not block system operation but may reduce performance if thresholds are poorly tuned.

**Task:** Run `benchmarks/bench_utils/quick_analysis_benchmark.py` on representative circuits to validate or update `quick_max_qubits`, `quick_max_gates`, and `quick_max_depth` defaults. Document the chosen values and update configuration guidance if the heuristics are enabled (they default to `None`).
