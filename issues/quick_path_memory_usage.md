# Issue: Balance quick-path memory usage

**Location:** `quasar/scheduler.py` lines 36-38 and `quasar/planner.py` lines 440-442

**Context:** Quick-path heuristics prioritize faster planning but their impact on peak memory is unclear. Benchmarks should compare memory consumption between quick-path and full planning.

**Impact:** Does not block system operation, yet suboptimal thresholds may increase memory usage on certain circuits.

**Task:** Measure peak memory for representative circuits with and without quick-path heuristics and adjust `quick_max_*` defaults to favor memory savings while maintaining acceptable runtime.
