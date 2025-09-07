# Issue: Tune cost model coefficients

**Location:** `quasar/cost.py` lines 89-121

**Context:** The cost model initializes baseline coefficients with a comment "Baseline coefficients; tuned empirically in a full system." These defaults are placeholders for calibrated values.

**Impact:** Does not impede system operation but may yield inaccurate cost estimates.

**Task:** Develop and document a calibration procedure for cost coefficients across supported backends and update `quasar/cost.py` with validated values.
