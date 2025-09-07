# Cleanup Checklist

This checklist tracks occurrences of baseline placeholders or heuristics that may require follow-up.

## Requires Follow-up
- [ ] **Cost model coefficients** – `quasar/cost.py` uses empirically tuned baseline coefficients that may not reflect current backend performance. See [issues/cost_model_coefficients.md](issues/cost_model_coefficients.md).
- [ ] **Quick-path thresholds** – default limits for bypassing planning may need validation. See [issues/quick_path_thresholds.md](issues/quick_path_thresholds.md).
- [ ] **Quick-path memory usage** – compare peak memory between quick and full planning to tune thresholds. See [issues/quick_path_memory_usage.md](issues/quick_path_memory_usage.md).

## No Action Needed
- References to "baseline" backends in benchmarking code and notebooks are intended for comparison and do not affect full-system operation.
- No occurrences of "placeholder" were found in the repository.
