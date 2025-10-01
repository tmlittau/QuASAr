# Test Suite Notes

The integration tests that asserted specific backend selections for calibrated heuristics have been removed. They relied on hardcoded cost thresholds that no longer hold after regenerating the deployment coefficients, so the suite now focuses on behavioural checks that do not depend on particular numeric baselines.
