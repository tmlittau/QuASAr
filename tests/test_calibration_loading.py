import json
from pathlib import Path

from quasar import CostEstimator


def test_cost_estimator_loads_latest_calibration():
    calib_dir = Path(__file__).resolve().parents[1] / "calibration"
    path = calib_dir / "coeff_v99.json"
    try:
        path.write_text(json.dumps({"sv_gate_1q": 2.0}))
        est = CostEstimator()
        assert est.coeff["sv_gate_1q"] == 2.0
    finally:
        if path.exists():
            path.unlink()
