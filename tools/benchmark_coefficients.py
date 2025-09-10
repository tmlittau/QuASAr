"""Run QuASAr calibration benchmarks and store results.

This script executes the microbenchmarks defined in
:mod:`quasar.calibration` to derive runtime and memory coefficients for
all supported simulation backends and conversion primitives.  The
measured coefficients are written to the next versioned JSON file under
``calibration/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from quasar.calibration import run_calibration, save_coefficients


def _next_version_path() -> Path:
    """Return destination path for the next calibration JSON file."""

    root = Path(__file__).resolve().parents[1] / "calibration"
    root.mkdir(exist_ok=True)
    existing = []
    for file in root.glob("coeff_v*.json"):
        try:
            existing.append(int(file.stem.split("v")[1]))
        except ValueError:
            continue
    version = max(existing, default=0) + 1
    return root / f"coeff_v{version}.json"


def main() -> None:
    """Run benchmarks and persist the resulting coefficients."""

    path = _next_version_path()
    coeff: Dict[str, float] = run_calibration()
    save_coefficients(path, coeff)
    print(f"Saved calibration data to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
