import numpy as np
import pytest
import time

CHIS = [2, 4, 8, 16, 32]
BASELINE = {
    2: 9.99e-06,
    4: 1.30e-05,
    8: 2.00e-05,
    16: 4.45e-05,
    32: 1.69e-04,
}


def svd_runtime(chi: int, repeats: int = 20) -> float:
    """Return the minimal runtime of an SVD on a random ``chi``×``chi`` matrix."""
    m = np.random.rand(chi, chi)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.linalg.svd(m, full_matrices=False)
        times.append(time.perf_counter() - start)
    return min(times)


def test_svd_runtime_vs_chi() -> None:
    """SVD runtimes grow with ``chi`` and match expected baselines."""
    np.random.seed(0)
    runtimes = [svd_runtime(c) for c in CHIS]
    assert all(x < y for x, y in zip(runtimes, runtimes[1:]))
    for chi, rt in zip(CHIS, runtimes):
        assert rt == pytest.approx(BASELINE[chi], rel=1.0)
