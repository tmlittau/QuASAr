import numpy as np
import time

CHIS = [2, 4, 8, 16, 32]


def svd_runtime(chi: int, repeats: int = 20) -> float:
    """Return a representative SVD runtime for a random ``chi``×``chi`` matrix.

    The first run is discarded to avoid including one-time library initialisation
    costs. The median of the remaining runs is used as a stable estimate that is
    resilient to sporadic outliers.
    """

    m = np.random.rand(chi, chi)
    times = []
    # Run one extra time and drop the first measurement to skip warm-up
    for _ in range(repeats + 1):
        start = time.perf_counter()
        np.linalg.svd(m, full_matrices=False)
        times.append(time.perf_counter() - start)
    # Discard the first run and take the median of the rest to reduce noise
    return float(np.median(times[1:]))


def test_svd_runtime_vs_chi() -> None:
    """SVD runtimes grow with ``chi``."""
    np.random.seed(0)
    runtimes = [svd_runtime(c) for c in CHIS]
    assert all(x < y for x, y in zip(runtimes, runtimes[1:]))
