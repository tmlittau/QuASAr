import numpy as np
import time

CHIS = [2, 4, 8, 16, 32]


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
    """SVD runtimes grow with ``chi``."""
    np.random.seed(0)
    runtimes = [svd_runtime(c) for c in CHIS]
    assert all(x < y for x, y in zip(runtimes, runtimes[1:]))
