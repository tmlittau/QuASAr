import numpy as np

CHIS = [2, 4, 8, 16, 32]


def test_svd_reconstruction() -> None:
    """SVD decomposition should perfectly reconstruct the original matrix."""
    np.random.seed(0)
    for chi in CHIS:
        m = np.random.rand(chi, chi)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        reconstructed = (u * s) @ vh
        assert np.allclose(m, reconstructed)
