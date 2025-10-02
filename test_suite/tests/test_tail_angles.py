import random

from test_suite.hybrid_random_tail import _dist_to_pi_over_4, sample_nonclifford_angle


def test_angles_avoid_pi_over_4():
    rng = random.Random(42)
    eps = 1e-3
    for _ in range(1000):
        theta = sample_nonclifford_angle(rng, eps=eps)
        assert _dist_to_pi_over_4(theta) > eps
