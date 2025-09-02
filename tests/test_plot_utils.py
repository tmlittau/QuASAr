import matplotlib
matplotlib.use("Agg")

import pandas as pd

from benchmarks.plot_utils import compute_baseline_best, plot_quasar_vs_baseline_best


def sample_df():
    return pd.DataFrame(
        [
            {"circuit": "c1", "framework": "sv", "run_time_mean": 1.0, "total_time_mean": 2.0},
            {"circuit": "c1", "framework": "mps", "run_time_mean": 0.5, "total_time_mean": 1.5},
            {"circuit": "c1", "framework": "quasar", "backend": "mps", "run_time_mean": 0.6, "total_time_mean": 1.2},
            {"circuit": "c2", "framework": "sv", "run_time_mean": 3.0, "total_time_mean": 4.0},
            {"circuit": "c2", "framework": "quasar", "backend": "sv", "run_time_mean": 2.5, "total_time_mean": 3.5},
        ]
    )


def test_compute_baseline_best():
    df = sample_df()
    best = compute_baseline_best(df)
    assert set(best["circuit"]) == {"c1", "c2"}
    assert best.loc[best["circuit"] == "c1", "run_time_mean"].iloc[0] == 0.5
    assert best.loc[best["circuit"] == "c2", "run_time_mean"].iloc[0] == 3.0


def test_plot_quasar_vs_baseline_best_annotations():
    df = sample_df()
    ax = plot_quasar_vs_baseline_best(df, annotate_backend=True)
    texts = {t.get_text() for t in ax.texts}
    assert "mps" in texts
    assert "sv" in texts
    assert len(ax.lines) == 2
