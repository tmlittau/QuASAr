import csv
from pathlib import Path

import pytest


def load_speedups():
    path = Path(__file__).resolve().parents[1] / 'benchmarks' / 'quick_analysis_results.csv'
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        quick = float(row['quick_time'])
        full = float(row['full_time'])
        expected = float(row['speedup'])
        yield full / quick, expected


def test_relative_speedup_bar_chart():
    for computed, expected in load_speedups():
        assert computed == pytest.approx(expected)
        assert computed > 1.0
