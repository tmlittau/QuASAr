#!/usr/bin/env python3
"""Aggregate backend selection logs.

Each log line is expected to contain five comma-separated fields::

    sparsity, nnz, rotation, backend, score

The script summarises the backend selections for each input log and the
overall distribution.

Usage
-----
```
python tools/analyze_backend_selection.py log1.csv log2.csv ...
```
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def _parse_file(path: Path) -> Counter:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                continue
            backend = row[3].strip()
            counts[backend] += 1
    return counts


def summarize(paths: Iterable[Path]) -> None:
    overall: Counter[str] = Counter()
    for p in paths:
        counts = _parse_file(p)
        overall.update(counts)
        print(f"{p.stem}:")
        for backend, count in sorted(counts.items()):
            print(f"  {backend}: {count}")
    if len(paths) > 1:
        print("Overall:")
        for backend, count in sorted(overall.items()):
            print(f"  {backend}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate backend selection logs"
    )
    parser.add_argument("logs", nargs="+", help="Paths to log files")
    args = parser.parse_args()
    summarize(Path(p) for p in args.logs)


if __name__ == "__main__":
    main()
