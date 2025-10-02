"""Plot baseline runtimes versus the QuASAr stabilizer prefix.

This helper consumes the JSON output produced by
``test_suite/cutoff_suite.py`` and renders bar charts comparing the
QuASAr tableau + conversion prefix against baseline simulators.  It can
also optionally write a compact CSV summary for the selected rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="darkgrid")


COL = {
    "quasar_prefix": "#2bbbad",  # Tableau+conversion
    "sv_meas": "#d32f2f",
    "sv_theory": "#bdbdbd",
    "dd": "#7e57c2",
    "es": "#00897b",
}


def fmt_time(value: float) -> str:
    """Format a runtime using a compact SI-like suffix."""

    if value is None or value <= 0:
        return "0"

    units = "", "k", "M", "G"
    unit_idx = 0
    while value >= 1000 and unit_idx < len(units) - 1:
        value /= 1000
        unit_idx += 1
    return f"{value:.3g}{units[unit_idx]}"


def _nearest_row(rows: List[Dict[str, Any]], target_depth: int) -> Dict[str, Any]:
    """Return the row whose depth is closest to ``target_depth``."""

    return min(rows, key=lambda row: abs(int(row["depth"]) - int(target_depth)))


def _select_rows(
    data: Dict[str, Any], depth_mode: str, depth_value: int | None
) -> List[Tuple[int, Dict[str, Any], int]]:
    """Select a single row per ``n`` based on the requested depth mode."""

    runs = data["runs"]
    cutoffs = {int(key): int(value) for key, value in data.get("cutoffs", {}).items()}
    by_n: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for run in runs:
        by_n[int(run["n"])].append(run)

    for rows in by_n.values():
        rows.sort(key=lambda row: row["depth"])

    selections: List[Tuple[int, Dict[str, Any], int]] = []
    for n, rows in by_n.items():
        if depth_mode == "cutoff" and n in cutoffs:
            target = cutoffs[n]
            row = _nearest_row(rows, target)
            selections.append((n, row, target))
        elif depth_mode == "max":
            row = rows[-1]
            selections.append((n, row, int(row["depth"])))
        elif depth_mode == "value" and depth_value is not None:
            row = _nearest_row(rows, depth_value)
            selections.append((n, row, depth_value))
        else:
            # Fallback to the shallowest run when no better match exists.
            row = rows[0]
            selections.append((n, row, int(row["depth"])))

    selections.sort(key=lambda entry: entry[0])
    return selections


def _bar_for_row(
    ax: plt.Axes,
    n: int,
    row: Dict[str, Any],
    picked_depth: int,
    show_mem: bool,
    annotate: bool,
) -> None:
    labels: List[str] = []
    values: List[float | None] = []
    colors: List[str] = []
    hatches: List[str | None] = []
    notes: List[str | None] = []

    # QuASAr prefix (Tableau + conversion)
    tableau_time = row.get("tableau_time") or 0.0
    convert_time = row.get("convert_time") or 0.0
    prefix_time = tableau_time + convert_time
    labels.append("QuASAr prefix")
    values.append(prefix_time)
    colors.append(COL["quasar_prefix"])
    hatches.append(None)
    notes.append(None)

    # SV (measured/theoretical)
    sv_time = row.get("sv_time")
    sv_mode = (row.get("sv_mode") or "").lower()
    if sv_time is not None:
        labels.append("SV" if "measured" in sv_mode else "SV (theory)")
        values.append(sv_time)
        if "measured" in sv_mode:
            colors.append(COL["sv_meas"])
            hatches.append(None)
        else:
            colors.append(COL["sv_theory"])
            hatches.append("//")
        notes.append("timeout" if row.get("sv_timed_out") else None)

    # DD baseline
    if "dd_runtime" in row and row.get("dd_runtime") is not None:
        labels.append("DD")
        values.append(row.get("dd_runtime"))
        colors.append(COL["dd"])
        hatches.append("xx" if row.get("dd_timed_out") else None)
        notes.append(
            "timeout"
            if row.get("dd_timed_out")
            else (row.get("dd_error") and "error")
        )

    # ES baseline
    if "es_runtime" in row and (
        row.get("es_runtime") is not None or row.get("es_mode") == "unsupported"
    ):
        es_mode = (row.get("es_mode") or "").lower()
        labels.append("ES")
        colors.append(COL["es"])
        if es_mode == "unsupported":
            values.append(float("nan"))
            hatches.append("..")
            notes.append("unsupported")
        else:
            values.append(row.get("es_runtime"))
            hatches.append("xx" if row.get("es_timed_out") else None)
            notes.append(
                "timeout"
                if row.get("es_timed_out")
                else (row.get("es_error") and "error")
            )

    xs = list(range(len(labels)))
    for idx, (x_pos, value, color, hatch) in enumerate(
        zip(xs, values, colors, hatches)
    ):
        if value is None or (isinstance(value, float) and (math.isinf(value) or math.isnan(value))):
            # draw a placeholder to keep legend alignment consistent
            ax.bar(
                x_pos,
                1e-12,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.4,
            )
            ax.text(x_pos, 3e-12, "n/a", ha="center", va="bottom", fontsize=9)
            effective_value = 1e-12
        else:
            ax.bar(
                x_pos,
                value,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.6,
            )
            effective_value = value
            if annotate:
                ax.text(
                    x_pos,
                    value * 1.05,
                    fmt_time(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        note = notes[idx]
        if note:
            ax.text(
                x_pos,
                effective_value * 1.2,
                note,
                ha="center",
                va="bottom",
                fontsize=9,
                color="tab:red",
            )

    subtitle = f"n={n}, depth≈{picked_depth}"
    if show_mem:
        mem = row.get("sv_peak_memory") or row.get("peak_memory")
        if mem:
            subtitle += f"\npeak mem≈{fmt_time(mem)}"
    ax.set_title(subtitle)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (log scale, s or a.u.)")
    ax.grid(True, which="both", axis="y", alpha=0.25)


def _write_csv(
    path: str, rows: List[Tuple[int, Dict[str, Any], int]]
) -> None:
    import csv

    columns = [
        "n",
        "depth",
        "quasar_prefix_time",
        "sv_time",
        "sv_mode",
        "sv_timed_out",
        "dd_runtime",
        "dd_mode",
        "dd_timed_out",
        "es_runtime",
        "es_mode",
        "es_timed_out",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for n, row, picked_depth in rows:
            writer.writerow(
                {
                    "n": n,
                    "depth": picked_depth,
                    "quasar_prefix_time": (row.get("tableau_time") or 0.0)
                    + (row.get("convert_time") or 0.0),
                    "sv_time": row.get("sv_time"),
                    "sv_mode": row.get("sv_mode"),
                    "sv_timed_out": row.get("sv_timed_out"),
                    "dd_runtime": row.get("dd_runtime"),
                    "dd_mode": row.get("dd_mode"),
                    "dd_timed_out": row.get("dd_timed_out"),
                    "es_runtime": row.get("es_runtime"),
                    "es_mode": row.get("es_mode"),
                    "es_timed_out": row.get("es_timed_out"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to cutoff results.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--depth-mode",
        choices=["cutoff", "max", "value"],
        default="cutoff",
        help="Which depth to pick per n.",
    )
    parser.add_argument("--depth", type=int, help="Used when --depth-mode=value")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="If set, draw a single grid figure; otherwise one PNG per n.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--csv", help="Optional CSV summary output")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate bar tops with numeric values.",
    )
    parser.add_argument(
        "--show-mem",
        action="store_true",
        help="Include peak memory if present (subtitle only).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = _select_rows(data, args.depth_mode, args.depth)

    if args.grid:
        num_plots = len(rows)
        ncols = 2 if num_plots >= 2 else 1
        nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7.5 * ncols, 4.2 * nrows),
            constrained_layout=True,
        )
        if isinstance(axes, plt.Axes):
            axes_list = [axes]
        else:
            axes_list = list(axes.flatten())
        for idx, (n, row, picked_depth) in enumerate(rows):
            ax = axes_list[idx]
            _bar_for_row(ax, n, row, picked_depth, args.show_mem, args.annotate)
        for idx in range(len(rows), len(axes_list)):
            fig.delaxes(axes_list[idx])
        fig.suptitle("Baselines vs QuASAr prefix at selected depths")
        out_path = os.path.join(args.out_dir, "baselines_grid.png")
        fig.savefig(out_path, dpi=args.dpi)
        print(f"[OK] wrote {out_path}")
    else:
        for n, row, picked_depth in rows:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
            _bar_for_row(ax, n, row, picked_depth, args.show_mem, args.annotate)
            out_path = os.path.join(
                args.out_dir, f"baselines_n{n}_d{picked_depth}.png"
            )
            fig.savefig(out_path, dpi=args.dpi)
            print(f"[OK] wrote {out_path}")

    if args.csv:
        _write_csv(args.csv, rows)
        print(f"[OK] wrote {args.csv}")


if __name__ == "__main__":
    main()
