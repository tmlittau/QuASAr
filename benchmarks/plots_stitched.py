#!/usr/bin/env python3
"""Stacked runtime and peak-memory plots for stitched QuASAr benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

COL_BACKEND = {
    "baseline": "#9e9e9e",
    "sv": "#d32f2f",
    "mps": "#1976d2",
    "dd": "#7e57c2",
    "es": "#00897b",
    "tableau": "#2e7d32",
    "conversion": "#f5a623",
    "other": "#546e7a",
    "sv_theoretical": "#bdbdbd",
}
THEORETICAL_BACKENDS = {"sv_theoretical"}
THEORETICAL_LABEL = "SV (theoretical)"
THEORETICAL_COLOR = "#bdbdbd"
THEORETICAL_EDGE = "#616161"
THEORETICAL_HATCH = "//"

BASELINE_BACKENDS = {"sv", "mps", "dd", "es", "tableau"}
BACKEND_LABELS = {
    "sv": "Statevector",
    "mps": "Matrix product state",
    "dd": "Decision diagram",
    "es": "Extended stabilizer",
    "tableau": "Tableau",
    "conversion": "Conversion",
    "other": "Other",
    "baseline": "Baseline",
    "sv_theoretical": THEORETICAL_LABEL,
}

BACKEND_ALIASES = {
    "statevector": "sv",
    "sv": "sv",
    "mps": "mps",
    "decision_diagram": "dd",
    "dd": "dd",
    "extended_stabilizer": "es",
    "extended stabilizer": "es",
    "ext": "es",
    "tableau": "tableau",
    "tab": "tableau",
    "quasar": "quasar",
    "sv_theoretical": "sv_theoretical",
    "statevector_theoretical": "sv_theoretical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help="Path to a JSON results file produced by the stitched benchmark runner.",
    )
    parser.add_argument(
        "--database",
        help="Path to the benchmarks SQLite database (alternative to --results).",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for output plots")
    parser.add_argument("--title", default="QuASAr stitched")
    parser.add_argument(
        "--show-all-baselines",
        action="store_true",
        help="Plot every available baseline backend instead of the fastest only.",
    )
    parser.add_argument("--csv", help="Optional CSV summary path")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--run-id",
        help="Restrict database loading to the given benchmark run identifier.",
    )
    parser.add_argument(
        "--circuit",
        dest="circuits",
        action="append",
        help="Limit plots to the specified circuit (repeat for multiple names).",
    )
    args = parser.parse_args()

    if bool(args.results) == bool(args.database):
        parser.error("exactly one of --results or --database must be provided")
    return args


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def label_from(record: Dict[str, Any]) -> str:
    name = record.get("name") or record.get("circuit") or "circuit"
    params = record.get("params") or {}
    n = params.get("num_qubits")
    bs = params.get("block_size")
    extras = []
    if isinstance(n, int):
        extras.append(f"n={n}")
    if isinstance(bs, int):
        extras.append(f"b={bs}")
    depth = params.get("depth")
    if isinstance(depth, int):
        extras.append(f"d={depth}")
    if isinstance(depth, (list, tuple)) and len(depth) == 3:
        extras.append(f"d={tuple(depth)}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{name}{suffix}"


def normalise_backend(name: Any) -> str:
    if not name:
        return "other"
    key = str(name).strip().lower()
    key = key.replace("-", "_")
    return BACKEND_ALIASES.get(key, key)


def is_quasar(rec: Dict[str, Any]) -> bool:
    if rec.get("is_quasar"):
        return True
    backend = str(rec.get("backend", "")).lower()
    if backend == "quasar":
        return True
    framework = str(rec.get("framework", "")).lower()
    return framework == "quasar"


def is_baseline(rec: Dict[str, Any]) -> bool:
    if rec.get("is_baseline"):
        return True
    backend = normalise_backend(rec.get("backend"))
    if backend in BASELINE_BACKENDS:
        return True
    framework = normalise_backend(rec.get("framework"))
    return framework in BASELINE_BACKENDS


def backend_key(rec: Dict[str, Any]) -> str:
    backend = normalise_backend(rec.get("backend"))
    if backend in COL_BACKEND:
        return backend
    return "other"


def choose_best_baseline(
    recs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    baselines = [
        r
        for r in recs
        if is_baseline(r)
        and isinstance(r.get("runtime"), (int, float))
        and backend_key(r) not in THEORETICAL_BACKENDS
    ]
    if not baselines:
        return None, []
    best = min(baselines, key=lambda r: r["runtime"])
    return best, baselines


def collect_quasar(recs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [r for r in recs if is_quasar(r)]
    if not candidates:
        return None
    candidates.sort(key=lambda r: 0 if r.get("segments") else 1)
    return candidates[0]


def normalize_segments(qrec: Dict[str, Any]) -> List[Tuple[str, float]]:
    segments: List[Tuple[str, float]] = []
    if not qrec:
        return segments
    raw_segments = qrec.get("segments")
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            backend = normalise_backend(seg.get("backend"))
            time_value = seg.get("time") or seg.get("runtime") or seg.get("duration")
            if time_value is None:
                continue
            try:
                time_float = float(time_value)
            except (TypeError, ValueError):
                continue
            if time_float > 0:
                segments.append((backend, time_float))
    conversions_total = qrec.get("conversions_total")
    if conversions_total:
        try:
            conv_value = float(conversions_total)
        except (TypeError, ValueError):
            conv_value = 0.0
        if conv_value > 0:
            segments.append(("conversion", conv_value))
    if not segments:
        runtime = qrec.get("runtime")
        try:
            runtime_val = float(runtime)
        except (TypeError, ValueError):
            runtime_val = 0.0
        if runtime_val > 0:
            segments = [("other", runtime_val)]
    return segments


def fmt_units(value: float) -> str:
    if not value or value <= 0:
        return "0"
    exponent = int(math.floor(math.log10(value)))
    if exponent < 6:
        return f"{value:.3g}"
    units = ["", "K", "M", "G", "T", "P", "E"]
    index = min(len(units) - 1, exponent // 3)
    scaled = value / (10 ** (3 * index))
    return f"{scaled:.3g}{units[index]}"


def load_json_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("results.json must contain a list of records")
    records: List[Dict[str, Any]] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        runtime = record.get("runtime")
        try:
            runtime_value = float(runtime) if runtime is not None else None
        except (TypeError, ValueError):
            runtime_value = None
        if runtime_value is not None:
            record = dict(record)
            record["runtime"] = runtime_value
        peak_mem = record.get("peak_mem")
        try:
            peak_value = float(peak_mem) if peak_mem is not None else None
        except (TypeError, ValueError):
            peak_value = None
        if peak_value is not None:
            record = dict(record)
            record["peak_mem"] = peak_value
        records.append(record)
    return records


def _load_database_rows(
    database: str, run_id: str | None
) -> Iterable[Dict[str, Any]]:
    connection = sqlite3.connect(database)
    try:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        resolved_run_id = run_id
        if resolved_run_id is None:
            row = cursor.execute(
                "SELECT id FROM benchmark_run ORDER BY datetime(created_at) DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("database does not contain any benchmark runs")
            resolved_run_id = row["id"]
        query = """
            SELECT
                sr.framework,
                sr.backend,
                sr.mode,
                sr.run_time_mean,
                sr.total_time_mean,
                sr.run_peak_memory_mean,
                sr.prepare_peak_memory_mean,
                sr.result_json,
                sr.extra,
                b.circuit_id,
                b.qubits AS benchmark_qubits,
                sr.qubits AS run_qubits,
                b.metadata,
                br.parameters AS run_parameters,
                br.id AS run_id
            FROM simulation_run sr
            JOIN benchmark b ON sr.benchmark_id = b.id
            JOIN benchmark_run br ON b.run_id = br.id
            WHERE b.run_id = ?
            ORDER BY b.circuit_id, COALESCE(sr.qubits, b.qubits), sr.framework
        """
        for row in cursor.execute(query, (resolved_run_id,)):
            yield dict(row)
    finally:
        connection.close()


def _extract_segments_from_extra(extra: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not extra:
        return []
    segments: List[Dict[str, Any]] = []
    raw_segments = extra.get("segments") or extra.get("segment_breakdown")
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            backend = seg.get("backend") or seg.get("name")
            time_value = seg.get("time") or seg.get("runtime") or seg.get("duration")
            if backend is None or time_value is None:
                continue
            try:
                time_float = float(time_value)
            except (TypeError, ValueError):
                continue
            segments.append({"backend": backend, "time": time_float})
    elif isinstance(raw_segments, dict):
        for backend, time_value in raw_segments.items():
            try:
                time_float = float(time_value)
            except (TypeError, ValueError):
                continue
            segments.append({"backend": backend, "time": time_float})
    return segments


def _parse_json_column(value: Any) -> Dict[str, Any] | None:
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:  # pragma: no cover - defensive parsing
        return None


def load_database_results(database: str, run_id: str | None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in _load_database_rows(database, run_id):
        framework = normalise_backend(row.get("framework"))
        backend_raw = normalise_backend(row.get("backend"))
        runtime = row.get("run_time_mean")
        if runtime is None:
            runtime = row.get("total_time_mean")
        peak_mem = row.get("run_peak_memory_mean")
        if runtime is None:
            continue
        try:
            runtime_float = float(runtime)
        except (TypeError, ValueError):
            continue
        record: Dict[str, Any] = {
            "name": row.get("circuit_id"),
            "params": {"num_qubits": row.get("run_qubits") or row.get("benchmark_qubits")},
            "runtime": runtime_float,
            "peak_mem": float(peak_mem) if peak_mem is not None else None,
            "framework": row.get("framework"),
        }
        if framework == "quasar":
            record["backend"] = "quasar"
            record["quasar_backend"] = backend_raw
        else:
            record["backend"] = backend_raw
        metadata = _parse_json_column(row.get("metadata")) or {}
        params = record.setdefault("params", {})
        if metadata:
            if "description" in metadata:
                record.setdefault("description", metadata.get("description"))
        parameters = _parse_json_column(row.get("run_parameters")) or {}
        if parameters:
            suite = parameters.get("suite")
            if suite:
                params.setdefault("suite", suite)
        params = {k: v for k, v in params.items() if v is not None}
        record["params"] = params
        extra = _parse_json_column(row.get("extra"))
        if extra:
            segments = _extract_segments_from_extra(extra)
            if segments:
                record["segments"] = segments
            conversions_total = extra.get("conversions_total") or extra.get(
                "conversion_time"
            )
            if conversions_total is not None:
                record["conversions_total"] = conversions_total
        records.append(record)
    return records


def filter_records(
    records: Sequence[Dict[str, Any]], filters: Sequence[str] | None
) -> List[Dict[str, Any]]:
    if not filters:
        return list(records)
    lowered = {f.lower() for f in filters}
    filtered: List[Dict[str, Any]] = []
    for record in records:
        candidates = []
        for key in ("name", "circuit"):
            value = record.get(key)
            if value:
                candidates.append(str(value))
        candidates.append(label_from(record))
        params = record.get("params") or {}
        base = str(params.get("circuit")) if params.get("circuit") else None
        if base:
            candidates.append(base)
        matched = False
        for candidate in candidates:
            if candidate and candidate.lower() in lowered:
                matched = True
                break
        if matched:
            filtered.append(record)
    return filtered


def summarise_and_plot(args: argparse.Namespace, records: List[Dict[str, Any]]) -> None:
    if not records:
        raise SystemExit("no records available for plotting")

    ensure_out_dir(args.out_dir)

    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        label = label_from(record)
        groups.setdefault(label, []).append(record)

    nrows = len(groups)
    fig_rt, axes_rt = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(6.7, 3.0 * nrows),
        constrained_layout=True,
    )
    if nrows == 1:
        axes_rt = [axes_rt]
    fig_rt.suptitle(args.title + " — runtime (log scale)")

    fig_mem, axes_mem = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(6.7, 3.0 * nrows),
        constrained_layout=True,
    )
    if nrows == 1:
        axes_mem = [axes_mem]
    fig_mem.suptitle(args.title + " — peak memory (log scale)")

    legend_entries: "OrderedDict[str, Any]" = OrderedDict()
    summary_rows: List[Dict[str, Any]] = []

    for (label, recs), ax_r, ax_m in zip(groups.items(), axes_rt, axes_mem):
        best_base, all_bases = choose_best_baseline(recs)
        qrec = collect_quasar(recs)

        tick_labels: List[str] = []
        tick_positions: List[int] = []
        xpos = 0

        base_list: List[Dict[str, Any]] = []
        if args.show_all_baselines:
            seen_backends: set[str] = set()
            for record in all_bases:
                key = backend_key(record)
                if key in seen_backends:
                    continue
                seen_backends.add(key)
                base_list.append(record)
        elif best_base:
            base_list = [best_base]

        theoretical_entries = [
            rec
            for rec in recs
            if backend_key(rec) in THEORETICAL_BACKENDS
            and isinstance(rec.get("runtime"), (int, float))
        ]
        for entry in theoretical_entries:
            if entry not in base_list:
                base_list.append(entry)

        for base in base_list:
            runtime = base.get("runtime")
            if runtime is None or runtime <= 0:
                continue
            key = backend_key(base)
            is_theoretical = key in THEORETICAL_BACKENDS or base.get("is_theoretical")
            if is_theoretical:
                label_text = THEORETICAL_LABEL
                bar = ax_r.bar(
                    xpos,
                    runtime,
                    color=THEORETICAL_COLOR,
                    edgecolor=THEORETICAL_EDGE,
                    linewidth=1.0,
                    hatch=THEORETICAL_HATCH,
                    label=label_text,
                )
            else:
                colour_key = key if args.show_all_baselines else "baseline"
                colour = COL_BACKEND.get(colour_key, COL_BACKEND["baseline"])
                label_text = (
                    f"{BACKEND_LABELS.get(key, key.upper())} baseline"
                    if args.show_all_baselines
                    else "Best baseline"
                )
                bar = ax_r.bar(xpos, runtime, color=colour, label=label_text)
            if label_text not in legend_entries:
                legend_entries[label_text] = bar[0]
            tick_positions.append(xpos)
            if is_theoretical:
                tick_labels.append(THEORETICAL_LABEL)
            else:
                tick_labels.append(key.upper() if args.show_all_baselines else "Best baseline")
            xpos += 1

        if qrec and qrec.get("runtime"):
            tick_positions.append(xpos)
            tick_labels.append("QuASAr")
            segments = normalize_segments(qrec)
            bottom = 0.0
            for backend, duration in segments:
                colour = COL_BACKEND.get(backend, COL_BACKEND["other"])
                legend_label = BACKEND_LABELS.get(backend, backend.title())
                bar = ax_r.bar(
                    xpos,
                    duration,
                    bottom=bottom,
                    color=colour,
                    label=legend_label,
                )
                if legend_label not in legend_entries:
                    legend_entries[legend_label] = bar[0]
                bottom += duration
            xpos += 1

        ax_r.set_title(label)
        ax_r.set_xticks(tick_positions)
        ax_r.set_xticklabels(tick_labels)
        ax_r.set_yscale("log")
        ax_r.set_ylabel("Runtime (a.u., log)")
        ax_r.grid(True, which="both", axis="y", alpha=0.25)

        mem_positions: List[int] = []
        mem_labels: List[str] = []
        mem_values: List[float] = []
        xpos_mem = 0

        for base in base_list:
            peak = base.get("peak_mem")
            if peak is None:
                peak = base.get("peak_memory")
            try:
                peak_val = float(peak) if peak is not None else float("nan")
            except (TypeError, ValueError):
                peak_val = float("nan")
            mem_positions.append(xpos_mem)
            key = backend_key(base)
            is_theoretical = key in THEORETICAL_BACKENDS or base.get("is_theoretical")
            if is_theoretical:
                mem_labels.append(THEORETICAL_LABEL)
                ax_m.bar(
                    xpos_mem,
                    peak_val,
                    color=THEORETICAL_COLOR,
                    edgecolor=THEORETICAL_EDGE,
                    linewidth=1.0,
                    hatch=THEORETICAL_HATCH,
                )
            else:
                mem_labels.append(key.upper() if args.show_all_baselines else "Best baseline")
                colour_key = key if args.show_all_baselines else "baseline"
                colour = COL_BACKEND.get(colour_key, COL_BACKEND["baseline"])
                ax_m.bar(xpos_mem, peak_val, color=colour)
            mem_values.append(peak_val)
            xpos_mem += 1

        if qrec:
            peak = qrec.get("peak_mem") or qrec.get("peak_memory")
            try:
                peak_val = float(peak) if peak is not None else float("nan")
            except (TypeError, ValueError):
                peak_val = float("nan")
            mem_positions.append(xpos_mem)
            mem_labels.append("QuASAr")
            ax_m.bar(xpos_mem, peak_val, color="#2bbbad")
            mem_values.append(peak_val)
        ax_m.set_xticks(mem_positions)
        ax_m.set_xticklabels(mem_labels)
        ax_m.set_yscale("log")
        ax_m.set_ylabel("Peak memory (a.u., log)")
        ax_m.set_title(label)
        ax_m.grid(True, which="both", axis="y", alpha=0.25)

        for pos, value in zip(mem_positions, mem_values):
            if value and value > 0:
                ax_m.text(pos, value * 1.05, fmt_units(value), ha="center", va="bottom", fontsize=9)

        if best_base and qrec and best_base.get("runtime") and qrec.get("runtime"):
            base_runtime = float(best_base["runtime"])
            quasar_runtime = float(qrec["runtime"])
            baseline_mem = best_base.get("peak_mem") or best_base.get("peak_memory")
            quasar_mem = qrec.get("peak_mem") or qrec.get("peak_memory")
            try:
                speedup = base_runtime / max(quasar_runtime, 1e-12)
            except ZeroDivisionError:
                speedup = float("nan")
            if baseline_mem is None or quasar_mem is None:
                mem_ratio = float("nan")
            else:
                try:
                    mem_ratio = float(quasar_mem) / max(float(baseline_mem), 1e-12)
                except (TypeError, ValueError, ZeroDivisionError):
                    mem_ratio = float("nan")
            summary_rows.append(
                {
                    "circuit": label,
                    "best_baseline": backend_key(best_base),
                    "baseline_runtime": base_runtime,
                    "quasar_runtime": quasar_runtime,
                    "speedup": speedup,
                    "baseline_peak_mem": baseline_mem,
                    "quasar_peak_mem": quasar_mem,
                    "mem_ratio": mem_ratio,
                }
            )

    if legend_entries:
        fig_rt.legend(
            legend_entries.values(),
            legend_entries.keys(),
            loc="upper right",
            bbox_to_anchor=(1.0, -0.02),
            frameon=True,
        )

    runtime_path = os.path.join(args.out_dir, "runtime_by_circuit.png")
    memory_path = os.path.join(args.out_dir, "memory_by_circuit.png")
    fig_rt.savefig(runtime_path, dpi=args.dpi)
    fig_mem.savefig(memory_path, dpi=args.dpi)
    plt.close(fig_rt)
    plt.close(fig_mem)
    print(f"[OK] wrote {runtime_path}")
    print(f"[OK] wrote {memory_path}")

    if args.csv and summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(args.csv, "w", encoding="utf-8", newline="") as csv_file:
            import csv

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[OK] wrote {args.csv}")


def main() -> None:
    args = parse_args()
    if args.results:
        records = load_json_results(args.results)
    else:
        records = load_database_results(args.database, args.run_id)
    records = filter_records(records, args.circuits)
    summarise_and_plot(args, records)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
