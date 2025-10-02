from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.results) as f:
        data = json.load(f)
    runs = data["runs"]
    cutoffs = {int(k): int(v) for k, v in data.get("cutoffs", {}).items()}
    by_n = defaultdict(list)
    for r in runs:
        by_n[r["n"]].append(r)
    for n in by_n:
        by_n[n].sort(key=lambda x: x["depth"])

    for n, rows in by_n.items():
        depths = [r["depth"] for r in rows]
        tabc = [r["tableau_time"] + r["convert_time"] for r in rows]
        sv = [r["sv_time"] for r in rows]
        timed = [r["sv_timed_out"] for r in rows]

        # Runtime plot
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(depths, tabc, label="Tableau + conversion", linewidth=2)
        ax.plot(depths, sv, label="Statevector (SV)", linewidth=2)
        # Mark timeouts
        for d, s, to in zip(depths, sv, timed):
            if to:
                ax.scatter([d], [s], marker="x", color="tab:red", s=50, label=None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Depth (log)")
        ax.set_ylabel("Runtime (log, s or a.u.)")
        title = f"Runtime vs depth — n={n}"
        if n in cutoffs:
            ax.axvline(cutoffs[n], color="tab:orange", ls="--", label=f"cutoff≈{cutoffs[n]}")
            title += f"  (cutoff≈{cutoffs[n]})"
        ax.set_title(title)
        ax.legend()
        fig.savefig(os.path.join(args.out_dir, f"runtime_n{n}.png"), dpi=args.dpi)

        # Speedup plot
        speed = [sv_i / max(tc_i, 1e-12) for sv_i, tc_i in zip(sv, tabc)]
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax2.plot(depths, speed, label="SV / (Tableau + conv)")
        for y in (2.0, 3.0, 4.0):
            ax2.axhline(y, color="gray", ls="--", lw=1, label=None)
        if n in cutoffs:
            ax2.axvline(cutoffs[n], color="tab:orange", ls="--", label=f"cutoff≈{cutoffs[n]}")
        ax2.set_xscale("log")
        ax2.set_xlabel("Depth (log)")
        ax2.set_ylabel("Speedup (×)")
        ax2.set_title(f"Speedup vs depth — n={n}")
        ax2.legend()
        fig2.savefig(os.path.join(args.out_dir, f"speedup_n{n}.png"), dpi=args.dpi)

    print(f"[OK] wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
