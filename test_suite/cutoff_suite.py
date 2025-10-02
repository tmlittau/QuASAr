from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

try:
    import stim
except Exception:
    stim = None

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
except Exception:
    QuantumCircuit = None
    Aer = None

from test_suite.theoretical_baselines import (
    predict_sv_peak_bytes,
    predict_sv_runtime_au,
    predict_conversion_time_au,
)


# ---------- Helpers ----------
def gate_counts_from_stim(circ: "stim.Circuit") -> Dict[str, int]:
    counts = {"1q": 0, "2q": 0, "diag2q": 0, "3q": 0, "other": 0, "total": 0}
    oneq = {"H", "S", "S_DAG", "X", "Y", "Z"}
    twoq = {"CX", "CZ", "SWAP"}
    for inst in circ:
        name = inst.name
        targs = inst.targets_copy()
        if name in oneq:
            counts["1q"] += len(targs)
            counts["total"] += len(targs)
        elif name in twoq:
            pairs = len(targs) // 2
            counts["2q"] += pairs
            counts["total"] += pairs
            if name == "CZ":
                counts["diag2q"] += pairs
        else:
            counts["other"] += 1
            counts["total"] += 1
    return counts


def build_random_clifford_stim(n: int, depth: int, seed: int = 1337) -> "stim.Circuit":
    if stim is None:
        raise RuntimeError("stim not installed. pip install stim")
    import random
    rnd = random.Random(seed)
    c = stim.Circuit()
    for layer in range(depth):
        # 1q layer
        for q in range(n):
            r = rnd.random()
            if r < 0.33:
                c.append_operation("H", [q])
            elif r < 0.66:
                c.append_operation("S", [q])
            else:
                c.append_operation("S_DAG", [q])
        # 2q layers (even/odd pairing)
        for q in range(0, n - 1, 2):
            c.append_operation("CX" if rnd.random() < 0.5 else "CZ", [q, q + 1])
        for q in range(1, n - 1, 2):
            c.append_operation("CX" if rnd.random() < 0.5 else "CZ", [q, q + 1])
    return c


def run_stim_tableau_time(circ: "stim.Circuit") -> Tuple[float, Any]:
    t0 = time.perf_counter()
    sim = stim.TableauSimulator()
    for inst in circ:
        name = inst.name
        targs = [int(t) for t in inst.targets_copy()]
        if name in ("H", "S", "S_DAG", "X", "Y", "Z"):
            for q in targs:
                sim.do(name, q)
        elif name in ("CX", "CZ", "SWAP"):
            for a, b in zip(targs[::2], targs[1::2]):
                sim.do(name, a, b)
        else:
            pass
    t1 = time.perf_counter()
    tableau = sim.canonical_stabilizers()
    return (t1 - t0), tableau


def measure_conversion_time_from_tableau(n: int, tableau: Any, repeats: int = 3) -> float:
    # Replace with your real converter if available; otherwise keep theoretical model.
    # Measured stub (fast), plus predicted value to ensure scaling with n:
    dummy = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = n * n
        t1 = time.perf_counter()
        dummy.append(t1 - t0)
    predicted = predict_conversion_time_au(n)
    return max(sum(dummy) / len(dummy), predicted)


def build_qiskit_from_stim(circ: "stim.Circuit") -> "QuantumCircuit":
    if QuantumCircuit is None:
        raise RuntimeError("qiskit-aer not installed. pip install qiskit qiskit-aer")
    qc = QuantumCircuit(max(int(t) for inst in circ for t in inst.targets_copy()) + 1)
    for inst in circ:
        name = inst.name
        targs = [int(t) for t in inst.targets_copy()]
        if name == "H":
            for q in targs:
                qc.h(q)
        elif name == "S":
            for q in targs:
                qc.s(q)
        elif name == "S_DAG":
            for q in targs:
                qc.sdg(q)
        elif name == "X":
            for q in targs:
                qc.x(q)
        elif name == "Y":
            for q in targs:
                qc.y(q)
        elif name == "Z":
            for q in targs:
                qc.z(q)
        elif name == "CX":
            for a, b in zip(targs[::2], targs[1::2]):
                qc.cx(a, b)
        elif name == "CZ":
            for a, b in zip(targs[::2], targs[1::2]):
                qc.cz(a, b)
        elif name == "SWAP":
            for a, b in zip(targs[::2], targs[1::2]):
                qc.swap(a, b)
    return qc


def run_qiskit_sv_time_and_mem_timeout(
    qc: "QuantumCircuit", sv_timeout_sec: float
) -> Tuple[float, int, bool, bool, str]:
    """
    Return: (runtime, peak_bytes, sv_oom_pred, sv_timed_out, mode_note)
    - Uses Aer job.result(timeout=sv_timeout_sec) to enforce timeout.
    - Peak bytes is predicted from n (sufficient for comparisons).
    """
    n = qc.num_qubits
    peak = predict_sv_peak_bytes(n)
    if Aer is None:
        counts = estimate_counts_from_qc(qc)
        rt = predict_sv_runtime_au(n, counts)
        return rt, peak, False, True, "theoretical (no Aer)"
    backend = Aer.get_backend("aer_simulator_statevector")
    t0 = time.perf_counter()
    job = backend.run(qc, shots=0)
    try:
        # If it doesn’t finish within timeout, this raises an Exception
        _ = job.result(timeout=sv_timeout_sec)
        t1 = time.perf_counter()
        return (t1 - t0), peak, False, False, "measured"
    except Exception as e:
        try:
            job.cancel()
        except Exception:
            pass
        # fallback theoretical
        counts = estimate_counts_from_qc(qc)
        rt = predict_sv_runtime_au(n, counts)
        return rt, peak, False, True, "theoretical (timeout)"


def estimate_counts_from_qc(qc: "QuantumCircuit") -> Dict[str, int]:
    counts = {"1q": 0, "2q": 0, "diag2q": 0, "3q": 0, "other": 0}
    for instr, qargs, _ in qc.data:
        name = instr.name.lower()
        qn = len(qargs)
        if qn == 1:
            counts["1q"] += 1
        elif qn == 2:
            counts["2q"] += 1
            if name in ("cz", "cp", "crz"):
                counts["diag2q"] += 1
        elif qn == 3:
            counts["3q"] += 1
        else:
            counts["other"] += 1
    return counts


# ---------- Cutoff search ----------
@dataclass
class TrialRecord:
    n: int
    depth: int
    tableau_time: float
    convert_time: float
    sv_time: float
    sv_peak_bytes: int
    sv_oom: bool
    sv_timed_out: bool
    sv_mode: str
    speedup_vs_sv: float


def measure_one(n: int, depth: int, seed: int, sv_timeout_sec: float) -> TrialRecord:
    circ = build_random_clifford_stim(n, depth, seed=seed)
    tab_t, tableau = run_stim_tableau_time(circ)
    conv_t = measure_conversion_time_from_tableau(n, tableau)
    if QuantumCircuit is None:
        # theoretical SV
        counts = gate_counts_from_stim(circ)
        sv_t = predict_sv_runtime_au(n, counts)
        sv_pk = predict_sv_peak_bytes(n)
        rec = TrialRecord(
            n,
            depth,
            tab_t,
            conv_t,
            sv_t,
            sv_pk,
            False,
            True,
            "theoretical (no Aer)",
            sv_t / max(tab_t + conv_t, 1e-12),
        )
        return rec
    qc = build_qiskit_from_stim(circ)
    sv_t, sv_pk, sv_oom, sv_to, sv_mode = run_qiskit_sv_time_and_mem_timeout(qc, sv_timeout_sec)
    return TrialRecord(
        n=n,
        depth=depth,
        tableau_time=tab_t,
        convert_time=conv_t,
        sv_time=sv_t,
        sv_peak_bytes=sv_pk,
        sv_oom=sv_oom,
        sv_timed_out=sv_to,
        sv_mode=sv_mode,
        speedup_vs_sv=sv_t / max(tab_t + conv_t, 1e-12),
    )


def find_cutoff(
    n: int,
    target_speedup: float,
    seed: int,
    depth_min: int,
    depth_max: int,
    sv_timeout_sec: float,
) -> Tuple[int, List[TrialRecord]]:
    lo, hi = depth_min, depth_max
    history: List[TrialRecord] = []
    while lo < hi:
        mid = (lo + hi) // 2
        rec = measure_one(n, mid, seed, sv_timeout_sec)
        history.append(rec)
        if rec.speedup_vs_sv >= target_speedup:
            hi = mid
        else:
            lo = mid + 1
    final = measure_one(n, lo, seed, sv_timeout_sec)
    history.append(final)
    return lo, history


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[20, 22, 24, 26])
    ap.add_argument("--depth-min", type=int, default=100)
    ap.add_argument("--depth-max", type=int, default=100000)
    ap.add_argument("--target-speedup", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--sv-timeout-sec",
        type=float,
        default=60.0,
        help="Wall-clock timeout for Aer SV. On timeout, fall back to theoretical.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results: Dict[str, Any] = {"params": vars(args), "runs": [], "cutoffs": {}}

    for n in args.ns:
        cutoff, hist = find_cutoff(
            n,
            args.target_speedup,
            args.seed,
            args.depth_min,
            args.depth_max,
            args.sv_timeout_sec,
        )
        print(
            f"[n={n}] cutoff≈{cutoff} (target {args.target_speedup}x). "
            f"Any SV timeouts among trials? {any(h.sv_timed_out for h in hist)}"
        )
        results["cutoffs"][str(n)] = cutoff
        results["runs"].extend(asdict(h) for h in hist)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] wrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
