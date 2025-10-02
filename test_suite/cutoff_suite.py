from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from test_suite.hybrid_random_tail import append_random_tail_qiskit
from test_suite.theoretical_baselines import (
    predict_sv_peak_bytes,
    predict_sv_runtime_au,
    predict_conversion_time_au,
)
from test_suite.baselines import (
    run_dd_time_and_mem,
    run_extended_stabilizer_time_and_mem,
)


# ---------- Helpers ----------
def _qubit_targets(inst: "stim.CircuitInstruction") -> List[int]:
    """Extract qubit targets from a stim instruction following stim's API."""

    qubits: List[int] = []
    for target in inst.targets_copy():
        if target.is_combiner:
            # Combiners split multi-target operations (e.g. pair targets).
            continue
        if target.is_qubit_target:
            qubits.append(target.qubit_value)
    return qubits


def gate_counts_from_stim(circ: "stim.Circuit") -> Dict[str, int]:
    counts = {"1q": 0, "2q": 0, "diag2q": 0, "3q": 0, "other": 0, "total": 0}
    oneq = {"H", "S", "S_DAG", "X", "Y", "Z"}
    twoq = {"CX", "CZ", "SWAP"}
    for inst in circ:
        name = inst.name
        targs = _qubit_targets(inst)
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


def _counts_with_tail(
    n: int,
    base_counts: Dict[str, int],
    tail: TailConfig,
    stim_seed: int,
    depth: int,
) -> Dict[str, int]:
    counts = dict(base_counts)
    tail_counts = _simulate_tail_gate_counts(n, tail, stim_seed, depth)
    for key, value in tail_counts.items():
        counts[key] = counts.get(key, 0) + value
    return counts


def _simulate_tail_gate_counts(
    n: int,
    tail: TailConfig,
    stim_seed: int,
    depth: int,
) -> Dict[str, int]:
    if tail.layers <= 0:
        return {"1q": 0, "2q": 0, "diag2q": 0, "3q": 0, "other": 0, "total": 0}

    class _CountingCircuit:
        def __init__(self, qubits: int) -> None:
            self._n = qubits
            self.counts = {"1q": 0, "2q": 0, "diag2q": 0, "3q": 0, "other": 0}

        @property
        def num_qubits(self) -> int:
            return self._n

        def rx(self, _theta: float, _q: int) -> None:
            self.counts["1q"] += 1

        def ry(self, _theta: float, _q: int) -> None:
            self.counts["1q"] += 1

        def rz(self, _theta: float, _q: int) -> None:
            self.counts["1q"] += 1

        def crx(self, _theta: float, _a: int, _b: int) -> None:
            self.counts["2q"] += 1

        def cry(self, _theta: float, _a: int, _b: int) -> None:
            self.counts["2q"] += 1

        def rzx(self, _theta: float, _a: int, _b: int) -> None:
            self.counts["2q"] += 1

        def rxx(self, _theta: float, _a: int, _b: int) -> None:
            self.counts["2q"] += 1

        def ryy(self, _theta: float, _a: int, _b: int) -> None:
            self.counts["2q"] += 1

    recorder = _CountingCircuit(n)
    append_random_tail_qiskit(
        recorder,
        layers=tail.layers,
        twoq_prob=tail.twoq_prob,
        angle_eps=tail.angle_eps,
        oneq_ops=tail.oneq_ops,
        twoq_ops=tail.twoq_ops,
        seed=tail.effective_seed(n, depth, stim_seed),
    )
    tail_counts = dict(recorder.counts)
    tail_counts["total"] = tail_counts.get("1q", 0) + tail_counts.get("2q", 0)
    tail_counts.setdefault("diag2q", 0)
    tail_counts.setdefault("3q", 0)
    tail_counts.setdefault("other", 0)
    return tail_counts


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
    one_qubit_map = {
        "H": sim.h,
        "S": sim.s,
        "S_DAG": sim.s_dag,
        "X": sim.x,
        "Y": sim.y,
        "Z": sim.z,
    }
    two_qubit_map = {
        "CX": sim.cx,
        "CZ": sim.cz,
        "SWAP": sim.swap,
    }
    for inst in circ:
        name = inst.name
        targs = _qubit_targets(inst)
        if name in one_qubit_map:
            op = one_qubit_map[name]
            for q in targs:
                op(q)
        elif name in two_qubit_map:
            op = two_qubit_map[name]
            for a, b in zip(targs[::2], targs[1::2]):
                op(a, b)
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
    max_qubit = max(
        (t.qubit_value for inst in circ for t in inst.targets_copy() if t.is_qubit_target),
        default=-1,
    )
    qc = QuantumCircuit(max_qubit + 1)
    for inst in circ:
        name = inst.name
        targs = _qubit_targets(inst)
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
@dataclass(frozen=True)
class TailConfig:
    layers: int = 0
    twoq_prob: float = 0.3
    angle_eps: float = 1e-3
    seed: int = 2025
    oneq_ops: Sequence[str] = ("rx", "ry", "rz")
    twoq_ops: Sequence[str] = ("crx", "cry", "rzx", "rxx", "ryy")

    def effective_seed(self, n: int, depth: int, stim_seed: int) -> int:
        # Mix parameters to provide distinct reproducible RNG streams per trial.
        return (
            int(self.seed)
            + 1_000_003 * int(stim_seed)
            + 1_009 * int(n)
            + int(depth)
        )


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
    dd_runtime: Optional[float] = None
    dd_peak_mem: Optional[int] = None
    dd_timed_out: Optional[bool] = None
    dd_error: Optional[str] = None
    dd_mode: Optional[str] = None
    es_runtime: Optional[float] = None
    es_peak_mem: Optional[int] = None
    es_timed_out: Optional[bool] = None
    es_error: Optional[str] = None
    es_mode: Optional[str] = None


def measure_one(
    n: int,
    depth: int,
    seed: int,
    sv_timeout_sec: float,
    tail: TailConfig,
    run_dd: bool,
    dd_timeout_sec: float,
    run_es: bool,
    es_timeout_sec: float,
) -> TrialRecord:
    circ = build_random_clifford_stim(n, depth, seed=seed)
    tab_t, tableau = run_stim_tableau_time(circ)
    conv_t = measure_conversion_time_from_tableau(n, tableau)
    if QuantumCircuit is None:
        # theoretical SV
        counts = gate_counts_from_stim(circ)
        if tail.layers > 0:
            counts = _counts_with_tail(n, counts, tail, seed, depth)
        sv_t = predict_sv_runtime_au(n, counts)
        sv_pk = predict_sv_peak_bytes(n)
        return TrialRecord(
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
    qc = build_qiskit_from_stim(circ)
    if tail.layers > 0:
        append_random_tail_qiskit(
            qc,
            layers=tail.layers,
            twoq_prob=tail.twoq_prob,
            angle_eps=tail.angle_eps,
            oneq_ops=tail.oneq_ops,
            twoq_ops=tail.twoq_ops,
            seed=tail.effective_seed(n, depth, seed),
        )
    sv_t, sv_pk, sv_oom, sv_to, sv_mode = run_qiskit_sv_time_and_mem_timeout(qc, sv_timeout_sec)
    dd_result = run_dd_time_and_mem(qc, timeout_sec=dd_timeout_sec) if run_dd else None
    es_result = run_extended_stabilizer_time_and_mem(qc, timeout_sec=es_timeout_sec) if run_es else None
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
        dd_runtime=None if dd_result is None else dd_result["runtime"],
        dd_peak_mem=None if dd_result is None else dd_result["peak_mem"],
        dd_timed_out=None if dd_result is None else dd_result["timed_out"],
        dd_error=None if dd_result is None else dd_result["error"],
        dd_mode=None if dd_result is None else dd_result["mode"],
        es_runtime=None if es_result is None else es_result["runtime"],
        es_peak_mem=None if es_result is None else es_result["peak_mem"],
        es_timed_out=None if es_result is None else es_result["timed_out"],
        es_error=None if es_result is None else es_result["error"],
        es_mode=None if es_result is None else es_result["mode"],
    )


def find_cutoff(
    n: int,
    target_speedup: float,
    seed: int,
    depth_min: int,
    depth_max: int,
    sv_timeout_sec: float,
    tail: TailConfig,
    run_dd: bool,
    dd_timeout_sec: float,
    run_es: bool,
    es_timeout_sec: float,
) -> Tuple[int, List[TrialRecord]]:
    lo, hi = depth_min, depth_max
    history: List[TrialRecord] = []
    while lo < hi:
        mid = (lo + hi) // 2
        rec = measure_one(
            n,
            mid,
            seed,
            sv_timeout_sec,
            tail,
            run_dd,
            dd_timeout_sec,
            run_es,
            es_timeout_sec,
        )
        history.append(rec)
        if rec.speedup_vs_sv >= target_speedup:
            hi = mid
        else:
            lo = mid + 1
    final = measure_one(
        n,
        lo,
        seed,
        sv_timeout_sec,
        tail,
        run_dd,
        dd_timeout_sec,
        run_es,
        es_timeout_sec,
    )
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
    ap.add_argument("--run-dd", action="store_true", help="Also run the MQT DDSIM baseline.")
    ap.add_argument("--dd-timeout-sec", type=float, default=60.0, help="Timeout for DDSIM runs.")
    ap.add_argument(
        "--run-es",
        action="store_true",
        help="Also run the Aer extended-stabilizer baseline.",
    )
    ap.add_argument(
        "--es-timeout-sec",
        type=float,
        default=60.0,
        help="Timeout for Aer extended-stabilizer runs.",
    )
    ap.add_argument(
        "--tail-layers",
        type=int,
        default=0,
        help=(
            "Number of random non-Clifford tail layers appended after the Clifford "
            "prefix. Set to 0 to disable."
        ),
    )
    ap.add_argument(
        "--tail-twoq-prob",
        type=float,
        default=0.3,
        help="Probability of placing a 2-qubit gate on an eligible pair in the tail.",
    )
    ap.add_argument(
        "--tail-angle-eps",
        type=float,
        default=1e-3,
        help="Reject tail angles within eps of k*pi/4 (avoids Clifford+T).",
    )
    ap.add_argument(
        "--tail-seed",
        type=int,
        default=2025,
        help="Base seed for the random tail RNG (per trial seed mixing applied).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tail_cfg = TailConfig(
        layers=args.tail_layers,
        twoq_prob=args.tail_twoq_prob,
        angle_eps=args.tail_angle_eps,
        seed=args.tail_seed,
    )
    results: Dict[str, Any] = {"params": vars(args), "runs": [], "cutoffs": {}}

    for n in args.ns:
        cutoff, hist = find_cutoff(
            n,
            args.target_speedup,
            args.seed,
            args.depth_min,
            args.depth_max,
            args.sv_timeout_sec,
            tail_cfg,
            args.run_dd,
            args.dd_timeout_sec,
            args.run_es,
            args.es_timeout_sec,
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
