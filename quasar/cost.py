from __future__ import annotations

"""Cost estimation for different quantum simulation backends."""

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Cost:
    """Simple container for time, memory and complexity estimates."""

    time: float
    memory: float
    log_depth: float = 0.0


class Backend(Enum):
    """Simulation backends supported by the estimator."""

    STATEVECTOR = "sv"
    TABLEAU = "tab"
    MPS = "mps"
    DECISION_DIAGRAM = "dd"


@dataclass
class ConversionEstimate:
    """Result of estimating a conversion between backends."""

    primitive: str
    cost: Cost


class CostEstimator:
    """Estimate runtime and memory for simulation and conversion.

    Simulation estimates follow the QuASAr draft: statevector uses
    ``2**n`` amplitudes, tableau methods require ``O(n^2)`` work, MPS
    depends on bond dimension ``chi`` and decision diagrams are linear in
    frontier size ``r``.  Conversion costs are modelled after the
    primitives described in Table~2 of the draft: boundary-to-boundary
    (B2B), local window (LW), staged (ST) and full extraction.  Each
    primitive is expressed as a simple polynomial in the SSD parameters
    (boundary size ``q``, rank ``s``, frontier ``r`` and optional window
    ``w``).  Constants can be tuned using the ``coeff`` dictionary.
    """

    def __init__(
        self,
        coeff: Optional[Dict[str, float]] = None,
        *,
        s_max: Optional[int] = None,
        r_max: Optional[int] = None,
        q_max: Optional[int] = None,
    ):
        # Baseline coefficients; tuned empirically in a full system.
        self.coeff: Dict[str, float] = {
            "sv_gate_1q": 1.0,
            "sv_gate_2q": 1.0,
            "sv_meas": 1.0,
            "sv_mem": 1.0,
            "tab_gate": 1.0,
            "tab_mem": 1.0,
            "mps_gate_1q": 1.0,
            "mps_gate_2q": 1.0,
            "mps_trunc": 1.0,
            "mps_mem": 1.0,
            "dd_gate": 1.0,
            "dd_mem": 1.0,
            # Conversion primitives
            "b2b_svd": 1.0,
            "b2b_copy": 1.0,
            "lw_extract": 1.0,
            "st_stage": 1.0,
            "full_extract": 1.0,
            # Ingestion cost per target backend
            "ingest_sv": 1.0,
            "ingest_tab": 1.0,
            "ingest_mps": 1.0,
            "ingest_dd": 1.0,
            # Fixed overhead applied to every backend switch
            "conversion_base": 0.0,
        }
        if coeff:
            self.coeff.update(coeff)
        # Policy caps for planner heuristics
        self.s_max = s_max
        self.r_max = r_max
        self.q_max = q_max

    # ------------------------------------------------------------------
    def update_coefficients(self, updates: Dict[str, float]) -> None:
        """Update calibration coefficients in-place."""
        self.coeff.update(updates)

    def to_file(self, path: str | Path) -> None:
        """Persist coefficients to a JSON file."""
        with Path(path).open("w") as fh:
            json.dump(self.coeff, fh, indent=2, sort_keys=True)

    @classmethod
    def from_file(cls, path: str | Path) -> "CostEstimator":
        """Construct an estimator with coefficients loaded from ``path``."""
        with Path(path).open() as fh:
            coeff = json.load(fh)
        return cls(coeff=coeff)

    def statevector(
        self,
        num_qubits: int,
        num_1q_gates: int,
        num_2q_gates: int,
        num_meas: int,
    ) -> Cost:
        amp = 2 ** num_qubits
        gate_time = (
            self.coeff["sv_gate_1q"] * num_1q_gates
            + self.coeff["sv_gate_2q"] * num_2q_gates
            + self.coeff["sv_meas"] * num_meas
        )
        time = gate_time * amp
        memory = self.coeff["sv_mem"] * amp
        depth = math.log2(num_qubits) if num_qubits > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    def tableau(self, num_qubits: int, num_gates: int) -> Cost:
        quad = num_qubits ** 2
        time = self.coeff["tab_gate"] * num_gates * quad
        memory = self.coeff["tab_mem"] * quad
        depth = math.log2(num_qubits) if num_qubits > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    def mps(
        self,
        num_qubits: int,
        num_1q_gates: int,
        num_2q_gates: int,
        chi: int,
        *,
        svd: bool = False,
    ) -> Cost:
        r"""Estimate cost for matrix product state simulation.

        Parameters
        ----------
        num_qubits:
            Number of qubits in the simulated register.
        num_1q_gates, num_2q_gates:
            Counts of single- and two-qubit gates respectively. Measurement
            operations should be included in ``num_1q_gates``.
        chi:
            Assumed bond dimension of the MPS.
        svd:
            If ``True``, include an additional cost for the singular value
            decomposition and truncation step performed after entangling gates.

        Notes
        -----
        Single-qubit gates scale with :math:`\chi^2` while two-qubit gates scale
        with :math:`\chi^3`.  The optional truncation step adds a term scaling
        as :math:`\chi^3 \log \chi` per two-qubit gate.
        """

        n = num_qubits
        chi2 = chi ** 2
        chi3 = chi ** 3
        time = (
            self.coeff["mps_gate_1q"] * num_1q_gates * n * chi2
            + self.coeff["mps_gate_2q"] * num_2q_gates * n * chi3
        )
        if svd and chi > 1 and num_2q_gates > 0:
            time += (
                self.coeff["mps_trunc"]
                * num_2q_gates
                * n
                * chi3
                * math.log2(chi)
            )
        memory = self.coeff["mps_mem"] * n * chi2
        depth = math.log2(num_qubits) if num_qubits > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    def decision_diagram(self, num_gates: int, frontier: int) -> Cost:
        time = self.coeff["dd_gate"] * num_gates * frontier
        memory = self.coeff["dd_mem"] * frontier
        depth = math.log2(frontier) if frontier > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    # Conversion cost estimation -------------------------------------

    def conversion(
        self,
        source: Backend,
        target: Backend,
        num_qubits: int,
        rank: int,
        frontier: int,
        window: Optional[int] = None,
        *,
        s_max: Optional[int] = None,
        r_max: Optional[int] = None,
        q_max: Optional[int] = None,
    ) -> ConversionEstimate:
        """Estimate cost to convert between representations.

        Parameters
        ----------
        source, target:
            Backends involved in the conversion.
        num_qubits:
            Size of the boundary set ``q``.
        rank:
            Upper bound on Schmidt rank ``s`` across the cut.
        frontier:
            Decision diagram frontier size ``r``.
        window:
            Optional dense extraction window ``w`` for the LW primitive.
        Notes
        -----
        A fixed ``conversion_base`` overhead is applied to every backend switch
        and ingestion costs are scaled with the full register size.
        """

        s_cap = s_max if s_max is not None else self.s_max
        r_cap = r_max if r_max is not None else self.r_max
        q_cap = q_max if q_max is not None else self.q_max

        if (
            (q_cap is not None and num_qubits > q_cap)
            or (s_cap is not None and rank > s_cap)
            or (r_cap is not None and frontier > r_cap)
        ):
            return ConversionEstimate("Full", Cost(float("inf"), float("inf")))

        full = 2 ** num_qubits
        ingest_time = self.coeff[f"ingest_{target.value}"] * full
        base_time = self.coeff.get("conversion_base", 0.0)
        overhead = ingest_time + base_time

        # --- B2B primitive ---
        b2b_time = (
            self.coeff["b2b_svd"] * (rank ** 3)
            + self.coeff["b2b_copy"] * num_qubits * (rank ** 2)
            + overhead
        )
        b2b_mem = max(num_qubits * rank ** 2, full)

        # --- LW primitive ---
        w = window if window is not None else min(num_qubits, 4)
        dense = 2 ** w
        lw_time = self.coeff["lw_extract"] * dense + overhead
        lw_mem = max(dense, full)

        # --- ST primitive ---
        chi_tilde = min(rank, 16)
        st_time = self.coeff["st_stage"] * (chi_tilde ** 3) + overhead
        st_mem = max(num_qubits * (chi_tilde ** 2), full)

        # --- Full extraction primitive ---
        full_time = self.coeff["full_extract"] * full + overhead
        full_mem = full

        candidates = {
            "B2B": (b2b_time, b2b_mem),
            "LW": (lw_time, lw_mem),
            "ST": (st_time, st_mem),
            "Full": (full_time, full_mem),
        }

        primitive, (time, memory) = min(candidates.items(), key=lambda kv: kv[1][0])
        depth = math.log2(rank) if rank > 0 else 0.0
        return ConversionEstimate(
            primitive=primitive, cost=Cost(time=time, memory=memory, log_depth=depth)
        )


__all__ = [
    "Backend",
    "Cost",
    "ConversionEstimate",
    "CostEstimator",
]

