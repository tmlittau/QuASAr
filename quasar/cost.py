from __future__ import annotations

"""Cost estimation for different quantum simulation backends.

The estimator exposes a ``sv_bytes_per_amp`` calibration coefficient to
approximate the bytes required per statevector amplitude, including
intermediate buffers.  The actual memory footprint is modelled as
``2**num_qubits * bytes_per_amplitude * sv_bytes_per_amp`` where the
``bytes_per_amplitude`` term depends on the chosen precision
(``complex64`` or ``complex128``).
"""

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .circuit import Gate


def _circuit_depth(gates: Iterable["Gate"]) -> int:
    """Return a lightweight depth estimate for ``gates``."""

    qubit_levels: Dict[int, int] = {}
    depth = 0
    for gate in gates:
        start = max((qubit_levels.get(q, 0) for q in gate.qubits), default=0)
        level = start + 1
        for q in gate.qubits:
            qubit_levels[q] = level
        depth = max(depth, level)
    return depth


@dataclass
class Cost:
    """Simple container for runtime and memory measurements."""

    time: float
    memory: float
    log_depth: float = 0.0
    conversion: float = 0.0
    replay: float = 0.0


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
        chi_max: Optional[int] = None,
    ):
        # Baseline coefficients; tuned empirically in a full system.
        self.coeff: Dict[str, float] = {
            "sv_gate_1q": 1.0,
            "sv_gate_2q": 1.0,
            "sv_meas": 1.0,
            "sv_bytes_per_amp": 2.0,
            "tab_gate": 1.0,
            "tab_mem": 1.0,
            "mps_gate_1q": 1.0,
            "mps_gate_2q": 1.0,
            "mps_trunc": 1.0,
            "mps_mem": 1.0,
            "mps_temp_mem": 1.0,
            "dd_gate": 1.0,
            "dd_mem": 1.0,
            # Conversion primitives
            "b2b_svd": 1.0,
            "b2b_copy": 1.0,
            "lw_extract": 1.0,
            "st_stage": 1.0,
            "full_extract": 1.0,
            "st_chi_cap": 16.0,
            # Ingestion cost per target backend
            "ingest_sv": 2.0,
            "ingest_tab": 2.0,
            "ingest_mps": 2.0,
            "ingest_dd": 2.0,
            # Fixed overhead applied to every backend switch
            "conversion_base": 5.0,
        }
        if coeff:
            self.coeff.update(coeff)
        # Policy caps for planner heuristics
        self.s_max = s_max
        self.r_max = r_max
        self.q_max = q_max
        self.chi_max = chi_max

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

    # ------------------------------------------------------------------
    # Entanglement heuristics
    # ------------------------------------------------------------------
    def bond_dimensions(self, num_qubits: int, gates: Iterable["Gate"]) -> list[int]:
        """Track bond dimensions across a linear qubit ordering.

        Each two-qubit gate acting on qubits ``a`` and ``b`` doubles the bond
        dimension of all cuts between ``a`` and ``b``.  The returned list has
        ``num_qubits - 1`` entries corresponding to the Schmidt rank for each
        cut along the chain.
        """

        bonds = [1] * max(0, num_qubits - 1)
        limit = 2**num_qubits
        for gate in gates:
            qubits = getattr(gate, "qubits", [])
            if len(qubits) < 2:
                continue
            q0, q1 = min(qubits), max(qubits)
            for i in range(q0, q1):
                bonds[i] = min(bonds[i] * 2, limit)
        return bonds

    def max_schmidt_rank(self, num_qubits: int, gates: Iterable["Gate"]) -> int:
        """Return an upper bound on the maximal Schmidt rank."""

        bonds = self.bond_dimensions(num_qubits, gates)
        return max(bonds, default=1)

    def entanglement_entropy(self, num_qubits: int, gates: Iterable["Gate"]) -> float:
        """Estimate maximal bipartite entanglement entropy.

        This simply computes ``log2`` of :meth:`max_schmidt_rank`.
        """

        chi = self.max_schmidt_rank(num_qubits, gates)
        return math.log2(chi) if chi > 0 else 0.0

    def chi_for_fidelity(
        self, num_qubits: int, gates: Iterable["Gate"], fidelity: float
    ) -> int:
        """Estimate bond dimension needed to achieve ``fidelity``.

        The heuristic derives the maximal Schmidt rank across the circuit and
        scales it by the target fidelity and the circuit depth.  Lower desired
        fidelities therefore permit smaller bond dimensions.
        """

        gates = list(gates)
        chi = self.max_schmidt_rank(num_qubits, gates)
        if fidelity >= 1.0 or chi <= 1:
            return chi
        depth = _circuit_depth(gates)
        scale = fidelity ** max(depth, 1)
        return max(1, int(chi * scale))

    def chi_from_memory(self, num_qubits: int, max_memory: float) -> int:
        """Return the largest bond dimension fitting in ``max_memory``.

        Parameters
        ----------
        num_qubits:
            Number of qubits in the simulated register.
        max_memory:
            Memory budget available to the MPS tensors.

        Returns
        -------
        int
            Maximum admissible bond dimension.  ``0`` indicates that even
            ``χ=1`` would exceed ``max_memory``.
        """

        coeff = self.coeff["mps_mem"] * num_qubits
        if max_memory <= 0 or coeff <= 0:
            return 0
        chi = int(math.sqrt(max_memory / coeff))
        return chi if chi >= 1 else 0

    def chi_for_constraints(
        self,
        num_qubits: int,
        gates: Iterable["Gate"],
        fidelity: float,
        max_memory: float | None = None,
    ) -> int:
        """Estimate MPS bond dimension under fidelity and memory constraints."""

        chi = self.chi_for_fidelity(num_qubits, gates, fidelity)
        if max_memory is not None:
            chi_mem = self.chi_from_memory(num_qubits, max_memory)
            if chi_mem < chi:
                return 0
        return chi if chi >= 1 else 0

    def statevector(
        self,
        num_qubits: int,
        num_1q_gates: int,
        num_2q_gates: int,
        num_meas: int,
        *,
        precision: str = "complex128",
    ) -> Cost:
        """Estimate cost for dense statevector simulation.

        Parameters
        ----------
        num_qubits:
            Number of qubits in the simulated register.
        num_1q_gates, num_2q_gates, num_meas:
            Counts of single-qubit, two-qubit and measurement operations.
        precision:
            Complex dtype for the amplitudes.  Supported values are
            ``"complex64"`` and ``"complex128"``.
        """

        amp = 2**num_qubits
        gate_time = (
            self.coeff["sv_gate_1q"] * num_1q_gates
            + self.coeff["sv_gate_2q"] * num_2q_gates
            + self.coeff["sv_meas"] * num_meas
        )
        time = gate_time * amp
        if precision == "complex64":
            bytes_per_amp = 8
        elif precision == "complex128":
            bytes_per_amp = 16
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"unsupported precision: {precision}")
        memory = self.coeff["sv_bytes_per_amp"] * amp * bytes_per_amp
        depth = math.log2(num_qubits) if num_qubits > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    def tableau(self, num_qubits: int, num_gates: int) -> Cost:
        quad = num_qubits**2
        time = self.coeff["tab_gate"] * num_gates * quad
        memory = self.coeff["tab_mem"] * quad
        depth = math.log2(num_qubits) if num_qubits > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    def mps(
        self,
        num_qubits: int,
        num_1q_gates: int,
        num_2q_gates: int,
        chi: int | Sequence[int] | None,
        *,
        svd: bool = False,
        gates: Iterable["Gate"] | None = None,
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
            Bond dimensions along the chain.  A scalar applies the same
            dimension to every bond while an iterable specifies per-bond
            dimensions.  If ``None``, ``gates`` must be provided and bond
            dimensions are derived from :meth:`bond_dimensions`.
        svd:
            If ``True``, include an additional cost for the singular value
            decomposition and truncation step performed after entangling gates.
        gates:
            Optional gate sequence used to derive bond dimensions when ``chi``
            is ``None``.

        Notes
        -----
        Single-qubit gates scale with the product of adjacent bond dimensions
        while two-qubit gates scale with the product of the left, shared and
        right bonds.  The optional truncation step adds a term scaling as
        :math:`\chi^3 \log \chi` per two-qubit gate.  Temporary workspace
        required for SVD is controlled via the ``mps_temp_mem`` coefficient.
        """

        n = num_qubits
        if chi is None:
            if gates is None:
                raise ValueError("gates must be provided when chi is None")
            bond_dims = self.bond_dimensions(n, gates)
        elif isinstance(chi, Iterable) and not isinstance(chi, (str, bytes)):
            bond_dims = list(chi)
        else:
            bond_dims = [int(chi)] * max(0, n - 1)

        if len(bond_dims) < max(0, n - 1):
            bond_dims.extend([1] * (max(0, n - 1) - len(bond_dims)))
        elif len(bond_dims) > max(0, n - 1):
            bond_dims = bond_dims[: n - 1]

        left = [1] + bond_dims
        right = bond_dims + [1]
        site_costs = [l * r for l, r in zip(left, right)]
        bond_costs = [left[i] * bond_dims[i] * right[i + 1] for i in range(len(bond_dims))]

        time = self.coeff["mps_gate_1q"] * num_1q_gates * sum(site_costs)
        if n > 1:
            avg_bond_cost = sum(bond_costs) / (n - 1)
            time += self.coeff["mps_gate_2q"] * num_2q_gates * n * avg_bond_cost
            if svd and num_2q_gates > 0:
                trunc = sum(
                    c * math.log2(b) if b > 1 else 0.0
                    for c, b in zip(bond_costs, bond_dims)
                )
                trunc /= (n - 1)
                time += self.coeff["mps_trunc"] * num_2q_gates * n * trunc
        elif svd and num_2q_gates > 0:
            # Defensive branch for single qubit with svd flag
            time += 0.0

        memory = self.coeff["mps_mem"] * sum(site_costs)
        if svd and num_2q_gates > 0 and bond_costs:
            memory += self.coeff.get("mps_temp_mem", 0.0) * max(bond_costs)

        depth = math.log2(num_qubits) if num_qubits > 0 else 0.0
        return Cost(time=time, memory=memory, log_depth=depth)

    def decision_diagram(self, num_gates: int, frontier: int) -> Cost:
        """Estimate cost for decision diagram simulation.

        The active node count is approximated by ``frontier * log2(frontier)``
        with a linear fallback for small frontiers.
        """

        threshold = 2
        if frontier < threshold:
            active_nodes = frontier
        else:
            active_nodes = frontier * math.log2(frontier)
        time = self.coeff["dd_gate"] * num_gates * active_nodes
        memory = self.coeff["dd_mem"] * active_nodes
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
        window_1q_gates: int = 0,
        window_2q_gates: int = 0,
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
        window_1q_gates, window_2q_gates:
            Gate counts within the dense window used by the LW primitive.
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

        full = 2**num_qubits
        ingest_time = self.coeff[f"ingest_{target.value}"] * full
        base_time = self.coeff.get("conversion_base", 0.0)
        overhead = ingest_time + base_time

        # --- B2B primitive ---
        svd_cost = min(num_qubits * (rank**2), rank * (num_qubits**2))
        b2b_time = (
            self.coeff["b2b_svd"] * svd_cost
            + self.coeff["b2b_copy"] * num_qubits * (rank**2)
            + overhead
        )
        b2b_mem = max(num_qubits * rank**2, full)

        # --- LW primitive ---
        w = window if window is not None else min(num_qubits, 4)
        dense = 2**w
        gate_time = (
            self.coeff["sv_gate_1q"] * window_1q_gates
            + self.coeff["sv_gate_2q"] * window_2q_gates
        )
        lw_time = self.coeff["lw_extract"] * dense + gate_time * dense + overhead
        lw_mem = max(dense, full)

        # --- ST primitive ---
        chi_cap = int(self.coeff.get("st_chi_cap", 16)) or 16
        chi_tilde = min(rank, chi_cap)
        st_time = self.coeff["st_stage"] * (chi_tilde**3) + overhead
        st_mem = max(num_qubits * (chi_tilde**2), full)

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
