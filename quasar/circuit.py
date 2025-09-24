"""Circuit representation and loading utilities for QuASAr."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, TYPE_CHECKING
import json
import os
import math

from qiskit.circuit import QuantumCircuit
from qiskit_qasm3_import import api as qasm3_api

from .ssd import SSD, SSDPartition
from .cost import Cost, CostEstimator, Backend
from .decompositions import decompose_mcx, decompose_ccz, decompose_cswap


if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as _nx


def _cost_to_dict(cost: Cost) -> Dict[str, float]:
    """Return a plain dictionary representation for :class:`~quasar.cost.Cost`."""

    return {
        "time": cost.time,
        "memory": cost.memory,
        "log_depth": cost.log_depth,
        "conversion": cost.conversion,
        "replay": cost.replay,
    }


def _dict_to_cost(data: Mapping[str, Any]) -> Cost:
    """Create a :class:`~quasar.cost.Cost` from a mapping of numeric values."""

    return Cost(
        time=float(data.get("time", 0.0)),
        memory=float(data.get("memory", 0.0)),
        log_depth=float(data.get("log_depth", 0.0)),
        conversion=float(data.get("conversion", 0.0)),
        replay=float(data.get("replay", 0.0)),
    )


def _is_multiple_of_pi(angle: float) -> bool:
    """Return True if ``angle`` is (approximately) an integer multiple of π."""
    if angle == 0:
        return True
    return math.isclose(angle / math.pi, round(angle / math.pi), abs_tol=1e-9)


@dataclass(eq=False)
class Gate:
    """Simple gate description used when constructing circuits.

    The structure now also carries explicit predecessor/successor links and
    per‑gate metadata used by the planner and scheduler, including
    entanglement annotations and backend compatibility.
    """

    gate: str
    qubits: List[int]
    params: Dict[str, Any] = field(default_factory=dict)
    predecessors: List["Gate"] = field(default_factory=list)
    successors: List["Gate"] = field(default_factory=list)
    entanglement: str = "none"
    compatible_methods: List[str] = field(default_factory=list)
    resource_estimates: Dict[str, "Cost"] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if not isinstance(other, Gate):
            return NotImplemented
        return (
            self.gate == other.gate
            and self.qubits == other.qubits
            and self.params == other.params
        )

    def to_dict(self, *, include_metadata: bool = False) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the gate."""

        data: Dict[str, Any] = {
            "gate": self.gate,
            "qubits": list(self.qubits),
        }
        if self.params:
            data["params"] = dict(self.params)
        if include_metadata:
            data["entanglement"] = self.entanglement
            if self.compatible_methods:
                data["compatible_methods"] = list(self.compatible_methods)
            if self.resource_estimates:
                data["resource_estimates"] = {
                    name: _cost_to_dict(cost)
                    for name, cost in self.resource_estimates.items()
                }
        return data

    def __json__(self) -> Dict[str, Any]:  # pragma: no cover - exercised indirectly
        """Return the structure used by :mod:`json` during serialisation."""

        return self.to_dict(include_metadata=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gate":
        """Create a :class:`Gate` instance from a mapping."""

        params = dict(data.get("params", {}))
        resources: Dict[str, Cost] = {}
        for name, cost in data.get("resource_estimates", {}).items():
            if isinstance(cost, Cost):
                resources[name] = cost
            elif isinstance(cost, Mapping):
                resources[name] = _dict_to_cost(cost)
        return cls(
            gate=str(data["gate"]),
            qubits=list(data["qubits"]),
            params=params,
            entanglement=data.get("entanglement", "none"),
            compatible_methods=list(data.get("compatible_methods", [])),
            resource_estimates=resources,
        )


class Circuit:
    """High level circuit container.

    Parameters
    ----------
    gates:
        Iterable of :class:`Gate` or dictionaries describing gates.

    Attributes
    ----------
    sparsity:
        Estimated sparsity of the circuit's state vector.  This is a heuristic
        and should not be taken as an exact metric.
    symmetry:
        Heuristic symmetry score of the circuit's layers.  Higher values
        indicate more repeated gate patterns.
    classical_state:
        Current classical values for each qubit. A value of ``None`` denotes a
        quantum superposition.
    """

    def __init__(
        self,
        gates: Iterable[Dict[str, Any] | Gate],
        *,
        use_classical_simplification: bool = True,
    ):
        self.use_classical_simplification = use_classical_simplification
        self.gates: List[Gate] = [g if isinstance(g, Gate) else Gate(**g) for g in gates]
        self._expand_decompositions()
        self._num_qubits = self._infer_qubit_count()
        max_index = max((q for gate in self.gates for q in gate.qubits), default=-1)
        # Track classical state: 0/1 for classical qubits, ``None`` for quantum.
        if self.use_classical_simplification:
            self.classical_state: List[int | None] = [0] * (max_index + 1)
        else:
            self.classical_state = [None] * (max_index + 1)
        if self.use_classical_simplification:
            self.simplify_classical_controls()
        else:
            self._build_dag()
            self._annotate_gates()
            self._num_gates = len(self.gates)
            self._depth = self._compute_depth()
            self.ssd = self._create_ssd()
            self.cost_estimates = self._estimate_costs()
            from .sparsity import sparsity_estimate
            self.sparsity = sparsity_estimate(self)
            from .symmetry import (
                symmetry_score,
                phase_rotation_diversity,
                amplitude_rotation_diversity,
            )
            self.symmetry = symmetry_score(self)
            self.phase_rotation_diversity = phase_rotation_diversity(self)
            self.amplitude_rotation_diversity = amplitude_rotation_diversity(self)
            # Backward compatibility
            self.rotation_diversity = self.phase_rotation_diversity

    # ------------------------------------------------------------------
    # JSON serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self, *, include_metadata: bool = True) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the circuit."""

        data: Dict[str, Any] = {
            "use_classical_simplification": self.use_classical_simplification,
            "gates": [gate.to_dict(include_metadata=include_metadata) for gate in self.gates],
        }
        if include_metadata:
            data["num_qubits"] = self._num_qubits
            data["classical_state"] = list(self.classical_state)
            if hasattr(self, "_num_gates"):
                data["num_gates"] = self._num_gates
            if hasattr(self, "_depth"):
                data["depth"] = self._depth
            for attr in (
                "sparsity",
                "symmetry",
                "phase_rotation_diversity",
                "amplitude_rotation_diversity",
                "rotation_diversity",
            ):
                if hasattr(self, attr):
                    data[attr] = getattr(self, attr)
            if hasattr(self, "cost_estimates"):
                data["cost_estimates"] = {
                    name: _cost_to_dict(cost)
                    for name, cost in self.cost_estimates.items()
                }
        return data

    def to_json(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        include_metadata: bool = True,
        **json_kwargs: Any,
    ) -> str:
        """Serialise the circuit to JSON and optionally write it to ``path``."""

        payload = self.to_dict(include_metadata=include_metadata)
        text = json.dumps(payload, **json_kwargs)
        if path is not None:
            with open(os.fspath(path), "w", encoding="utf8") as fh:
                fh.write(text)
        return text

    def __json__(self) -> Dict[str, Any]:  # pragma: no cover - exercised indirectly
        """Return the structure used by :mod:`json` during serialisation."""

        return self.to_dict(include_metadata=True)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def to_networkx_ssd(
        self,
        *,
        include_dependencies: bool = True,
        include_entanglement: bool = True,
        include_conversions: bool = True,
        include_backends: bool = True,
    ) -> "_nx.MultiDiGraph":
        """Return a :class:`networkx.MultiDiGraph` for the circuit's SSD.

        The returned graph mirrors the structure produced by
        :meth:`quasar.ssd.SSD.to_networkx`, exposing partitions, optional
        conversion layers and backend assignments as labelled nodes.  The
        parameters are forwarded verbatim to
        :meth:`quasar.ssd.SSD.to_networkx`.

        Raises
        ------
        RuntimeError
            If :mod:`networkx` is not installed.
        """

        return self.ssd.to_networkx(
            include_dependencies=include_dependencies,
            include_entanglement=include_entanglement,
            include_conversions=include_conversions,
            include_backends=include_backends,
        )

    def _expand_decompositions(self) -> None:
        """Replace higher-level gates by their basic equivalents."""

        expanded: List[Gate] = []
        for gate in self.gates:
            name = gate.gate.upper()
            if name in {"CCX", "MCX"} or (
                name.endswith("X")
                and len(gate.qubits) >= 3
                and name[:-1]
                and set(name[:-1]) == {"C"}
            ):
                *controls, target = gate.qubits
                expanded.extend(decompose_mcx(list(controls), target))
            elif name == "CCZ":
                c1, c2, t = gate.qubits
                expanded.extend(decompose_ccz(c1, c2, t))
            elif name == "CSWAP":
                c, a, b = gate.qubits
                expanded.extend(decompose_cswap(c, a, b))
            else:
                expanded.append(gate)
        self.gates = expanded

    # ------------------------------------------------------------------
    # Classical state tracking and simplification
    # ------------------------------------------------------------------

    def _build_dag(self) -> None:
        """Populate predecessor and successor lists for all gates."""

        for gate in self.gates:
            gate.predecessors = []
            gate.successors = []
        last_seen: Dict[int, Gate] = {}
        for gate in self.gates:
            preds_dict: Dict[int, Gate] = {}
            for q in gate.qubits:
                if q in last_seen:
                    preds_dict[id(last_seen[q])] = last_seen[q]
                last_seen[q] = gate
            preds = list(preds_dict.values())
            gate.predecessors.extend(preds)
            for p in preds:
                p.successors.append(gate)

    def _annotate_gates(self) -> None:
        """Attach per-gate metadata such as entanglement and costs."""

        from .planner import _supported_backends

        estimator = CostEstimator()
        max_index = max((q for g in self.gates for q in g.qubits), default=-1)
        parent = list(range(max_index + 1))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for gate in self.gates:
            qubits = gate.qubits
            if len(qubits) < 2:
                gate.entanglement = "none"
            else:
                roots = {find(q) for q in qubits}
                if len(roots) > 1:
                    gate.entanglement = "creates"
                    base = qubits[0]
                    for q in qubits[1:]:
                        union(base, q)
                else:
                    gate.entanglement = "modifies"

            backends = _supported_backends([gate], circuit=self, estimator=estimator)
            gate.compatible_methods = [b.name.lower() for b in backends]
            resources: Dict[str, Cost] = {}
            num_qubits = (max(qubits) + 1) if qubits else 0
            name = gate.gate.upper()
            num_meas = 1 if name in {"MEASURE", "RESET"} else 0
            num_1q = 1 if len(qubits) == 1 and not num_meas else 0
            num_2q = 1 if len(qubits) > 1 else 0
            for backend in backends:
                if backend == Backend.STATEVECTOR:
                    resources[backend.name.lower()] = estimator.statevector(
                        num_qubits, num_1q, num_2q, num_meas
                    )
                elif backend == Backend.TABLEAU:
                    resources[backend.name.lower()] = estimator.tableau(num_qubits, 1)
                elif backend == Backend.MPS:
                    resources[backend.name.lower()] = estimator.mps(
                        num_qubits, num_1q + num_meas, num_2q, chi=4, svd=True
                    )
                elif backend == Backend.DECISION_DIAGRAM:
                    resources[backend.name.lower()] = estimator.decision_diagram(
                        num_gates=1, frontier=num_qubits
                    )
            gate.resource_estimates = resources

    def topological(self) -> List[Gate]:
        """Return gates in topological order (based on dependencies)."""

        indegree: Dict[int, int] = {id(g): len(g.predecessors) for g in self.gates}
        ready = [g for g in self.gates if indegree[id(g)] == 0]
        order: List[Gate] = []
        while ready:
            g = ready.pop(0)
            order.append(g)
            for s in g.successors:
                key = id(s)
                indegree[key] -= 1
                if indegree[key] == 0:
                    ready.append(s)
        return order

    # Backward compatibility helper
    def to_linear(self) -> List[Gate]:
        """Return gates in topological order.

        This acts as a migration helper for code assuming a linear gate list.
        """

        return self.topological()
    def update_classical_state(self, gate: Gate) -> None:
        """Update ``classical_state`` given ``gate``.

        Classical bits are flipped for ``X``/``Y`` gates, phase gates leave the
        state untouched and branching gates like ``H`` or non-π ``RX``/``RY``
        rotations promote the qubit to a fully quantum (``None``) state.
        Any multi-qubit gate is assumed to create entanglement, marking all
        participating qubits as quantum.
        """

        if not self.use_classical_simplification:
            return

        if len(gate.qubits) != 1:
            for q in gate.qubits:
                self.classical_state[q] = None
            return

        q = gate.qubits[0]
        name = gate.gate.upper()
        state = self.classical_state[q]

        phase_only = {"Z", "S", "T", "SDG", "TDG", "RZ", "P"}

        if state is None:
            if name == "H":
                self.classical_state[q] = None
            elif name in {"RX", "RY"}:
                params = gate.params.values()
                angle = float(next(iter(params), 0.0))
                if not _is_multiple_of_pi(angle):
                    self.classical_state[q] = None
            return

        if name in {"X", "Y"}:
            self.classical_state[q] = 1 - state
        elif name in phase_only:
            pass
        elif name == "H":
            self.classical_state[q] = None
        elif name in {"RX", "RY"}:
            params = gate.params.values()
            angle = float(next(iter(params), 0.0))
            if _is_multiple_of_pi(angle):
                if int(round(angle / math.pi)) % 2 == 1:
                    self.classical_state[q] = 1 - state
            else:
                self.classical_state[q] = None
        else:
            self.classical_state[q] = None

    def simplify_classical_controls(self) -> List[Gate]:
        """Remove gates acting purely on classical bits and reduce controlled gates.

        Returns
        -------
        List[Gate]
            The simplified gate sequence.
        """

        if not self.use_classical_simplification:
            return self.gates

        new_gates: List[Gate] = []
        phase_only = {"Z", "S", "T", "SDG", "TDG", "RZ", "P"}

        for gate in self.gates:
            name = gate.gate.upper()

            # Generic handling for gates with classical controls
            if name.startswith("C") and len(gate.qubits) > 1:
                controls = gate.qubits[:-1]
                target = gate.qubits[-1]
                ctrl_states = [self.classical_state[c] for c in controls]

                # Any quantum control prevents simplification
                if any(state is None for state in ctrl_states):
                    self.update_classical_state(gate)
                    new_gates.append(gate)
                    continue

                # Classical controls evaluated to 0 – gate never fires
                if any(state == 0 for state in ctrl_states):
                    continue

                # All controls are classical 1 – reduce to single-qubit gate
                base = name.lstrip("C")
                reduced = Gate(base, [target], gate.params)
                tgt_state = self.classical_state[target]
                if tgt_state is not None:
                    if base in {"X", "Y"}:
                        self.update_classical_state(reduced)
                        continue
                    if base in phase_only:
                        continue
                    if base in {"RX", "RY"}:
                        params = reduced.params.values()
                        angle = float(next(iter(params), 0.0))
                        if _is_multiple_of_pi(angle):
                            if int(round(angle / math.pi)) % 2 == 1:
                                eq_gate = Gate("X" if base == "RX" else "Y", [target])
                                self.update_classical_state(eq_gate)
                            continue
                self.update_classical_state(reduced)
                new_gates.append(reduced)
                continue

            if len(gate.qubits) == 1:
                q = gate.qubits[0]
                state = self.classical_state[q]

                if state is not None:
                    if name in {"X", "Y"}:
                        self.update_classical_state(gate)
                        continue
                    if name in phase_only:
                        continue
                    if name in {"RX", "RY"}:
                        params = gate.params.values()
                        angle = float(next(iter(params), 0.0))
                        if _is_multiple_of_pi(angle):
                            if int(round(angle / math.pi)) % 2 == 1:
                                eq_gate = Gate("X" if name == "RX" else "Y", [q])
                                self.update_classical_state(eq_gate)
                            continue

                self.update_classical_state(gate)
                new_gates.append(gate)
                continue

            # Multi-qubit gate (either reduced control or entangling)
            self.update_classical_state(gate)
            new_gates.append(gate)

        self.gates = new_gates
        self._build_dag()
        self._annotate_gates()
        self._num_gates = len(new_gates)
        self._depth = self._compute_depth()
        self.ssd = self._create_ssd()
        from .sparsity import sparsity_estimate
        from .symmetry import (
            symmetry_score,
            phase_rotation_diversity,
            amplitude_rotation_diversity,
        )
        self.sparsity = sparsity_estimate(self)
        self.symmetry = symmetry_score(self)
        self.phase_rotation_diversity = phase_rotation_diversity(self)
        self.amplitude_rotation_diversity = amplitude_rotation_diversity(self)
        self.rotation_diversity = self.phase_rotation_diversity
        self.cost_estimates = self._estimate_costs()
        return new_gates

    def enable_classical_simplification(self) -> None:
        """Enable classical control simplification on an existing circuit.

        The classical state is reset to all zeros before re-running
        :meth:`simplify_classical_controls` so that cached metrics such as
        depth, sparsity and cost estimates reflect the simplified circuit.
        """

        self.use_classical_simplification = True
        max_index = max((q for gate in self.gates for q in gate.qubits), default=-1)
        self.classical_state = [0] * (max_index + 1)
        self.simplify_classical_controls()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: Iterable[Dict[str, Any]] | Mapping[str, Any],
        *,
        use_classical_simplification: bool = True,
    ) -> "Circuit":
        """Build a circuit from an iterable of gate dictionaries or a mapping."""

        if isinstance(data, Mapping):
            gates_data = data.get("gates")
            if gates_data is None:
                raise ValueError("Circuit dictionary must contain a 'gates' entry")
            use_classical_simplification = bool(
                data.get("use_classical_simplification", use_classical_simplification)
            )
            gates_iterable = gates_data
        else:
            gates_iterable = data

        gates: List[Dict[str, Any] | Gate] = []
        for gate in gates_iterable:
            if isinstance(gate, Gate):
                gates.append(gate)
            elif isinstance(gate, Mapping):
                gates.append(Gate.from_dict(gate))
            else:
                gates.append(gate)
        return cls(gates, use_classical_simplification=use_classical_simplification)

    @classmethod
    def from_json(
        cls,
        path: str,
        *,
        use_classical_simplification: bool = True,
    ):
        """Load a circuit from a JSON file.

        The JSON file must contain a list of gate dictionaries.
        """
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return cls.from_dict(
            data,
            use_classical_simplification=use_classical_simplification,
        )

    @classmethod
    def from_qiskit(
        cls,
        circuit: QuantumCircuit,
        *,
        use_classical_simplification: bool = True,
    ) -> "Circuit":
        """Build a :class:`Circuit` from a Qiskit ``QuantumCircuit``.

        Parameters
        ----------
        circuit:
            The input Qiskit circuit to convert.
        """
        gates = []
        for ci in circuit.data:
            op = ci.operation
            qubits = [q._index for q in ci.qubits]
            params: Dict[str, Any] = {}
            if getattr(op, "params", None):
                for i, val in enumerate(op.params):
                    params[f"param{i}"] = float(val) if isinstance(val, (int, float)) else val
            gates.append({"gate": op.name.upper(), "qubits": qubits, "params": params})
        return cls(gates, use_classical_simplification=use_classical_simplification)

    @classmethod
    def from_qasm(
        cls,
        path_or_str: str,
        *,
        use_classical_simplification: bool = True,
    ) -> "Circuit":
        """Build a :class:`Circuit` from an OpenQASM 3 string or file.

        Parameters
        ----------
        path_or_str:
            Either a filesystem path to an OpenQASM 3 file or a string
            containing the OpenQASM program.
        """
        if os.path.exists(path_or_str):
            with open(path_or_str, "r", encoding="utf8") as f:
                qasm = f.read()
        else:
            qasm = path_or_str
        qc = qasm3_api.parse(qasm)
        return cls.from_qiskit(qc, use_classical_simplification=use_classical_simplification)

    # ------------------------------------------------------------------
    def _infer_qubit_count(self) -> int:
        if not self.gates:
            return 0
        qubit_indices = [q for gate in self.gates for q in gate.qubits]
        min_q = min(qubit_indices)
        max_q = max(qubit_indices)
        return max_q - min_q + 1

    def _compute_depth(self) -> int:
        """Compute the circuit depth using the dependency DAG."""
        indegree: Dict[int, int] = {id(g): len(g.predecessors) for g in self.gates}
        ready = [g for g in self.gates if indegree[id(g)] == 0]
        depth = 0
        while ready:
            depth += 1
            next_ready: List[Gate] = []
            for gate in ready:
                for succ in gate.successors:
                    key = id(succ)
                    indegree[key] -= 1
                    if indegree[key] == 0:
                        next_ready.append(succ)
            ready = next_ready
        return depth

    def _create_ssd(self) -> SSD:
        """Construct the initial subsystem descriptor."""
        if self._num_qubits == 0:
            return SSD([])
        part = SSDPartition(subsystems=(tuple(range(self._num_qubits)),))
        ssd = SSD([part])
        ssd.build_metadata()
        return ssd

    def _estimate_costs(self) -> Dict[str, Cost]:
        """Estimate simulation costs for standard backends."""

        from .analyzer import CircuitAnalyzer

        analyzer = CircuitAnalyzer(self)
        estimates = analyzer.resource_estimates()
        return {backend.name.lower(): cost for backend, cost in estimates.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        """Total circuit depth."""
        return self._depth

    @property
    def num_gates(self) -> int:
        """Number of gates in the circuit."""
        return self._num_gates

    @property
    def num_qubits(self) -> int:
        """Number of qubits spanned by the circuit."""
        return self._num_qubits


_JSON_SUPPORT_INSTALLED = False


def _install_json_support() -> None:
    """Extend :mod:`json` so that it recognises objects exposing ``__json__``."""

    global _JSON_SUPPORT_INSTALLED
    if _JSON_SUPPORT_INSTALLED:
        return
    _JSON_SUPPORT_INSTALLED = True

    original_default = json.JSONEncoder.default

    def _quasar_json_default(self: json.JSONEncoder, obj: Any) -> Any:
        if hasattr(obj, "__json__"):
            return obj.__json__()
        return original_default(self, obj)

    json.JSONEncoder.default = _quasar_json_default  # type: ignore[assignment]
    json._default_encoder = json.JSONEncoder()


_install_json_support()
