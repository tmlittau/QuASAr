from __future__ import annotations

"""Circuit analysis utilities for QuASAr.

The analyzer inspects a :class:`~quasar.circuit.Circuit` and derives
several high level metrics:

* **Gate distribution** – frequency of each gate type in the circuit.
* **Entanglement metrics** – connectivity properties of the qubit
  interaction graph induced by multi-qubit gates.
* **Resource estimates** – runtime and memory predictions for the
  supported simulation backends using :class:`~quasar.cost.CostEstimator`.
"""

from dataclasses import dataclass
from collections import Counter, defaultdict, deque
import math
from typing import Dict, Optional, Set, List, Tuple

from .circuit import Circuit, _is_multiple_of_pi
from .cost import Backend, Cost, CostEstimator


@dataclass
class AnalysisResult:
    """Container bundling the results of a circuit analysis.

    Attributes
    ----------
    gate_entanglement:
        Entanglement annotation for each gate in topological order.
    method_compatibility:
        Simulation backends compatible with each gate.
    """

    gate_distribution: Dict[str, int]
    entanglement: Dict[str, float]
    resources: Dict[Backend, Cost]
    parallel_layers: List[List[int]]
    critical_path_length: int
    rotation_angles: Dict[str, int]
    graph_metrics: Dict[str, float]
    clifford_counts: Dict[str, int]
    gate_depths: Dict[str, int]
    gate_entanglement: List[str]
    method_compatibility: List[List[str]]


class CircuitAnalyzer:
    """Perform static analysis on a :class:`~quasar.circuit.Circuit`."""

    def __init__(self, circuit: Circuit, estimator: Optional[CostEstimator] = None, chi: int = 4):
        self.circuit = circuit
        self.estimator = estimator or CostEstimator()
        self.chi = chi

    # ------------------------------------------------------------------
    def gate_distribution(self) -> Dict[str, int]:
        """Return the frequency of each gate type in the circuit."""

        return dict(Counter(g.gate for g in self.circuit.topological()))

    # ------------------------------------------------------------------
    def _entanglement_graph(self) -> Dict[int, Set[int]]:
        """Build an undirected graph from multi-qubit gates."""

        graph: Dict[int, Set[int]] = defaultdict(set)
        for gate in self.circuit.topological():
            qubits = gate.qubits
            if len(qubits) < 2:
                continue
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    a, b = qubits[i], qubits[j]
                    graph[a].add(b)
                    graph[b].add(a)
        return graph

    def entanglement_metrics(self) -> Dict[str, float]:
        """Compute simple connectivity metrics for the circuit.

        The routine interprets the circuit as an undirected graph where
        vertices are qubits and edges connect qubits that participate in a
        multi-qubit gate.  It then reports the number of connected
        components, the size of the largest component and the number of
        multi-qubit gates.
        """

        graph = self._entanglement_graph()
        visited: Set[int] = set()
        component_sizes = []
        for q in graph:
            if q in visited:
                continue
            queue = deque([q])
            visited.add(q)
            size = 0
            while queue:
                node = queue.popleft()
                size += 1
                for nb in graph[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            component_sizes.append(size)

        # Account for isolated qubits which never appear in a multi-qubit
        # interaction.
        isolated = [q for q in range(self.circuit.num_qubits) if q not in graph]
        component_sizes.extend([1] * len(isolated))

        multi_qubit_gate_count = sum(1 for g in self.circuit.topological() if len(g.qubits) > 1)

        if component_sizes:
            max_size = max(component_sizes)
            avg_size = sum(component_sizes) / len(component_sizes)
        else:
            max_size = 0
            avg_size = 0.0

        return {
            "multi_qubit_gate_count": float(multi_qubit_gate_count),
            "connected_components": float(len(component_sizes)),
            "max_connected_size": float(max_size),
            "avg_connected_size": float(avg_size),
        }

    # ------------------------------------------------------------------
    def graph_metrics(self) -> Dict[str, float]:
        """Return graph-theoretic metrics for the interaction graph."""

        graph = self._entanglement_graph()
        clustering: Dict[int, float] = {}
        for node, neighbors in graph.items():
            if len(neighbors) < 2:
                clustering[node] = 0.0
                continue
            links = 0
            neigh = list(neighbors)
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    if neigh[j] in graph.get(neigh[i], set()):
                        links += 1
            possible = len(neigh) * (len(neigh) - 1) / 2
            clustering[node] = links / possible if possible else 0.0
        avg_clustering = sum(clustering.values()) / len(clustering) if clustering else 0.0
        return {"avg_clustering_coefficient": float(avg_clustering)}

    # ------------------------------------------------------------------
    def rotation_angle_stats(self) -> Dict[str, int]:
        """Collect statistics about rotation gate angles."""

        stats: Dict[str, int] = {
            "multiple_of_pi": 0,
            "non_multiple_of_pi": 0,
        }
        angles: Counter[float] = Counter()
        for gate in self.circuit.topological():
            if not gate.params:
                continue
            angle = float(next(iter(gate.params.values())))
            angles[angle] += 1
            if _is_multiple_of_pi(angle):
                stats["multiple_of_pi"] += 1
            else:
                stats["non_multiple_of_pi"] += 1
        stats.update({f"angle_{a}": c for a, c in angles.items()})
        return stats

    # ------------------------------------------------------------------
    def clifford_counts(self) -> Dict[str, int]:
        """Count Clifford versus non-Clifford gates."""

        clifford_gate_names = {
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "SDG",
            "CX",
            "CY",
            "CZ",
            "SWAP",
        }
        counts = {"clifford": 0, "non_clifford": 0}
        for g in self.circuit.topological():
            name = g.gate.upper()
            is_clifford = False
            if name in clifford_gate_names:
                is_clifford = True
            elif name in {"RX", "RY", "RZ", "P"} and g.params:
                angle = float(next(iter(g.params.values())))
                is_clifford = math.isclose(
                    angle / (math.pi / 2), round(angle / (math.pi / 2)), abs_tol=1e-9
                )
            if is_clifford:
                counts["clifford"] += 1
            else:
                counts["non_clifford"] += 1
        return counts

    # ------------------------------------------------------------------
    def gate_depths(self, layers: Optional[List[List[int]]] = None) -> Dict[str, int]:
        """Return per-gate-type depth based on parallel layers."""

        if layers is None:
            layers = self.parallel_layers()
        depth: Dict[str, int] = defaultdict(int)
        for idx, layer in enumerate(layers, start=1):
            for gate_idx in layer:
                name = self.circuit.gates[gate_idx].gate
                depth[name] = max(depth[name], idx)
        return dict(depth)

    # ------------------------------------------------------------------
    def gate_entanglement(self) -> List[str]:
        """Return per-gate entanglement annotations.

        Gates are tagged as ``"none"`` if they act on a single qubit,
        ``"creates"`` if they introduce entanglement between previously
        separate qubit sets and ``"modifies"`` when operating within an
        already entangled group.
        """

        max_index = max((q for g in self.circuit.gates for q in g.qubits), default=-1)
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

        annotations: List[str] = []
        for gate in self.circuit.topological():
            qubits = gate.qubits
            if len(qubits) < 2:
                tag = "none"
            else:
                roots = {find(q) for q in qubits}
                if len(roots) > 1:
                    tag = "creates"
                    base = qubits[0]
                    for q in qubits[1:]:
                        union(base, q)
                else:
                    tag = "modifies"
            annotations.append(tag)
            gate.entanglement = tag
        return annotations

    # ------------------------------------------------------------------
    def method_compatibility(self) -> List[List[str]]:
        """Return compatible simulation backends for each gate."""

        from .planner import _supported_backends

        compat: List[List[str]] = []
        for gate in self.circuit.topological():
            backends = _supported_backends(
                [gate], circuit=self.circuit, estimator=self.estimator
            )
            methods = [b.name.lower() for b in backends]
            gate.compatible_methods = methods
            compat.append(methods)
        return compat

    # ------------------------------------------------------------------
    def _dependency_graph(self) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """Construct a gate dependency graph.

        Returns
        -------
        tuple
            ``(preds, succs)`` adjacency lists where ``preds[i]`` contains the
            indices of gates that must precede gate ``i`` and ``succs[i]`` the
            gates depending on ``i``.
        """

        order = self.circuit.gates
        mapping = {id(g): i for i, g in enumerate(order)}
        preds: Dict[int, Set[int]] = {
            mapping[id(g)]: {mapping[id(p)] for p in g.predecessors} for g in order
        }
        succs: Dict[int, Set[int]] = {
            mapping[id(g)]: {mapping[id(s)] for s in g.successors} for g in order
        }
        return preds, succs

    def parallel_layers(self) -> List[List[int]]:
        """Return groups of gates that can execute in parallel."""

        preds, succs = self._dependency_graph()
        indegree = {i: len(preds[i]) for i in preds}
        ready = sorted(i for i, d in indegree.items() if d == 0)
        layers: List[List[int]] = []
        while ready:
            layers.append(ready)
            next_ready: List[int] = []
            for node in ready:
                for nb in succs[node]:
                    indegree[nb] -= 1
                    if indegree[nb] == 0:
                        next_ready.append(nb)
            ready = sorted(next_ready)
        return layers

    def critical_path_length(self) -> int:
        """Return the circuit depth derived from dependencies."""

        return len(self.parallel_layers())

    # ------------------------------------------------------------------
    def resource_estimates(self) -> Dict[Backend, Cost]:
        """Estimate runtime and memory for supported simulation backends."""

        num_qubits = self.circuit.num_qubits
        num_gates = len(self.circuit.gates)
        num_meas = sum(
            1 for g in self.circuit.gates if g.gate.upper() in {"MEASURE", "RESET"}
        )
        num_1q = sum(
            1
            for g in self.circuit.gates
            if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = num_gates - num_1q - num_meas
        estimates: Dict[Backend, Cost] = {
            Backend.STATEVECTOR: self.estimator.statevector(
                num_qubits, num_1q, num_2q, num_meas
            ),
            Backend.TABLEAU: self.estimator.tableau(num_qubits, num_gates),
            Backend.MPS: self.estimator.mps(
                num_qubits,
                num_1q + num_meas,
                num_2q,
                chi=self.chi,
                svd=True,
            ),
            Backend.DECISION_DIAGRAM: self.estimator.decision_diagram(
                num_gates=num_gates, frontier=num_qubits
            ),
        }
        return estimates

    # ------------------------------------------------------------------
    def analyze(self) -> AnalysisResult:
        """Return all analysis information in a single structure."""

        layers = self.parallel_layers()
        return AnalysisResult(
            gate_distribution=self.gate_distribution(),
            entanglement=self.entanglement_metrics(),
            resources=self.resource_estimates(),
            parallel_layers=layers,
            critical_path_length=len(layers),
            rotation_angles=self.rotation_angle_stats(),
            graph_metrics=self.graph_metrics(),
            clifford_counts=self.clifford_counts(),
            gate_depths=self.gate_depths(layers),
            gate_entanglement=self.gate_entanglement(),
            method_compatibility=self.method_compatibility(),
        )
