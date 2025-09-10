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
from typing import Dict, Optional, Set, List, Tuple

from .circuit import Circuit
from .cost import Backend, Cost, CostEstimator


@dataclass
class AnalysisResult:
    """Container bundling the results of a circuit analysis."""

    gate_distribution: Dict[str, int]
    entanglement: Dict[str, float]
    resources: Dict[Backend, Cost]
    parallel_layers: List[List[int]]
    critical_path_length: int


class CircuitAnalyzer:
    """Perform static analysis on a :class:`~quasar.circuit.Circuit`."""

    def __init__(self, circuit: Circuit, estimator: Optional[CostEstimator] = None, chi: int = 4):
        self.circuit = circuit
        self.estimator = estimator or CostEstimator()
        self.chi = chi

    # ------------------------------------------------------------------
    def gate_distribution(self) -> Dict[str, int]:
        """Return the frequency of each gate type in the circuit."""

        return dict(Counter(g.gate for g in self.circuit.gates))

    # ------------------------------------------------------------------
    def _entanglement_graph(self) -> Dict[int, Set[int]]:
        """Build an undirected graph from multi-qubit gates."""

        graph: Dict[int, Set[int]] = defaultdict(set)
        for gate in self.circuit.gates:
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

        multi_qubit_gate_count = sum(1 for g in self.circuit.gates if len(g.qubits) > 1)

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
    def _dependency_graph(self) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """Construct a gate dependency graph.

        Returns
        -------
        tuple
            ``(preds, succs)`` adjacency lists where ``preds[i]`` contains the
            indices of gates that must precede gate ``i`` and ``succs[i]`` the
            gates depending on ``i``.
        """

        n = len(self.circuit.gates)
        preds: Dict[int, Set[int]] = {i: set() for i in range(n)}
        succs: Dict[int, Set[int]] = {i: set() for i in range(n)}
        last_seen: Dict[int, int] = {}
        for idx, gate in enumerate(self.circuit.gates):
            for q in gate.qubits:
                if q in last_seen:
                    dep = last_seen[q]
                    preds[idx].add(dep)
                    succs[dep].add(idx)
                last_seen[q] = idx
        return preds, succs

    def parallel_layers(self) -> List[List[int]]:
        """Return groups of gates that can execute in parallel."""

        preds, succs = self._dependency_graph()
        indegree = {i: len(p) for i, p in preds.items()}
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
        )
