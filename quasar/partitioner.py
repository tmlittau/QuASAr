from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

from .ssd import SSD, SSDPartition
from .cost import Backend, CostEstimator, Cost

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import Circuit, Gate


CLIFFORD_GATES = {
    "I",
    "ID",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SDG",
    "CX",
    "CY",
    "CZ",
    "SWAP",
    "CSWAP",
}


class Partitioner:
    """Partition circuits and assign simulation methods."""

    def __init__(self, estimator: CostEstimator | None = None):
        self.estimator = estimator or CostEstimator()

    def partition(self, circuit: 'Circuit') -> SSD:
        if not circuit.gates:
            return SSD([])

        gates = circuit.gates
        all_qubits = sorted({q for g in gates for q in g.qubits})
        q_to_idx = {q: i for i, q in enumerate(all_qubits)}
        idx_to_q = {i: q for q, i in q_to_idx.items()}
        n = len(all_qubits)

        parent = list(range(n))
        history: Dict[int, List['Gate']] = {i: [] for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parent[rb] = ra
            history[ra].extend(history.pop(rb))

        for gate in gates:
            qubits = [q_to_idx[q] for q in gate.qubits]
            if len(qubits) > 1:
                base = qubits[0]
                for other in qubits[1:]:
                    union(base, other)
                root = find(base)
                history[root].append(gate)
            else:
                root = find(qubits[0])
                history[root].append(gate)

        subsystems: Dict[int, List[int]] = {find(i): [] for i in range(n)}
        for idx in range(n):
            subsystems[find(idx)].append(idx_to_q[idx])

        root_info = [(tuple(sorted(qs)), history[r]) for r, qs in subsystems.items()]

        hist_map: Dict[Tuple[str, ...], List[Tuple[Tuple[int, ...], List['Gate']]]] = {}
        for qs, gate_list in root_info:
            names = tuple(g.gate for g in gate_list)
            hist_map.setdefault(names, []).append((qs, gate_list))

        partitions = []
        for hist, group_list in hist_map.items():
            qubit_groups = [qs for qs, _ in group_list]
            sample_gates = group_list[0][1]
            backend, cost = self._choose_backend(sample_gates, len(qubit_groups[0]))
            partitions.append(
                SSDPartition(
                    subsystems=tuple(qubit_groups),
                    history=hist,
                    backend=backend,
                    cost=cost,
                )
            )
        return SSD(partitions)

    # ------------------------------------------------------------------
    def _choose_backend(self, gates: List['Gate'], num_qubits: int) -> Tuple[Backend, 'Cost']:
        """Select the best simulation backend for a partition."""

        names = [g.gate.upper() for g in gates]
        num_gates = len(gates)
        if all(name in CLIFFORD_GATES for name in names):
            backend = Backend.TABLEAU
            cost = self.estimator.tableau(num_qubits, num_gates)
            return backend, cost

        multi = [g for g in gates if len(g.qubits) > 1]
        local = all(len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi)
        if local:
            backend = Backend.MPS
            cost = self.estimator.mps(num_qubits, num_gates, chi=4)
            return backend, cost

        if num_gates <= 2 ** num_qubits:
            backend = Backend.DECISION_DIAGRAM
            cost = self.estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
            return backend, cost

        backend = Backend.STATEVECTOR
        cost = self.estimator.statevector(num_qubits, num_gates)
        return backend, cost
