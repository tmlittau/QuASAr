from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

from .ssd import SSD, SSDPartition

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import Circuit


class Partitioner:
    """Partition circuits into independent and equal-state subsystems."""

    def partition(self, circuit: 'Circuit') -> SSD:
        if not circuit.gates:
            return SSD([])

        gates = circuit.gates
        all_qubits = sorted({q for g in gates for q in g.qubits})
        q_to_idx = {q: i for i, q in enumerate(all_qubits)}
        idx_to_q = {i: q for q, i in q_to_idx.items()}
        n = len(all_qubits)

        parent = list(range(n))
        history: Dict[int, List[str]] = {i: [] for i in range(n)}

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
            name = gate.gate
            if len(qubits) > 1:
                base = qubits[0]
                for other in qubits[1:]:
                    union(base, other)
                root = find(base)
                history[root].append(name)
            else:
                root = find(qubits[0])
                history[root].append(name)

        subsystems: Dict[int, List[int]] = {find(i): [] for i in range(n)}
        for idx in range(n):
            subsystems[find(idx)].append(idx_to_q[idx])

        root_info = [(tuple(sorted(qs)), tuple(history[r])) for r, qs in subsystems.items()]

        hist_map: Dict[Tuple[str, ...], List[Tuple[int, ...]]] = {}
        for qs, hist in root_info:
            hist_map.setdefault(hist, []).append(qs)

        partitions = [SSDPartition(subsystems=tuple(groups), history=hist) for hist, groups in hist_map.items()]
        return SSD(partitions)
