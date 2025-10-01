from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence

from .cost import Backend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .circuit import Gate


@dataclass
class FragmentMetrics:
    """Incremental statistics for a gate fragment.

    The helper mirrors the logic previously embedded in ``quasar.ssd``'s
    ``_PathMetrics`` class but is now available as a shared utility.  Metrics
    are updated incrementally as gates are appended and can be cloned or
    reconstructed from existing sequences without materialising a
    :class:`~quasar.circuit.Circuit` instance.
    """

    gates: List["Gate"] = field(default_factory=list)
    gate_names: List[str] = field(default_factory=list)
    qubits: set[int] = field(default_factory=set)
    num_gates: int = 0
    num_meas: int = 0
    num_1q: int = 0
    num_2q: int = 0
    num_t: int = 0
    phase_rotations: set[float] = field(default_factory=set)
    amplitude_rotations: set[float] = field(default_factory=set)
    nnz: int = 1

    def update(self, gate: "Gate") -> None:
        """Incorporate ``gate`` into the tracked statistics."""

        from .sparsity import BRANCHING_GATES, is_controlled
        from .symmetry import AMPLITUDE_ROTATION_GATES, PHASE_ROTATION_GATES

        name = gate.gate.upper()
        self.gates.append(gate)
        self.gate_names.append(name)
        if gate.qubits:
            self.qubits.update(gate.qubits)
        self.num_gates += 1
        if name in {"MEASURE", "RESET"}:
            self.num_meas += 1
        elif len(gate.qubits) <= 1:
            self.num_1q += 1
        else:
            self.num_2q += 1
        if name in {"T", "TDG"}:
            self.num_t += 1

        numeric_value: float | None = None
        for param in gate.params.values():
            if isinstance(param, (int, float)):
                numeric_value = float(param)
                break
        if numeric_value is not None:
            rounded = round(numeric_value, 12)
            if name in PHASE_ROTATION_GATES:
                self.phase_rotations.add(rounded)
            if name in AMPLITUDE_ROTATION_GATES:
                self.amplitude_rotations.add(rounded)

        base_gate = gate.gate.upper().lstrip("C")
        if base_gate in BRANCHING_GATES:
            if is_controlled(gate):
                self.nnz += 1
            else:
                self.nnz *= 2
        self._clamp_nnz()

    def update_many(self, gates: Iterable["Gate"]) -> None:
        """Update the metrics with ``gates`` in sequence."""

        for gate in gates:
            self.update(gate)

    def copy(self) -> "FragmentMetrics":
        """Return a deep-ish copy suitable for speculative updates."""

        return FragmentMetrics(
            gates=list(self.gates),
            gate_names=list(self.gate_names),
            qubits=set(self.qubits),
            num_gates=self.num_gates,
            num_meas=self.num_meas,
            num_1q=self.num_1q,
            num_2q=self.num_2q,
            num_t=self.num_t,
            phase_rotations=set(self.phase_rotations),
            amplitude_rotations=set(self.amplitude_rotations),
            nnz=self.nnz,
        )

    @classmethod
    def from_gates(cls, gates: Sequence["Gate"] | Iterable["Gate"]) -> "FragmentMetrics":
        """Construct metrics for ``gates`` without creating a circuit."""

        metrics = cls()
        metrics.update_many(gates)
        return metrics

    def _clamp_nnz(self) -> None:
        full_dim = 2 ** len(self.qubits) if self.qubits else 1
        if self.nnz > full_dim:
            self.nnz = full_dim

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def sparsity(self) -> float:
        num_qubits = self.num_qubits
        if num_qubits == 0:
            return 1.0
        full_dim = 2 ** num_qubits
        nnz = min(self.nnz, full_dim)
        if nnz >= full_dim and num_qubits <= 12:
            slack = max(1, full_dim // (4 * max(1, num_qubits)))
            nnz = max(full_dim - slack, 1)
        return 1 - nnz / full_dim

    @property
    def phase_rotation_diversity(self) -> int:
        return len(self.phase_rotations)

    @property
    def amplitude_rotation_diversity(self) -> int:
        return len(self.amplitude_rotations)

    def metrics_entry(self) -> Dict[str, Any]:
        return {
            "qubits": tuple(sorted(self.qubits)),
            "num_qubits": self.num_qubits,
            "num_gates": self.num_gates,
            "num_meas": self.num_meas,
            "num_1q": self.num_1q,
            "num_2q": self.num_2q,
            "num_t": self.num_t,
        }


@dataclass
class FragmentMetricRecord:
    """Structured fragment metrics suitable for serialization."""

    start: int
    end: int
    backend: Backend | None
    metrics: Dict[str, Any]
    parallel: Sequence[Sequence[int]] | None = None

    def to_mapping(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping for ``self``."""

        payload: Dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "backend": self.backend.name if self.backend is not None else None,
        }
        if self.parallel:
            payload["parallel_groups"] = [list(group) for group in self.parallel]

        for key, value in self.metrics.items():
            payload[key] = _normalise_value(value)
        return payload


def _normalise_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalise_value(v) for v in value]
    if isinstance(value, list):
        return [_normalise_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise_value(v) for k, v in value.items()}
    return value


def _normalise_record(record: FragmentMetricRecord | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(record, FragmentMetricRecord):
        return record.to_mapping()
    if isinstance(record, Mapping):
        return {k: _normalise_value(v) for k, v in record.items()}
    raise TypeError(f"Unsupported fragment metric record: {type(record)!r}")


def export_fragment_metrics_json(
    records: Sequence[FragmentMetricRecord | Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write ``records`` to ``path`` in JSON format.

    Parameters
    ----------
    records:
        Iterable of :class:`FragmentMetricRecord` or plain mappings describing
        fragment metrics.
    path:
        Destination file.  Parent directories are created automatically.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [_normalise_record(record) for record in records]
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")
    return destination


def export_fragment_metrics_csv(
    records: Sequence[FragmentMetricRecord | Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write ``records`` to ``path`` in CSV format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [_normalise_record(record) for record in records]
    if not rows:
        destination.write_text("", encoding="utf8")
        return destination

    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: _format_csv_value(row.get(key)) for key in fieldnames
            }
            writer.writerow(formatted)
    return destination


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value
