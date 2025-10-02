"""Utilities for appending random non-Clifford tails to Qiskit circuits."""
from __future__ import annotations

import math
import random
from typing import Any, Sequence, Tuple, Union

try:  # pragma: no cover - only used for typing
    from typing import Protocol
except ImportError:  # Python <3.8 fallback (not expected in project)
    Protocol = object  # type: ignore


_TWO_PI = 2.0 * math.pi
_PI_OVER_4 = math.pi / 4.0


class _GateChoiceProtocol(Protocol):  # pragma: no cover - typing helper only
    """Subset of the QuantumCircuit API used by the tail generator."""

    @property
    def num_qubits(self) -> int:
        ...

    def rx(self, theta: float, qubit: int) -> Any:
        ...

    def ry(self, theta: float, qubit: int) -> Any:
        ...

    def rz(self, theta: float, qubit: int) -> Any:
        ...

    def crx(self, theta: float, control: int, target: int) -> Any:
        ...

    def cry(self, theta: float, control: int, target: int) -> Any:
        ...

    def rzx(self, theta: float, qubit1: int, qubit2: int) -> Any:
        ...

    def rxx(self, theta: float, qubit1: int, qubit2: int) -> Any:
        ...

    def ryy(self, theta: float, qubit1: int, qubit2: int) -> Any:
        ...


GateSpec = Union[str, Tuple[str, float]]


def _dist_to_pi_over_4(theta: float) -> float:
    """Distance from *theta* to the nearest multiple of π/4 on the circle."""

    t = theta % _TWO_PI
    min_distance = _TWO_PI
    for k in range(8):  # multiples covering [0, 2π)
        target = k * _PI_OVER_4
        direct = abs(t - target)
        min_distance = min(min_distance, direct, _TWO_PI - direct)
    return min_distance


def sample_nonclifford_angle(rng: random.Random, eps: float = 1e-3) -> float:
    """Sample θ ∼ U(0, 2π) rejecting values within *eps* of Clifford-compatible angles."""

    while True:
        theta = rng.random() * _TWO_PI
        if _dist_to_pi_over_4(theta) > eps:
            return theta


def _normalize_gate_choices(choices: Sequence[GateSpec]) -> Tuple[Tuple[str, float], ...]:
    weighted: list[Tuple[str, float]] = []
    for spec in choices:
        if isinstance(spec, tuple):
            name, weight = spec
        else:
            name, weight = spec, 1.0
        if weight <= 0:
            continue
        weighted.append((name, float(weight)))
    if not weighted:
        raise ValueError("At least one gate choice with positive weight is required.")
    return tuple(weighted)


def _weighted_choice(rng: random.Random, choices: Tuple[Tuple[str, float], ...]) -> str:
    total = sum(weight for _, weight in choices)
    pick = rng.random() * total
    accum = 0.0
    for name, weight in choices:
        accum += weight
        if pick <= accum:
            return name
    return choices[-1][0]


def append_random_tail_qiskit(
    qc: _GateChoiceProtocol,
    *,
    layers: int = 2,
    twoq_prob: float = 0.3,
    angle_eps: float = 1e-3,
    oneq_ops: Sequence[GateSpec] = ("rx", "ry", "rz"),
    twoq_ops: Sequence[GateSpec] = ("crx", "cry", "rzx", "rxx", "ryy"),
    seed: int = 2025,
) -> None:
    """Append layers of random non-Clifford rotations to a Qiskit circuit in-place."""

    if layers <= 0:
        return
    if not (0.0 <= twoq_prob <= 1.0):
        raise ValueError("twoq_prob must lie in [0, 1].")
    rng = random.Random(seed)
    num_qubits = qc.num_qubits

    oneq_choices = _normalize_gate_choices(oneq_ops)
    twoq_choices = _normalize_gate_choices(twoq_ops)

    def apply_single_qubit_layer() -> None:
        for qubit in range(num_qubits):
            gate = _weighted_choice(rng, oneq_choices)
            theta = sample_nonclifford_angle(rng, eps=angle_eps)
            if gate == "rx":
                qc.rx(theta, qubit)
            elif gate == "ry":
                qc.ry(theta, qubit)
            elif gate == "rz":
                qc.rz(theta, qubit)
            else:
                raise ValueError(f"Unsupported 1-qubit gate for tail: {gate}")

    def apply_two_qubit_layer(offset: int) -> None:
        if num_qubits < 2:
            return
        for a in range(offset, num_qubits - 1, 2):
            b = a + 1
            if rng.random() > twoq_prob:
                continue
            gate = _weighted_choice(rng, twoq_choices)
            theta = sample_nonclifford_angle(rng, eps=angle_eps)
            if gate == "crx":
                qc.crx(theta, a, b)
            elif gate == "cry":
                qc.cry(theta, a, b)
            elif gate == "rzx":
                qc.rzx(theta, a, b)
            elif gate == "rxx":
                qc.rxx(theta, a, b)
            elif gate == "ryy":
                qc.ryy(theta, a, b)
            else:
                raise ValueError(f"Unsupported 2-qubit gate for tail: {gate}")

    for _ in range(layers):
        apply_single_qubit_layer()
        apply_two_qubit_layer(0)
        apply_two_qubit_layer(1)


__all__ = [
    "append_random_tail_qiskit",
    "sample_nonclifford_angle",
    "_dist_to_pi_over_4",
]
