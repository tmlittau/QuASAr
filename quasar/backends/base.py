from __future__ import annotations

"""Common backend interface for QuASAr simulators."""

from typing import Sequence, Dict, Any, TYPE_CHECKING

from ..cost import Backend as BackendType

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..ssd import SSD


class Backend:
    """Abstract simulation backend.

    Concrete backends need to implement three core methods:

    ``load``
        Prepare the backend for a given number of qubits.
    ``apply_gate``
        Execute a quantum gate on the backend's internal state.
    ``extract_ssd``
        Convert the backend's state into a :class:`~quasar.ssd.SSD` object
        so that the scheduler can reason about further conversions or
        simulations.
    """

    #: Backend type exposed to the scheduler.
    backend: BackendType = BackendType.STATEVECTOR

    def load(self, num_qubits: int, **kwargs: Any) -> None:
        """Initialise the simulator for ``num_qubits`` qubits."""
        raise NotImplementedError

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        """Execute ``name`` on ``qubits``.

        Parameters
        ----------
        name:
            Gate identifier (e.g. ``"H"`` or ``"CX"``).
        qubits:
            Target qubit indices.
        params:
            Optional gate parameters.
        """
        raise NotImplementedError

    def extract_ssd(self) -> 'SSD':
        """Return a :class:`~quasar.ssd.SSD` describing the backend state."""
        raise NotImplementedError
