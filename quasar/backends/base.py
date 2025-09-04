from __future__ import annotations

"""Common backend interface for QuASAr simulators."""

from typing import Sequence, Dict, Any, TYPE_CHECKING, List, Tuple

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

    # ------------------------------------------------------------------
    def prepare_benchmark(self, circuit: Any | None = None) -> None:
        """Enable benchmark mode so that ``apply_gate`` merely records operations.

        Backends can use this hook to defer expensive state updates until
        :meth:`run_benchmark` is invoked.  Implementations are expected to
        populate ``_benchmark_mode`` and ``_benchmark_ops`` attributes but no
        further action is required here.
        """
        self._benchmark_mode = True  # type: ignore[attr-defined]
        self._benchmark_ops = []  # type: ignore[attr-defined]

    def run_benchmark(self) -> Any:
        """Execute any operations queued during benchmark preparation.

        The default implementation replays the stored gate descriptors via
        :meth:`apply_gate` with benchmark mode disabled and then attempts to
        return a state representation.  Sub-classes may override this method to
        customise the returned data.
        """
        ops: List[Tuple[str, Sequence[int], Dict[str, float] | None]] = getattr(
            self, "_benchmark_ops", []
        )
        self._benchmark_mode = False  # type: ignore[attr-defined]
        for name, qubits, params in ops:
            self.apply_gate(name, qubits, params)
        self._benchmark_ops = []  # type: ignore[attr-defined]
        try:
            return self.statevector()
        except Exception:
            try:
                return self.extract_ssd()
            except Exception:
                return None

    # ------------------------------------------------------------------
    def ingest(
        self,
        state: Any,
        *,
        num_qubits: int | None = None,
        mapping: Sequence[int] | None = None,
    ) -> None:
        """Load an externally prepared ``state`` into the backend.

        Parameters
        ----------
        state:
            Backend specific representation to ingest.
        num_qubits:
            Optional global register size.  When provided, the backend must
            embed ``state`` into a register of this size.
        mapping:
            Mapping of qubits in ``state`` to positions in the global
            register.  ``len(mapping)`` must match the number of qubits
            represented by ``state``.  When ``None``, the state is assumed to
            describe the full register in order ``0..n-1``.

        Implementations should update ``num_qubits`` and any internal caches
        accordingly.
        """
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

    # ------------------------------------------------------------------
    def statevector(self) -> Sequence[complex]:
        """Return the full statevector representing the backend state.

        Backends that do not maintain a dense representation may override
        this to reconstruct a statevector on demand or raise
        ``NotImplementedError`` if such extraction is not supported.
        """
        raise NotImplementedError
