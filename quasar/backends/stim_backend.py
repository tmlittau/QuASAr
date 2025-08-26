from __future__ import annotations

"""Wrapper around the Stim tableau simulator."""

from dataclasses import dataclass, field
from typing import Dict, Sequence
import stim

from ..ssd import SSD, SSDPartition
from ..cost import Backend as BackendType
from .base import Backend


@dataclass
class StimBackend(Backend):
    backend: BackendType = BackendType.TABLEAU
    simulator: stim.TableauSimulator | None = field(default=None, init=False)
    num_qubits: int = field(default=0, init=False)
    history: list[str] = field(default_factory=list, init=False)

    _ALIASES: Dict[str, str] = field(
        default_factory=lambda: {
            "SDG": "s_dag",
        }
    )

    def load(self, num_qubits: int, **_: dict) -> None:
        self.simulator = stim.TableauSimulator()
        self.num_qubits = num_qubits
        self.history.clear()

    def ingest(self, state: stim.Tableau | stim.TableauSimulator) -> None:
        """Initialise simulator from a Stim tableau or simulator."""
        if isinstance(state, stim.TableauSimulator):
            self.simulator = state
            self.num_qubits = state.num_qubits
        elif isinstance(state, stim.Tableau):
            self.simulator = stim.TableauSimulator(state)
            self.num_qubits = state.num_qubits
        elif getattr(state, "num_qubits", None) is not None:
            n = int(getattr(state, "num_qubits"))
            self.simulator = stim.TableauSimulator()
            self.simulator.do_tableau(stim.Tableau(n), list(range(n)))
            self.num_qubits = n
        else:
            raise TypeError("Unsupported state for Stim backend")
        self.history.clear()

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self.simulator is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        lname = self._ALIASES.get(name.upper(), name.lower())
        if lname == "i" or lname == "id":
            self.history.append(name.upper())
            return
        if lname == "cswap":
            c, a, b = qubits
            self.simulator.cx(c, b)
            self.simulator.cx(a, b)
            self.simulator.cx(c, a)
            self.simulator.cx(a, b)
            self.simulator.cx(c, b)
            self.history.append(name.upper())
            return
        func = getattr(self.simulator, lname, None)
        if func is None:
            raise ValueError(f"Unsupported Stim gate {name}")
        func(*qubits)
        self.history.append(name.upper())

    def extract_ssd(self) -> SSD:
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> Sequence[complex]:
        """Return a dense statevector for the current tableau state."""
        if self.simulator is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        # Stim returns a numpy array of complex amplitudes
        return self.simulator.state_vector().astype(complex)
