from __future__ import annotations

r"""Wrapper around the Stim tableau simulator.

This backend directly uses :class:`stim.TableauSimulator` instead of building a
``stim.Circuit`` and calling :meth:`stim.Circuit.to_tableau`.  The circuit
conversion method always assumes the qubits begin in the :math:`|0\dots0\rangle`
state and exposes no way to supply an existing tableau.  By working with
``TableauSimulator`` we can ingest arbitrary stabilizer states via
``do_tableau`` and continue evolving them.  Until ``stim`` gains native support
for starting ``Circuit.to_tableau`` from an initial tableau, this approach keeps
the backend flexible without an extra conversion step.
"""

from dataclasses import dataclass, field
from typing import Dict, Sequence, List, Tuple
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
    _benchmark_mode: bool = field(default=False, init=False)
    _benchmark_ops: List[Tuple[str, Sequence[int], Dict[str, float] | None]] = field(
        default_factory=list, init=False
    )
    _benchmark_tableau: stim.Tableau | None = field(default=None, init=False)

    _ALIASES: Dict[str, str] = field(
        default_factory=lambda: {
            "SDG": "s_dag",
        }
    )

    def load(self, num_qubits: int, **_: dict) -> None:
        """Initialise the Stim simulator with a given number of qubits."""
        self.simulator = stim.TableauSimulator()
        self.simulator.do_tableau(stim.Tableau(num_qubits), list(range(num_qubits)))
        self.num_qubits = num_qubits
        self.history.clear()

    def ingest(
        self,
        state: stim.Tableau | stim.TableauSimulator,
        *,
        num_qubits: int | None = None,
        mapping: Sequence[int] | None = None,
    ) -> None:
        """Initialise simulator from a Stim tableau or simulator."""
        if isinstance(state, stim.TableauSimulator):
            tableau = state.current_inverse_tableau().inverse()
            n = state.num_qubits
        elif isinstance(state, stim.Tableau):
            tableau = state.inverse()
            n = len(state)
        elif getattr(state, "num_qubits", None) is not None:
            n = int(getattr(state, "num_qubits"))
            tableau = stim.Tableau(n)
        else:
            raise TypeError("Unsupported state for Stim backend")
        if num_qubits is None:
            num_qubits = n
        if mapping is None:
            if n != num_qubits:
                raise ValueError("num_qubits does not match state size")
            mapping = list(range(n))
        if len(mapping) != n:
            raise ValueError("Mapping length does not match state size")
        self.simulator = stim.TableauSimulator()
        self.simulator.do_tableau(stim.Tableau(num_qubits), list(range(num_qubits)))
        self.simulator.do_tableau(tableau, list(mapping))
        self.num_qubits = num_qubits
        self.history.clear()

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self._benchmark_mode:
            self._benchmark_ops.append((name, tuple(qubits), params))
            return
        if self.simulator is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        lname = self._ALIASES.get(name.upper(), name.lower())
        if lname in {"ccx", "ccz"}:
            raise NotImplementedError(
                "CCX and CCZ gates must be decomposed before execution"
            )
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
            raise NotImplementedError(f"Unsupported Stim gate {name}")
        func(*qubits)
        self.history.append(name.upper())

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Apply any operations queued during benchmark preparation."""
        if not self._benchmark_ops:
            return
        ops = self._benchmark_ops
        self._benchmark_ops = []
        self._benchmark_mode = False
        for name, qubits, params in ops:
            self.apply_gate(name, qubits, params)

    def run_benchmark(self, *, return_state: bool = False) -> SSD | None:
        """Execute queued operations and optionally return the final state."""
        self.run()
        self._benchmark_tableau = None
        if self.simulator is not None:
            try:
                self._benchmark_tableau = self.simulator.current_inverse_tableau()
            except Exception:
                self._benchmark_tableau = None
        if return_state:
            return self.extract_ssd()
        return None

    def extract_ssd(self) -> SSD:
        tableau = getattr(self, "_benchmark_tableau", None)
        if tableau is not None:
            self._benchmark_tableau = None
        else:
            self.run()
            if self.simulator is not None:
                try:
                    tableau = self.simulator.current_inverse_tableau()
                except Exception:
                    tableau = None
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
            state=tableau,
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> Sequence[complex]:
        """Return a dense statevector for the current tableau state."""
        self.run()
        if self.simulator is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        # Stim returns a numpy array of complex amplitudes
        return self.simulator.state_vector().astype(complex)
