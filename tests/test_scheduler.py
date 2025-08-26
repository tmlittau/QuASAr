from quasar import Circuit, Scheduler, Planner
from quasar.backends import StatevectorBackend
from quasar.cost import Backend
from quasar_convert import ConversionEngine
import numpy as np


class CountingConversionEngine(ConversionEngine):
    def __init__(self):
        self.calls = 0

    def convert(self, ssd):
        self.calls += 1
        return super().convert(self.extract_ssd([], 0))


def build_switch_circuit():
    # Initial Clifford segment followed by several non-Clifford ``T`` gates
    # which force the scheduler to switch from a Clifford backend to a dense
    # statevector simulation.
    return Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
        {"gate": "T", "qubits": [1]},
        {"gate": "T", "qubits": [0]},
        {"gate": "T", "qubits": [1]},
        {"gate": "T", "qubits": [0]},
    ])


def test_scheduler_triggers_conversion():
    engine = CountingConversionEngine()
    scheduler = Scheduler(conversion_engine=engine)
    circuit = build_switch_circuit()
    scheduler.run(circuit)
    assert engine.calls == 1


class CountingPlanner(Planner):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def plan(self, circuit):
        self.calls += 1
        return super().plan(circuit)


def test_scheduler_reoptimises_when_requested():
    planner = CountingPlanner()
    scheduler = Scheduler(planner=planner, conversion_engine=CountingConversionEngine())
    circuit = build_switch_circuit()

    triggered = {"done": False}

    def monitor(step, cost):
        if not triggered["done"]:
            triggered["done"] = True
            return True
        return False

    scheduler.run(circuit, monitor=monitor)
    assert planner.calls >= 2


def test_scheduler_matches_statevector_reference():
    """Ensure backend switching preserves the quantum state."""

    circuit = build_switch_circuit()

    # Ground truth using a plain statevector simulation
    ref = StatevectorBackend()
    ref.load(circuit.num_qubits)
    for gate in circuit.gates:
        ref.apply_gate(gate.gate, gate.qubits, gate.params)
    reference = ref.state.copy()

    # Run through the scheduler which will switch from Stim to Statevector
    scheduler = Scheduler()
    # Replace the decision diagram backend with a statevector simulator so we
    # can access the final amplitudes for verification.
    scheduler.backends[Backend.DECISION_DIAGRAM] = StatevectorBackend()
    scheduler.run(circuit)

    sim_state = scheduler.backends[Backend.DECISION_DIAGRAM].state
    assert np.allclose(sim_state, reference)
