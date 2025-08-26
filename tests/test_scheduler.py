from quasar import Circuit, Scheduler, Planner, Backend, CostEstimator
from quasar_convert import ConversionEngine


class CountingConversionEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.bridges = 0

    def convert(self, ssd):
        self.calls += 1
        return super().convert(self.extract_ssd([], 0))

    def build_bridge_tensor(self, left, right):
        self.bridges += 1
        return super().build_bridge_tensor(left, right)


def build_switch_circuit():
    return Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
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


def build_cross_circuit():
    return Circuit([
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [2]},
        {"gate": "CX", "qubits": [1, 2]},
    ])


def test_scheduler_builds_bridge_for_cross_fragment_gate():
    coeff = {
        "b2b_svd": 0.0,
        "b2b_copy": 0.0,
        "ingest_dd": 0.0,
        "ingest_tab": 0.0,
        "ingest_sv": 0.0,
        "dd_gate": 10.0,
    }
    planner = Planner(CostEstimator(coeff))
    engine = CountingConversionEngine()
    scheduler = Scheduler(planner=planner, conversion_engine=engine)
    circuit = build_cross_circuit()
    scheduler.run(circuit)
    assert engine.bridges == 1
    sv_hist = scheduler.backends[Backend.STATEVECTOR].history
    tab_hist = scheduler.backends[Backend.TABLEAU].history
    assert "BRIDGE" in sv_hist
    assert "BRIDGE" in tab_hist
    assert sv_hist.count("CX") >= 1
    assert tab_hist.count("CX") >= 1
