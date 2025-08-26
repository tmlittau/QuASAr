from quasar import Circuit, Scheduler, Planner
from quasar_convert import ConversionEngine
import threading


class CountingConversionEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def convert(self, ssd):
        self.calls += 1
        return super().convert(self.extract_ssd([], 0))


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


def test_scheduler_uses_conversion_cache():
    engine = CountingConversionEngine()
    scheduler = Scheduler(conversion_engine=engine)
    circuit = build_switch_circuit()
    scheduler.run(circuit)
    assert engine.calls == 1
    scheduler.run(circuit)
    assert engine.calls == 1  # result reused from cache


def test_scheduler_dispatches_async():
    start = threading.Event()
    finish = threading.Event()

    class BlockingEngine(ConversionEngine):
        def __init__(self):
            super().__init__()

        def convert(self, ssd):
            start.set()
            finish.wait()
            return super().convert(self.extract_ssd([], 0))

    engine = BlockingEngine()
    scheduler = Scheduler(conversion_engine=engine)
    circuit = build_switch_circuit()

    t = threading.Thread(target=scheduler.run, args=(circuit,))
    t.start()
    assert start.wait(1)
    assert t.is_alive()
    finish.set()
    t.join(1)
    assert not t.is_alive()
