from benchmarks.circuits import ghz_circuit
from quasar import Backend
from quasar.scheduler import Scheduler


def test_backend_selection_logging(tmp_path):
    log_file = tmp_path / "sel.csv"
    sched = Scheduler(
        quick_max_qubits=10,
        quick_max_gates=100,
        quick_max_depth=10,
        backend_selection_log=str(log_file),
    )
    circuit = ghz_circuit(3)
    backend = sched.select_backend(circuit)
    assert backend == Backend.TABLEAU
    content = log_file.read_text().strip().split(",")
    assert content[3] == "TABLEAU"
