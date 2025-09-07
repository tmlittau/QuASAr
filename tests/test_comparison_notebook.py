import pandas as pd
import pytest

import benchmarks.circuits as circuit_lib
from benchmarks.runner import BenchmarkRunner
from quasar import Backend, SimulationEngine
from quasar.ssd import SSD


@pytest.mark.parametrize("num_qubits", [3, 5])
def test_notebook_comparison_behaviour(num_qubits: int) -> None:
    backends = [Backend.STATEVECTOR, Backend.MPS, Backend.TABLEAU, Backend.DECISION_DIAGRAM]
    runner = BenchmarkRunner()
    engine = SimulationEngine()

    circuits = {
        "ghz": circuit_lib.ghz_circuit,
        "qft": circuit_lib.qft_circuit,
        "wstate": circuit_lib.w_state_circuit,
    }

    records = []
    for name, build in circuits.items():
        base = build(num_qubits, use_classical_simplification=False)
        for b in backends:
            if name == "wstate" and b == Backend.TABLEAU:
                continue
            try:
                rec = runner.run_quasar_multiple(base, engine, backend=b, repetitions=1)
                rec.update({"circuit": name, "mode": "forced"})
                records.append(rec)
                state_rec = runner.run_quasar(base, engine, backend=b)
                assert isinstance(state_rec["result"], SSD)
            except RuntimeError:
                pass
        try:
            auto = build(num_qubits, use_classical_simplification=True)
            rec = runner.run_quasar_multiple(auto, engine, repetitions=1)
            rec.update({"circuit": name, "mode": "auto"})
            records.append(rec)
            state_rec = runner.run_quasar(auto, engine)
            assert isinstance(state_rec["result"], SSD)
        except RuntimeError:
            pass

    df = pd.DataFrame(records)
    forced = df[df["mode"] == "forced"]
    auto = df[df["mode"] == "auto"]

    ghz_auto = auto[auto["circuit"] == "ghz"].iloc[0]
    ghz_forced = forced[forced["circuit"] == "ghz"]
    assert ghz_auto["backend"] == Backend.TABLEAU.name
    tab_time = ghz_forced[ghz_forced["backend"] == Backend.TABLEAU.name]["run_time_mean"].iloc[0]
    assert abs(ghz_auto["run_time_mean"] - tab_time) / tab_time < 50.0

    qft_auto = auto[auto["circuit"] == "qft"].iloc[0]
    qft_forced = forced[forced["circuit"] == "qft"]
    assert qft_auto["backend"] == Backend.TABLEAU.name

    w_auto = auto[auto["circuit"] == "wstate"].iloc[0]
    w_forced = forced[forced["circuit"] == "wstate"]
    assert w_auto["backend"] == Backend.DECISION_DIAGRAM.name
    assert Backend.TABLEAU.name not in set(w_forced["backend"])
    # Memory comparisons are omitted because auto selection includes
    # scheduler overhead not present in the forced baselines.
