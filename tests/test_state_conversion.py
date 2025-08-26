from quasar.backends import StatevectorBackend, StimBackend, DecisionDiagramBackend
from quasar_convert import ConversionEngine


def test_conversion_engine_outputs_ingestible_states():
    eng = ConversionEngine()
    ssd = eng.extract_ssd([0, 1], 2)

    vec = eng.convert_boundary_to_statevector(ssd)
    sv = StatevectorBackend()
    sv.ingest(vec)
    assert sv.num_qubits == 2

    tab = eng.convert_boundary_to_tableau(ssd)
    stim = StimBackend()
    stim.ingest(tab)
    assert stim.num_qubits == 2


def test_conversion_to_decision_diagram_backend():
    eng = ConversionEngine()
    ssd = eng.extract_ssd([0], 1)
    dd_state = eng.convert_boundary_to_dd(ssd)
    dd = DecisionDiagramBackend()
    dd.ingest(dd_state)
    assert dd.num_qubits == 1
