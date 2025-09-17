import numpy as np
import pytest

from quasar.backends import DecisionDiagramBackend
from quasar.scheduler import Scheduler
from quasar_convert import ConversionEngine

mqt_core = pytest.importorskip("mqt.core")


def test_clone_preserves_decision_diagram_state():
    dd = mqt_core.dd
    original = DecisionDiagramBackend()
    original.load(1)
    original.apply_gate("X", [0])
    extracted = original.extract_ssd()

    engine = ConversionEngine()
    scheduler = Scheduler(conversion_engine=engine)

    target = DecisionDiagramBackend()
    target.load(1)

    clone = scheduler._clone_decision_diagram_state(extracted, target)
    assert clone is not None
    assert isinstance(clone[1], dd.VectorDD)

    target.ingest(clone, num_qubits=1)
    state = target.statevector()
    np.testing.assert_allclose(state, np.array([0.0, 1.0], dtype=complex))
