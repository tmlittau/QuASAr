import pytest

from quasar.circuit import Circuit, Gate


def _sample_gates():
    return [
        Gate("H", [0]),
        Gate("CX", [0, 1]),
        Gate("MEASURE", [1]),
    ]


def test_flat_ssd_single_partition():
    circuit = Circuit(_sample_gates(), ssd_mode="flat")
    ssd = circuit.ssd

    assert len(ssd.partitions) == 1
    partition = ssd.partitions[0]
    assert partition.subsystems == ((0, 1),)
    assert partition.dependencies == ()
    assert partition.history == ("H", "CX", "MEASURE")


def test_flat_ssd_serialization_roundtrip():
    circuit = Circuit(_sample_gates(), ssd_mode="flat")
    data = circuit.to_dict()

    assert data["ssd_mode"] == "flat"

    restored = Circuit.from_dict(data)
    assert restored.ssd_mode == "flat"
    assert len(restored.ssd.partitions) == 1


def test_hierarchical_mode_remains_default():
    circuit = Circuit(_sample_gates())
    assert circuit.ssd_mode == "hierarchical"
    # Hierarchy information should be populated in default mode.
    assert circuit.ssd.hierarchy is not None


def test_invalid_ssd_mode_raises():
    with pytest.raises(ValueError):
        Circuit(_sample_gates(), ssd_mode="invalid")
