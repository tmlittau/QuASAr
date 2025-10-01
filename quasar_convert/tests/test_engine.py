import math
import unittest

import quasar_convert as qc

class ConversionPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.eng = qc.ConversionEngine()

    def test_b2b_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = [0, 1]
        ssd.top_s = 2
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.B2B)
        self.assertGreater(res.cost, 0)
        self.assertIsNone(res.window)

    def test_lw_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(8))
        ssd.top_s = 2
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.LW)
        self.assertEqual(res.window, 4)

    def test_lw_gate_counts_increase_cost(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(8))
        ssd.top_s = 2
        base = self.eng.convert(ssd)
        with_gates = self.eng.convert(ssd, window_1q_gates=3, window_2q_gates=1)
        self.assertGreater(with_gates.cost, base.cost)
        self.assertEqual(base.window, 4)
        self.assertEqual(with_gates.window, 4)

    def test_window_override_expands_dense_region(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(8))
        ssd.top_s = 2
        default = self.eng.convert(ssd)
        widened = self.eng.convert(ssd, window=6)
        self.assertGreater(widened.cost, default.cost)
        self.assertEqual(default.window, 4)
        self.assertEqual(widened.window, 6)

    def test_st_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(20))
        ssd.top_s = 8
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.ST)
        self.assertIsNone(res.window)

    def test_full_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(20))
        ssd.top_s = 32
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.Full)
        self.assertIsNone(res.window)

    def test_boundary_truncation_records_stats(self):
        eng = qc.ConversionEngine(truncation_tolerance=0.6)
        ssd = qc.SSD(boundary_qubits=[0, 1], top_s=2, vectors=[[1.0, -1.0]])
        state = eng.convert_boundary_to_statevector(ssd)
        # Only a single amplitude should remain after aggressive thresholding.
        non_zero = [amp for amp in state if abs(amp) > 1e-9]
        self.assertEqual(len(non_zero), 1)
        stats = eng.last_compression_stats()
        self.assertEqual(stats.retained_terms, 1)
        # Fidelity corresponds to the retained weight (1/4 for a 2-qubit uniform state).
        self.assertAlmostEqual(stats.fidelity, 0.25, places=6)
        self.assertEqual(eng.compressed_cardinality(), 1)

    def test_local_window_max_terms_truncates(self):
        eng = qc.ConversionEngine(truncation_max_terms=2)
        state = [1.0 + 0j, 0.1 + 0j, 0.2 + 0j, 0.05 + 0j]
        window = eng.extract_local_window(state, [0, 1])
        stats = eng.last_compression_stats()
        self.assertEqual(stats.retained_terms, 2)
        retained_indices = [idx for idx, amp in enumerate(window) if abs(amp) > 1e-9]
        self.assertEqual(sorted(retained_indices), [0, 2])
        total_norm = sum(abs(val) ** 2 for val in state)
        retained_norm = abs(state[0]) ** 2 + abs(state[2]) ** 2
        self.assertAlmostEqual(stats.fidelity, retained_norm / total_norm, places=7)
        scale = math.sqrt(total_norm / retained_norm)
        self.assertAlmostEqual(window[0], state[0] * scale)
        self.assertAlmostEqual(window[2], state[2] * scale)

if __name__ == '__main__':
    unittest.main()
