import unittest
from unittest import mock
import numpy as np
import quasar_convert as qc

class PrimitiveHelperTests(unittest.TestCase):
    def setUp(self):
        self.eng = qc.ConversionEngine()

    def test_ssd_vectors(self):
        ssd = self.eng.extract_ssd([0, 1, 2], 2)
        self.assertEqual(ssd.top_s, 2)
        self.assertEqual(ssd.vectors[0], [1.0, 0.0, 0.0])
        self.assertEqual(ssd.vectors[1], [0.0, 1.0, 0.0])

    def test_boundary_extraction(self):
        bridges = [(0, 5), (1, 6)]
        ssd = self.eng.extract_boundary_ssd(bridges, 2)
        self.assertEqual(ssd.boundary_qubits, [0, 1])
        self.assertEqual(ssd.top_s, 2)
        self.assertAlmostEqual(ssd.vectors[0][0], 1.0)
        self.assertAlmostEqual(ssd.vectors[1][1], 1.0)

    def test_local_window(self):
        state = [0j] * 8
        state[5] = 1.0 + 0j  # binary 101
        window = self.eng.extract_local_window(state, [2, 0])
        self.assertEqual(len(window), 4)
        self.assertAlmostEqual(window[3], 1.0)

    def test_bridge_tensor(self):
        left = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        right = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        tensor = self.eng.build_bridge_tensor(left, right)
        self.assertEqual(len(tensor), 16)
        # Check that only indices where left == right have amplitude 1
        for l in range(4):
            for r in range(4):
                amp = tensor[(l << 2) | r]
                if l == r:
                    self.assertAlmostEqual(amp, 1.0)
                else:
                    self.assertAlmostEqual(amp, 0.0)

    def test_stabilizer_learner(self):
        if hasattr(self.eng, 'learn_stabilizer'):
            state = [0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j]
            tab = self.eng.learn_stabilizer(state)
            self.assertIsNotNone(tab)
            self.assertEqual(tab.num_qubits, 2)
        else:
            self.skipTest('Stim support not built')

    def test_vecs2pauli_fallback(self):
        if hasattr(self.eng, 'learn_stabilizer'):
            state = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
            try:
                import stim
            except Exception:
                self.skipTest('Stim not available')
            with mock.patch('stim.Tableau.from_state_vector', side_effect=ValueError):
                tab = self.eng.learn_stabilizer(state)
            self.assertIsNotNone(tab)
            self.assertEqual(tab.num_qubits, 2)
        else:
            self.skipTest('Stim support not built')

if __name__ == '__main__':
    unittest.main()
