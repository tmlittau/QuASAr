import unittest
import quasar_convert as qc

class PrimitiveHelperTests(unittest.TestCase):
    def setUp(self):
        self.eng = qc.ConversionEngine()

    def test_boundary_extraction(self):
        bridges = [(0, 5), (1, 6), (0, 7)]
        ssd = self.eng.extract_boundary_ssd(bridges, 2)
        self.assertEqual(sorted(ssd.boundary_qubits), [0, 1])
        self.assertEqual(ssd.top_s, 2)

    def test_local_window(self):
        state = [0j, 1+0j, 0j, 0j, 0j, 0j, 0j, 0j]
        window = self.eng.extract_local_window(state, [0, 1])
        self.assertEqual(len(window), 4)
        self.assertAlmostEqual(window[1], 1.0)

    def test_bridge_tensor(self):
        left = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        right = qc.SSD(boundary_qubits=[2], top_s=1)
        tensor = self.eng.build_bridge_tensor(left, right)
        self.assertEqual(len(tensor), 8)
        self.assertAlmostEqual(tensor[0], 1.0)

    def test_stabilizer_learner(self):
        if hasattr(self.eng, 'learn_stabilizer'):
            state = [0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j]
            tab = self.eng.learn_stabilizer(state)
            self.assertIsNotNone(tab)
            self.assertEqual(tab.num_qubits, 2)
        else:
            self.skipTest('Stim support not built')

if __name__ == '__main__':
    unittest.main()
