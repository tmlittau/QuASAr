import unittest
import quasar_convert as qc

class OptionalBackendTests(unittest.TestCase):
    def test_stim_conversion(self):
        eng = qc.ConversionEngine()
        if hasattr(eng, 'convert_boundary_to_tableau'):
            ssd = qc.SSD()
            ssd.boundary_qubits = [0, 1]
            ssd.top_s = 2
            tab = eng.convert_boundary_to_tableau(ssd)
            self.assertEqual(tab.num_qubits, 2)
        else:
            self.skipTest('Stim support not built')

    def test_dd_conversion(self):
        eng = qc.ConversionEngine()
        if hasattr(eng, 'convert_boundary_to_dd') and hasattr(eng, 'dd_to_statevector'):
            ssd = qc.SSD()
            ssd.boundary_qubits = [0, 1]
            ssd.top_s = 2
            edge = eng.convert_boundary_to_dd(ssd)
            self.assertIsNotNone(edge)
            # Round-trip from decision diagram back to a concrete statevector.
            vec = eng.dd_to_statevector(*edge)
            self.assertEqual(len(vec), 4)
            self.assertAlmostEqual(vec[0], 1.0 + 0.0j)
            for amp in vec[1:]:
                self.assertAlmostEqual(abs(amp), 0.0)
        else:
            self.skipTest('MQT DD support not built')

    def test_dd_to_mps_conversion(self):
        eng = qc.ConversionEngine()
        if hasattr(eng, 'convert_boundary_to_dd') and hasattr(eng, 'dd_to_mps'):
            ssd = qc.SSD()
            ssd.boundary_qubits = [0, 1]
            ssd.top_s = 2
            edge = eng.convert_boundary_to_dd(ssd)
            tensors = eng.dd_to_mps(*edge)

            # Reconstruct the state from the returned MPS tensors.
            import numpy as np
            vec = np.array([[1.0 + 0.0j]])
            left = 1
            for t in tensors:
                arr = np.asarray(t, dtype=complex).reshape(left, 2, -1)
                vec = np.tensordot(vec, arr, axes=(1, 0)).reshape(-1, arr.shape[2])
                left = arr.shape[2]
            rec = vec.reshape(-1)

            # Expected state from the Python MPS backend (zero state).
            from quasar.backends import MPSBackend

            mps = MPSBackend()
            mps.load(2)
            expected = mps.statevector()
            np.testing.assert_allclose(rec, expected)
        else:
            self.skipTest('MQT DD support not built')

if __name__ == '__main__':
    unittest.main()
