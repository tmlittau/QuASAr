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
        if hasattr(eng, 'convert_boundary_to_dd'):
            ssd = qc.SSD()
            ssd.boundary_qubits = [0, 1]
            ssd.top_s = 2
            edge = eng.convert_boundary_to_dd(ssd)
            self.assertIsNotNone(edge)
        else:
            self.skipTest('MQT DD support not built')

if __name__ == '__main__':
    unittest.main()
