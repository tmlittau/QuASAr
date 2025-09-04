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

    def test_lw_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(8))
        ssd.top_s = 2
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.LW)

    def test_lw_gate_counts_increase_cost(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(8))
        ssd.top_s = 2
        base = self.eng.convert(ssd)
        with_gates = self.eng.convert(ssd, window_1q_gates=3, window_2q_gates=1)
        self.assertGreater(with_gates.cost, base.cost)

    def test_st_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(20))
        ssd.top_s = 8
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.ST)

    def test_full_selected(self):
        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(20))
        ssd.top_s = 32
        res = self.eng.convert(ssd)
        self.assertEqual(res.primitive, qc.Primitive.Full)

if __name__ == '__main__':
    unittest.main()
