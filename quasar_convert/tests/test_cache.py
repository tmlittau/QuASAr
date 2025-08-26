import unittest
from unittest import mock
import quasar_convert as qc

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.eng = qc.ConversionEngine()
        if hasattr(self.eng, "_ensure_impl"):
            # ensure underlying implementation exists for patching
            self.eng._ensure_impl()

    def _patch(self, base_name: str):
        if hasattr(self.eng, "_impl"):
            target = self.eng._impl
            attr = base_name
        else:
            target = self.eng
            attr = f"_{base_name}_impl"
        return mock.patch.object(target, attr, wraps=getattr(target, attr))

    def test_extract_ssd_cached(self):
        with self._patch("extract_ssd") as spy:
            self.eng.extract_ssd([0, 1], 2)
            self.eng.extract_ssd([0, 1], 2)
            self.assertEqual(spy.call_count, 1)

    def test_extract_boundary_ssd_cached(self):
        bridges = [(0, 5), (1, 6)]
        with self._patch("extract_boundary_ssd") as spy:
            self.eng.extract_boundary_ssd(bridges, 2)
            # Different list instance but identical contents should hit cache
            self.eng.extract_boundary_ssd(list(bridges), 2)
            self.assertEqual(spy.call_count, 1)

    def test_build_bridge_tensor_cached(self):
        left = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        right = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        with self._patch("build_bridge_tensor") as spy:
            self.eng.build_bridge_tensor(left, right)
            self.eng.build_bridge_tensor(left, right)
            self.assertEqual(spy.call_count, 1)

if __name__ == "__main__":
    unittest.main()
