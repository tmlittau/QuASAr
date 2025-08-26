import unittest
from unittest import mock
import quasar_convert as qc

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.eng = qc.ConversionEngine()
        if hasattr(self.eng, "_ensure_impl"):
            # ensure underlying implementation exists so that we can spy on it
            self.eng._ensure_impl()

    def _spy(self, base_name: str):
        """Return a spy for the underlying implementation method."""
        if hasattr(self.eng, "_impl"):
            # Replace the native implementation with a wrapping mock so that we
            # can observe how often it is invoked.
            wrapped = mock.Mock(wraps=self.eng._impl)
            self.eng._impl = wrapped
            return getattr(wrapped, base_name)
        # Fall back to the pure Python stub helpers when the native extension
        # isn't present.
        attr = f"_{base_name}_impl"
        spy = mock.Mock(wraps=getattr(self.eng, attr))
        setattr(self.eng, attr, spy)
        return spy

    def test_extract_ssd_cached(self):
        spy = self._spy("extract_ssd")
        self.eng.extract_ssd([0, 1], 2)
        self.eng.extract_ssd([0, 1], 2)
        self.assertEqual(spy.call_count, 1)

    def test_extract_boundary_ssd_cached(self):
        bridges = [(0, 5), (1, 6)]
        spy = self._spy("extract_boundary_ssd")
        self.eng.extract_boundary_ssd(bridges, 2)
        # Different list instance but identical contents should hit cache
        self.eng.extract_boundary_ssd(list(bridges), 2)
        self.assertEqual(spy.call_count, 1)

    def test_build_bridge_tensor_cached(self):
        left = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        right = qc.SSD(boundary_qubits=[0, 1], top_s=2)
        spy = self._spy("build_bridge_tensor")
        self.eng.build_bridge_tensor(left, right)
        self.eng.build_bridge_tensor(left, right)
        self.assertEqual(spy.call_count, 1)

if __name__ == "__main__":
    unittest.main()
