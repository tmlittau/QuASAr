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

    def test_dd_to_mps_sparse_path(self):
        eng = qc.ConversionEngine()
        required = all(
            hasattr(eng, attr)
            for attr in ('convert_boundary_to_dd', 'dd_to_mps', 'dense_statevector_queries')
        )
        if not required:
            self.skipTest('MQT DD support not built')

        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(6))
        ssd.top_s = 6
        edge = eng.convert_boundary_to_dd(ssd)
        before = eng.dense_statevector_queries()
        tensors = eng.dd_to_mps(*edge)
        after = eng.dense_statevector_queries()

        self.assertEqual(after, before, 'dd_to_mps should avoid dense exports')
        self.assertEqual(len(tensors), len(ssd.boundary_qubits))

    def test_dd_to_tableau_sparse_extraction(self):
        eng = qc.ConversionEngine()
        required = all(
            hasattr(eng, attr)
            for attr in (
                'convert_boundary_to_dd',
                'dd_to_tableau',
                'dense_statevector_queries',
            )
        ) and hasattr(qc, 'StimTableau')
        if not required:
            self.skipTest('Stim or MQT support not built')

        ssd = qc.SSD()
        ssd.boundary_qubits = list(range(4))
        ssd.top_s = 4
        edge = eng.convert_boundary_to_dd(ssd)
        before = eng.dense_statevector_queries()
        tableau = eng.dd_to_tableau(*edge)
        after = eng.dense_statevector_queries()

        self.assertIsNotNone(tableau)
        self.assertEqual(tableau.num_qubits, len(ssd.boundary_qubits))
        self.assertEqual(after, before, 'dd_to_tableau should avoid dense exports')

    def test_tableau_to_statevector(self):
        eng = qc.ConversionEngine()
        if hasattr(eng, 'tableau_to_statevector') and hasattr(qc, 'StimTableau'):
            import stim
            import numpy as np
            circuit_text = 'H 0\nCX 0 1\nS 1\n'
            tab = qc.StimTableau.from_circuit(circuit_text)
            vec_cpp = eng.tableau_to_statevector(tab)
            vec_stim = stim.Tableau.from_circuit(stim.Circuit(circuit_text)).to_state_vector(endian='little')
            np.testing.assert_allclose(vec_cpp, vec_stim, atol=1e-6)
        else:
            self.skipTest('Stim support not built')

    def test_tableau_to_dd_roundtrip(self):
        eng = qc.ConversionEngine()
        if all(hasattr(eng, attr) for attr in ('tableau_to_dd', 'dd_to_statevector', 'tableau_to_statevector')) and hasattr(qc, 'StimTableau'):
            import numpy as np
            circuit_text = 'H 0\nCX 0 1\nS 1\n'
            tab = qc.StimTableau.from_circuit(circuit_text)
            edge = eng.tableau_to_dd(tab)
            vec_dd = eng.dd_to_statevector(*edge)
            vec_tab = eng.tableau_to_statevector(tab)
            np.testing.assert_allclose(vec_dd, vec_tab, atol=1e-6)
        else:
            self.skipTest('Stim or MQT support not built')

    def test_tableau_to_mps_reconstruction(self):
        eng = qc.ConversionEngine()
        if hasattr(eng, 'tableau_to_mps') and hasattr(qc, 'StimTableau'):
            import stim
            import numpy as np

            def reconstruct(tensors):
                vec = np.array([[1.0 + 0.0j]])
                left = 1
                for t in tensors:
                    arr = np.asarray(t, dtype=complex).reshape(left, 2, -1)
                    vec = np.tensordot(vec, arr, axes=(1, 0)).reshape(-1, arr.shape[2])
                    left = arr.shape[2]
                return vec.reshape(-1)

            # Bell state |00> + |11>
            bell = 'H 0\nCX 0 1\n'
            tab_bell = qc.StimTableau.from_circuit(bell)
            tensors = eng.tableau_to_mps(tab_bell)
            rec = reconstruct(tensors)
            expected = stim.Tableau.from_circuit(stim.Circuit(bell)).to_state_vector(endian='little')
            np.testing.assert_allclose(rec, expected, atol=1e-6)

            # Three-qubit GHZ state
            ghz = 'H 0\nCX 0 1\nCX 0 2\n'
            tab_ghz = qc.StimTableau.from_circuit(ghz)
            tensors = eng.tableau_to_mps(tab_ghz)
            rec = reconstruct(tensors)
            expected = stim.Tableau.from_circuit(stim.Circuit(ghz)).to_state_vector(endian='little')
            np.testing.assert_allclose(rec, expected, atol=1e-6)
        else:
            self.skipTest('Stim support not built')

if __name__ == '__main__':
    unittest.main()
