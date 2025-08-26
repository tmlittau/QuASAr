#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <cstdint>

#include "conversion_engine.hpp"

namespace py = pybind11;

// The compiled module lives alongside the Python package as
// ``quasar_convert._conversion_engine``.  A small Python stub in
// ``__init__`` falls back to a pure Python implementation when the native
// extension is unavailable.  Expose the C++ API using pybind11.
PYBIND11_MODULE(_conversion_engine, m) {
    py::class_<quasar::SSD>(m, "SSD")
        .def(py::init([](std::vector<uint32_t> boundary_qubits,
                         std::size_t top_s,
                         std::vector<std::vector<double>> vectors) {
                 quasar::SSD s;
                 s.boundary_qubits = std::move(boundary_qubits);
                 s.top_s = top_s;
                 s.vectors = std::move(vectors);
                 return s;
             }),
             py::arg("boundary_qubits") = std::vector<uint32_t>{},
             py::arg("top_s") = 0,
             py::arg("vectors") = std::vector<std::vector<double>>{})
        .def_readwrite("boundary_qubits", &quasar::SSD::boundary_qubits)
        .def_readwrite("top_s", &quasar::SSD::top_s)
        .def_readwrite("vectors", &quasar::SSD::vectors);

#ifdef QUASAR_USE_STIM
    py::class_<quasar::StimTableau>(m, "StimTableau")
        .def(py::init<size_t>())
        .def_readwrite("num_qubits", &quasar::StimTableau::num_qubits);
#endif

    py::enum_<quasar::Backend>(m, "Backend")
        .value("StimTableau", quasar::Backend::StimTableau)
        .value("DecisionDiagram", quasar::Backend::DecisionDiagram);

    py::enum_<quasar::Primitive>(m, "Primitive")
        .value("B2B", quasar::Primitive::B2B)
        .value("LW", quasar::Primitive::LW)
        .value("ST", quasar::Primitive::ST)
        .value("Full", quasar::Primitive::Full);

    py::class_<quasar::ConversionResult>(m, "ConversionResult")
        .def_readonly("primitive", &quasar::ConversionResult::primitive)
        .def_readonly("cost", &quasar::ConversionResult::cost);

    py::class_<quasar::ConversionEngine>(m, "ConversionEngine")
        .def(py::init<>())
        .def("estimate_cost", &quasar::ConversionEngine::estimate_cost)
        .def("extract_ssd", &quasar::ConversionEngine::extract_ssd)
        .def("extract_boundary_ssd", &quasar::ConversionEngine::extract_boundary_ssd)
        .def("extract_local_window", &quasar::ConversionEngine::extract_local_window)
        .def("convert", &quasar::ConversionEngine::convert)
        .def("build_bridge_tensor", &quasar::ConversionEngine::build_bridge_tensor)
#ifdef QUASAR_USE_STIM
        .def("convert_boundary_to_tableau", &quasar::ConversionEngine::convert_boundary_to_tableau)
        .def("try_build_tableau", &quasar::ConversionEngine::try_build_tableau)
        .def("learn_stabilizer", &quasar::ConversionEngine::learn_stabilizer)
#endif
#ifdef QUASAR_USE_MQT
        .def("convert_boundary_to_dd", [](quasar::ConversionEngine& eng, const quasar::SSD& ssd) {
            // Return the raw pointer of the decision diagram edge as an integer
            // handle. This avoids binding the full dd::vEdge type while still
            // allowing callers to verify that a non-null edge was produced.
            auto edge = eng.convert_boundary_to_dd(ssd);
            return reinterpret_cast<std::uintptr_t>(edge.p);
        })
#endif
        ;
}

