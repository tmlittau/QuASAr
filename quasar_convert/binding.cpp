#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <cstdint>

#include "conversion_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quasar_convert, m) {
    py::class_<quasar::SSD>(m, "SSD")
        .def(py::init<>())
        .def_readwrite("boundary_qubits", &quasar::SSD::boundary_qubits)
        .def_readwrite("top_s", &quasar::SSD::top_s);

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

