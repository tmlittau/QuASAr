#include "conversion_engine.hpp"

#include <cmath>
#include <algorithm>
#include <complex>
#include <vector>

namespace quasar {

ConversionEngine::ConversionEngine() {
#ifdef QUASAR_USE_MQT
    dd_pkg = std::make_unique<dd::Package<>>();
#endif
}

std::pair<double, double> ConversionEngine::estimate_cost(std::size_t fragment_size, Backend backend) const {
    // Simple placeholder model: cost grows linearly with fragment size.
    double time_cost = static_cast<double>(fragment_size);
    double memory_cost = static_cast<double>(fragment_size) * 0.1;
    if (backend == Backend::DecisionDiagram) {
        time_cost *= 1.5;  // assume DD conversion is slightly more expensive
    }
    return {time_cost, memory_cost};
}

SSD ConversionEngine::extract_ssd(const std::vector<uint32_t>& qubits, std::size_t s) const {
    SSD ssd;
    ssd.boundary_qubits = qubits;
    ssd.top_s = s;
    return ssd;
}

ConversionResult ConversionEngine::convert(const SSD& ssd) const {
    const std::size_t boundary = ssd.boundary_qubits.size();
    const std::size_t rank = ssd.top_s;

    Primitive chosen;
    double cost = 0.0;

    // Heuristics inspired by the draft:
    // - B2B for small rank and small boundary.
    // - LW for larger boundaries but still within a manageable window.
    // - ST for moderate/large rank where approximation is attempted.
    // - Full as a last resort.
    if (rank <= 4 && boundary <= 6) {
        chosen = Primitive::B2B;
        // Simulate cubic cost with nested loops.
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j < rank; ++j) {
                for (std::size_t k = 0; k < rank; ++k) {
                    cost += static_cast<double>((i + j + k) % 5);
                }
            }
        }
    } else if (boundary <= 10) {
        chosen = Primitive::LW;
        const std::size_t w = std::min<std::size_t>(boundary, 4);
        const std::size_t dim = 1ULL << w;
        std::vector<std::complex<double>> window(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            window[i] = std::complex<double>(0.0, 0.0);
        }
        cost = static_cast<double>(dim);
    } else if (rank <= 16) {
        chosen = Primitive::ST;
        const std::size_t chi = std::min<std::size_t>(rank, 8);
        for (std::size_t i = 0; i < chi; ++i) {
            for (std::size_t j = 0; j < chi; ++j) {
                for (std::size_t k = 0; k < chi; ++k) {
                    cost += static_cast<double>((i * j + k) % 7);
                }
            }
        }
    } else {
        chosen = Primitive::Full;
        const std::size_t dim = 1ULL << std::min<std::size_t>(boundary, 16);
        std::vector<std::complex<double>> state(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            state[i] = std::complex<double>(0.0, 0.0);
        }
        cost = static_cast<double>(dim);
    }

    return {chosen, cost};
}

#ifdef QUASAR_USE_MQT
dd::vEdge ConversionEngine::convert_boundary_to_dd(const SSD& ssd) const {
    // Produce a zero-state decision diagram for the boundary qubits.
    // The MQT Core package expects the number of qubits as a standard size type.
    // Earlier versions used dd::QubitCount, but this alias is no longer exposed
    // in recent releases. Using std::size_t keeps the code compatible across
    // versions without introducing a direct dependency on an internal typedef.
    return dd_pkg->makeZeroState(ssd.boundary_qubits.size());
}
#endif

#ifdef QUASAR_USE_STIM
StimTableau ConversionEngine::convert_boundary_to_tableau(const SSD& ssd) const {
    // Return an identity tableau of the requested size.
    return StimTableau(ssd.boundary_qubits.size());
}

std::optional<StimTableau> ConversionEngine::try_build_tableau(const std::vector<std::complex<double>>& state) const {
    if (state.empty()) {
        return std::nullopt;
    }
    // Check whether the state is |0...0>.
    bool zero_state = std::abs(state[0] - std::complex<double>(1.0, 0.0)) < 1e-9;
    for (std::size_t i = 1; i < state.size() && zero_state; ++i) {
        if (std::abs(state[i]) > 1e-9) {
            zero_state = false;
        }
    }
    if (zero_state) {
        std::size_t n = static_cast<std::size_t>(std::log2(state.size()));
        return StimTableau(n);
    }
    return std::nullopt;
}
#endif

} // namespace quasar

