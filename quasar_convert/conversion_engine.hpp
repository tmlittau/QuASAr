#pragma once

#include <optional>
#include <utility>
#include <vector>
#include <complex>
#include <memory>
#include <cstddef>
#include <cstdint>

#ifdef QUASAR_USE_STIM
#include <stim.h>
#endif
#ifdef QUASAR_USE_MQT
#include <dd/Package.hpp>
#endif

namespace quasar {

struct SSD {
    std::vector<uint32_t> boundary_qubits;  // indices of qubits on the boundary
    std::size_t top_s;                      // number of Schmidt vectors kept
};

enum class Backend {
    StimTableau,
    DecisionDiagram
};

// Conversion primitive selected by the engine. These correspond to the
// strategies described in Table 2 of the QuASAr draft: boundary-to-boundary
// (B2B), local-window (LW), staged (ST) and full extraction (Full).
enum class Primitive {
    B2B,
    LW,
    ST,
    Full
};

struct ConversionResult {
    Primitive primitive;   // primitive that was chosen
    double cost;           // simplistic cost measure
};

class ConversionEngine {
  public:
    ConversionEngine();

    std::pair<double, double> estimate_cost(std::size_t fragment_size, Backend backend) const;

    SSD extract_ssd(const std::vector<uint32_t>& qubits, std::size_t s) const;

    // Choose a conversion primitive for the given SSD and simulate the
    // associated cost. The implementation uses simple heuristics based on the
    // boundary size and Schmidt rank to select between B2B, LW, ST and Full.
    ConversionResult convert(const SSD& ssd) const;

#ifdef QUASAR_USE_MQT
    dd::Package<>::vEdge convert_boundary_to_dd(const SSD& ssd) const;
#endif

#ifdef QUASAR_USE_STIM
    stim::Tableau convert_boundary_to_tableau(const SSD& ssd) const;
    std::optional<stim::Tableau> try_build_tableau(const std::vector<std::complex<double>>& state) const;
#endif

  private:
#ifdef QUASAR_USE_MQT
    std::unique_ptr<dd::Package<>> dd_pkg;
#endif
};

} // namespace quasar

