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

#ifdef QUASAR_USE_STIM
using StimTableau = stim::Tableau<stim::MAX_BITWORD_WIDTH>;
#endif

// Minimal stabilizer tensor used by the STN prototype.  The tensor stores a
// dense amplitude vector along with an optional stabilizer tableau when Stim is
// available.  Downstream routines can use the tableau to perform log-depth
// contractions without materialising a dense representation.
struct StnTensor {
    std::vector<std::complex<double>> amplitudes;
#ifdef QUASAR_USE_STIM
    std::optional<StimTableau> tableau;
#endif
};

struct SSD {
    std::vector<uint32_t> boundary_qubits;  // indices of qubits on the boundary
    std::size_t top_s;                      // number of Schmidt vectors kept
    // Left singular vectors associated with the boundary decomposition.
    // Each entry has length equal to the number of boundary qubits.
    std::vector<std::vector<double>> vectors;
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
    double cost;           // estimated time cost
    double fidelity;       // crude fidelity estimate
};

class ConversionEngine {
  public:
    ConversionEngine();

    std::pair<double, double> estimate_cost(std::size_t fragment_size, Backend backend) const;

    SSD extract_ssd(const std::vector<uint32_t>& qubits, std::size_t s) const;

    // Extract the boundary directions induced by cross-fragment operations.
    // The input ``bridges`` lists gates that connect a qubit in the current
    // fragment (first element of the pair) with a qubit in another fragment
    // (second element).  A small SVD of the induced connection matrix yields a
    // Schmidt-style descriptor whose left singular vectors form the ``vectors``
    // entry of the returned SSD.
    SSD extract_boundary_ssd(const std::vector<std::pair<uint32_t, uint32_t>>& bridges,
                             std::size_t s) const;

    // Return a vector containing the amplitudes of a local window of qubits.
    // The window is specified by the indices in `window_qubits` and assumes all
    // other qubits are in the |0> state.  The returned vector has dimension
    // 2**len(window_qubits) and is ordered with qubit 0 as the least
    // significant bit.
    std::vector<std::complex<double>> extract_local_window(
        const std::vector<std::complex<double>>& state,
        const std::vector<uint32_t>& window_qubits) const;

    // Construct a simple bridge tensor that links two fragments described by
    // their SSD descriptors.  The tensor corresponds to an identity operation
    // on the overlapping boundary qubits and is returned as a flat amplitude
    // vector.
    std::vector<std::complex<double>> build_bridge_tensor(const SSD& left,
                                                          const SSD& right) const;

    // Choose a conversion primitive for the given SSD by estimating the cost of
    // each available strategy (B2B, LW, ST and Full) and returning the minimal
    // option.
    ConversionResult convert(const SSD& ssd) const;

    // Provide a concrete statevector representation for the boundary qubits.
    // A phase-factorable stabilizer state is synthesised from the leading
    // boundary directions so that downstream components can reinterpret the
    // vector as an STN tensor.  The dense representation is always returned.
    std::vector<std::complex<double>> convert_boundary_to_statevector(const SSD& ssd) const;

    // Build a stabilizer tensor network component from the given boundary.
    // The amplitude vector is always populated while the optional tableau is
    // provided when the boundary admits a stabilizer description.
    StnTensor convert_boundary_to_stn(const SSD& ssd) const;

#ifdef QUASAR_USE_MQT
    // The decision diagram package exposes `vEdge` at the namespace level,
    // so we use it directly instead of the previous `Package<>::vEdge` alias.
    dd::vEdge convert_boundary_to_dd(const SSD& ssd) const;

    // Export the amplitudes represented by a decision diagram edge as a
    // normalised statevector.  The returned vector has dimension ``2^n`` where
    // ``n`` is inferred from the edge's variable index.
    std::vector<std::complex<double>> dd_to_statevector(const dd::vEdge& edge) const;

    // Factor the amplitudes represented by a decision diagram edge into a
    // matrix product state.  The returned chain stores each tensor as a flat
    // vector with dimensions ``(left, 2, right)``.  When ``chi`` is non-zero the
    // intermediate bond dimensions are truncated to ``chi``.
    std::vector<std::vector<std::complex<double>>> dd_to_mps(const dd::vEdge& edge,
                                                            std::size_t chi = 0) const;
#endif

#if defined(QUASAR_USE_MQT) && defined(QUASAR_USE_STIM)
    // Convert a stabilizer tableau into a decision diagram representation by
    // first materialising the corresponding statevector and then ingesting it
    // into the DD package.
    dd::vEdge tableau_to_dd(const StimTableau& tableau) const;
#endif

#ifdef QUASAR_USE_STIM
    StimTableau convert_boundary_to_tableau(const SSD& ssd) const;
    // Convert a stabilizer tableau into a dense statevector.  The returned
    // vector has dimension ``2^n`` with qubit 0 as the least significant bit.
    std::vector<std::complex<double>> tableau_to_statevector(const StimTableau& tableau) const;
    std::optional<StimTableau> try_build_tableau(const std::vector<std::complex<double>>& state) const;
    // Attempt to learn a stabilizer tableau from an arbitrary state vector.
    // Simple analytic checks recognise computational basis states and uniform
    // superpositions with phases in {±1, ±i}.  Returns ``std::nullopt`` if the
    // state does not appear to be a stabilizer state under these checks.
    std::optional<StimTableau> learn_stabilizer(
        const std::vector<std::complex<double>>& state) const;
#endif

  private:
#ifdef QUASAR_USE_MQT
    std::unique_ptr<dd::Package<>> dd_pkg;
#endif
};

} // namespace quasar

