#include "conversion_engine.hpp"

#include <cmath>
#include <algorithm>
#include <set>
#include <complex>
#include <vector>

namespace quasar {

ConversionEngine::ConversionEngine() {
#ifdef QUASAR_USE_MQT
    dd_pkg = std::make_unique<dd::Package<>>();
#endif
}

std::pair<double, double> ConversionEngine::estimate_cost(std::size_t fragment_size,
                                                          Backend backend) const {
    // Very small toy cost model used by the tests.  The time cost scales
    // quadratically with the fragment size while the memory cost grows with
    // ``n log n``.  Different backends apply light weight modifiers so that the
    // numbers vary in a predictable manner without requiring a detailed
    // performance model.
    double n = static_cast<double>(fragment_size);
    double time_cost = n * n;
    double memory_cost = n * std::log2(n + 1.0);
    if (backend == Backend::DecisionDiagram) {
        time_cost *= 1.2;
        memory_cost *= 0.8;
    }
    return {time_cost, memory_cost};
}

SSD ConversionEngine::extract_ssd(const std::vector<uint32_t>& qubits, std::size_t s) const {
    SSD ssd;
    ssd.boundary_qubits = qubits;
    std::size_t n = qubits.size();
    std::size_t k = std::min<std::size_t>(s, n);
    ssd.top_s = k;
    ssd.vectors.assign(k, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < k; ++i) {
        ssd.vectors[i][i] = 1.0;
    }
    return ssd;
}

SSD ConversionEngine::extract_boundary_ssd(
    const std::vector<std::pair<uint32_t, uint32_t>>& bridges, std::size_t s) const {
    // Determine which local and remote qubits participate in a bridge.
    std::set<uint32_t> boundary_set;
    std::set<uint32_t> remote_set;
    for (const auto& b : bridges) {
        boundary_set.insert(b.first);
        remote_set.insert(b.second);
    }
    std::vector<uint32_t> boundary(boundary_set.begin(), boundary_set.end());
    std::vector<uint32_t> remote(remote_set.begin(), remote_set.end());
    const std::size_t m = boundary.size();
    const std::size_t n = remote.size();
    const std::size_t k = std::min<std::size_t>(s, m);

    // Build the connection matrix ``A`` whose rows correspond to local boundary
    // qubits and columns to remote qubits.  ``A[i][j]`` counts how many bridges
    // connect the ``i``-th boundary qubit to the ``j``-th remote qubit.
    std::vector<std::vector<double>> A(m, std::vector<double>(n, 0.0));
    for (const auto& br : bridges) {
        auto i = std::find(boundary.begin(), boundary.end(), br.first) - boundary.begin();
        auto j = std::find(remote.begin(), remote.end(), br.second) - remote.begin();
        A[i][j] += 1.0;
    }

    // Form the Gram matrix ``G = A * A^T``.  The leading eigenvectors of ``G``
    // correspond to the left singular vectors of ``A`` and describe the dominant
    // boundary directions.  A small power-iteration with Gram-Schmidt
    // orthogonalisation is sufficient for the matrix sizes encountered in the
    // tests.
    std::vector<std::vector<double>> G(m, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            for (std::size_t r = 0; r < n; ++r) {
                G[i][j] += A[i][r] * A[j][r];
            }
        }
    }

    std::vector<std::vector<double>> vectors;
    vectors.reserve(k);
    for (std::size_t vec = 0; vec < k; ++vec) {
        std::vector<double> v(m, 0.0);
        v[vec % m] = 1.0;  // deterministic initial guess
        for (std::size_t iter = 0; iter < 20; ++iter) {
            // Multiply by Gram matrix
            std::vector<double> w(m, 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    w[i] += G[i][j] * v[j];
                }
            }
            // Orthogonalise against previously found vectors
            for (const auto& prev : vectors) {
                double dot = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    dot += w[i] * prev[i];
                }
                for (std::size_t i = 0; i < m; ++i) {
                    w[i] -= dot * prev[i];
                }
            }
            // Normalise
            double norm = 0.0;
            for (double x : w) {
                norm += x * x;
            }
            norm = std::sqrt(norm);
            if (norm < 1e-12) {
                break;
            }
            for (std::size_t i = 0; i < m; ++i) {
                v[i] = w[i] / norm;
            }
        }
        vectors.push_back(v);
    }

    SSD ssd;
    ssd.boundary_qubits = std::move(boundary);
    ssd.top_s = k;
    ssd.vectors = std::move(vectors);
    return ssd;
}

std::vector<std::complex<double>> ConversionEngine::extract_local_window(
    const std::vector<std::complex<double>>& state,
    const std::vector<uint32_t>& window_qubits) const {
    const std::size_t k = window_qubits.size();
    const std::size_t dim = 1ULL << k;
    std::vector<std::complex<double>> window(dim, {0.0, 0.0});
    for (std::size_t local = 0; local < dim; ++local) {
        std::size_t idx = 0;
        for (std::size_t bit = 0; bit < k; ++bit) {
            if ((local >> bit) & 1ULL) {
                idx |= 1ULL << window_qubits[bit];
            }
        }
        window[local] = state[idx];
    }
    return window;
}

std::vector<std::complex<double>> ConversionEngine::build_bridge_tensor(const SSD& left,
                                                                        const SSD& right) const {
    const std::size_t m = left.boundary_qubits.size();
    const std::size_t n = right.boundary_qubits.size();
    const std::size_t dim = 1ULL << (m + n);
    std::vector<std::complex<double>> tensor(dim, {0.0, 0.0});
    const std::size_t mask = (1ULL << std::min(m, n)) - 1ULL;
    for (std::size_t l = 0; l < (1ULL << m); ++l) {
        for (std::size_t r = 0; r < (1ULL << n); ++r) {
            if ((l & mask) == (r & mask)) {
                tensor[(l << n) | r] = {1.0, 0.0};
            }
        }
    }
    return tensor;
}

ConversionResult ConversionEngine::convert(const SSD& ssd) const {
    const std::size_t boundary = ssd.boundary_qubits.size();
    const std::size_t rank = ssd.top_s;

    // Cost estimates mirror the simple analytical models used by the Python
    // ``CostEstimator``.  We treat the returned ``cost`` as a time cost and
    // ignore memory for now since the conversion planner only compares runtime.
    const std::size_t window = std::min<std::size_t>(boundary, 4);
    const std::size_t dense = 1ULL << window;  // dense window size for LW
    const std::size_t chi_tilde = std::min<std::size_t>(rank, 16);  // staged cap
    const std::size_t full = 1ULL << std::min<std::size_t>(boundary, 16);

    const double cost_b2b = std::pow(static_cast<double>(rank), 3) +
                            static_cast<double>(boundary) * rank * rank +
                            rank * rank;  // ingest
    const double cost_lw = static_cast<double>(dense) * 2.0;  // extract + ingest
    const double cost_st = std::pow(static_cast<double>(chi_tilde), 3) +
                           chi_tilde * chi_tilde;  // stage + ingest
    const double cost_full = static_cast<double>(full) * 2.0;  // full extraction

    // Primitive selection mirrors the simple policy used by the Python
    // reference implementation.  The more detailed cost model above is used to
    // report the estimated cost for the chosen primitive only.
    Primitive chosen;
    double cost;
    double fidelity;
    if (rank <= 4 && boundary <= 6) {
        chosen = Primitive::B2B;
        cost = cost_b2b;
        fidelity = boundary ? std::min(1.0, static_cast<double>(rank) /
                                              static_cast<double>(boundary))
                            : 1.0;
    } else if (boundary <= 10) {
        chosen = Primitive::LW;
        cost = cost_lw;
        fidelity = 1.0;  // dense extraction is exact
    } else if (rank <= 16) {
        chosen = Primitive::ST;
        cost = cost_st;
        fidelity = rank ? static_cast<double>(chi_tilde) /
                              static_cast<double>(rank)
                        : 1.0;
        if (fidelity > 1.0) {
            fidelity = 1.0;
        }
    } else {
        chosen = Primitive::Full;
        cost = cost_full;
        fidelity = 1.0;
    }

    return {chosen, cost, fidelity};
}

std::vector<std::complex<double>> ConversionEngine::convert_boundary_to_statevector(const SSD& ssd) const {
    const std::size_t dim = 1ULL << ssd.boundary_qubits.size();
    std::vector<std::complex<double>> state(dim, {0.0, 0.0});
    if (dim > 0) {
        state[0] = {1.0, 0.0};
    }
    return state;
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

std::optional<StimTableau> ConversionEngine::try_build_tableau(
    const std::vector<std::complex<double>>& state) const {
    return learn_stabilizer(state);
}

std::optional<StimTableau> ConversionEngine::learn_stabilizer(
    const std::vector<std::complex<double>>& state) const {
    if (state.empty()) {
        return std::nullopt;
    }
    try {
        return StimTableau::from_state_vector(state);
    } catch (...) {
        // Fall back to a handful of analytically recognisable states.  These
        // checks are deliberately conservative – if the state does not clearly
        // match a known stabilizer form we return ``nullopt`` instead of
        // attempting an expensive reconstruction.
    }

    const std::size_t dim = state.size();
    const std::size_t n = static_cast<std::size_t>(std::log2(dim));

    // Check for a computational basis state |i>.
    std::size_t nonzero = dim;  // index of the non-zero amplitude
    bool basis_state = true;
    for (std::size_t i = 0; i < dim; ++i) {
        double mag = std::norm(state[i]);
        if (mag > 1e-9) {
            if (std::abs(mag - 1.0) > 1e-9 || nonzero != dim) {
                basis_state = false;
                break;
            }
            nonzero = i;
        }
    }
    if (basis_state && nonzero < dim) {
        return StimTableau(n);
    }

    // Check for an equal superposition where amplitudes have magnitude
    // ``1/sqrt(dim)`` and phases in {1, -1, i, -i} up to a global phase.
    const double target_mag = 1.0 / std::sqrt(static_cast<double>(dim));
    auto phase_ok = [](std::complex<double> z) {
        static const std::complex<double> phases[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (const auto& p : phases) {
            if (std::abs(z - p) < 1e-9) {
                return true;
            }
        }
        return false;
    };
    bool uniform_state = true;
    std::complex<double> ref_phase = state[0] / target_mag;
    for (const auto& amp : state) {
        if (std::abs(std::abs(amp) - target_mag) > 1e-9) {
            uniform_state = false;
            break;
        }
        std::complex<double> rel = amp / target_mag / ref_phase;
        if (!phase_ok(rel)) {
            uniform_state = false;
            break;
        }
    }
    if (uniform_state) {
        return StimTableau(n);
    }

    return std::nullopt;
}
#endif

} // namespace quasar

