#include "conversion_engine.hpp"

#include <cmath>
#include <algorithm>
#include <set>
#include <complex>
#include <vector>
#include <random>
#ifdef QUASAR_USE_MQT
#include <dd/StateGeneration.hpp>
#endif

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
    const std::size_t chi_tilde = std::min<std::size_t>(rank, st_chi_cap);  // staged cap
    const std::size_t full = 1ULL << std::min<std::size_t>(boundary, 16);

    const double svd_cost = std::min<double>(boundary * rank * rank,
                                             rank * boundary * boundary);
    const double cost_b2b = svd_cost + static_cast<double>(boundary) * rank * rank +
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
    const std::size_t n = ssd.boundary_qubits.size();
    const std::size_t dim = 1ULL << n;
    std::vector<std::complex<double>> state(dim, {0.0, 0.0});
    if (!dim) {
        return state;
    }
    const double norm = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<std::complex<double>> phases(n, {1.0, 0.0});
    if (!ssd.vectors.empty()) {
        const auto& vec = ssd.vectors[0];
        for (std::size_t i = 0; i < n && i < vec.size(); ++i) {
            if (vec[i] < 0) {
                phases[i] = {-1.0, 0.0};
            }
        }
    }
    for (std::size_t idx = 0; idx < dim; ++idx) {
        std::complex<double> amp{1.0, 0.0};
        for (std::size_t bit = 0; bit < n; ++bit) {
            if ((idx >> bit) & 1ULL) {
                amp *= phases[bit];
            }
        }
        state[idx] = amp * norm;
    }
    return state;
}

StnTensor ConversionEngine::convert_boundary_to_stn(const SSD& ssd) const {
    StnTensor tensor;
    tensor.amplitudes = convert_boundary_to_statevector(ssd);
#ifdef QUASAR_USE_STIM
    tensor.tableau = learn_stabilizer(tensor.amplitudes);
#endif
    return tensor;
}

std::vector<std::complex<double>>
ConversionEngine::mps_to_statevector(const MPS& mps) const {
    const auto& tensors = mps.tensors;
    if (tensors.empty()) {
        return {};
    }
    const std::size_t n = tensors.size();

    // Determine bond dimensions.  When the caller does not provide them they
    // are inferred from the tensor sizes by sweeping from left to right.
    std::vector<std::size_t> bond_dims = mps.bond_dims;
    if (bond_dims.size() != n + 1) {
        bond_dims.assign(n + 1, 0);
        bond_dims[0] = 1;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t left = bond_dims[i];
            std::size_t right = tensors[i].size() / (left * 2);
            bond_dims[i + 1] = right;
        }
    }

    // Initialise the running matrix with the first tensor contracted over the
    // left boundary which is assumed to have dimension one.
    const std::size_t first_chi = bond_dims[1];
    std::vector<std::complex<double>> current(2 * first_chi);
    for (std::size_t p = 0; p < 2; ++p) {
        for (std::size_t r = 0; r < first_chi; ++r) {
            current[p * first_chi + r] = tensors[0][p * first_chi + r];
        }
    }

    std::size_t left_dim = 2;  // 2^1 after absorbing the first tensor
    for (std::size_t qubit = 1; qubit < n; ++qubit) {
        const std::size_t bond = bond_dims[qubit];
        const std::size_t next_bond = bond_dims[qubit + 1];
        const auto& tensor = tensors[qubit];

        std::vector<std::complex<double>> next(left_dim * 2 * next_bond,
                                               {0.0, 0.0});
        for (std::size_t i = 0; i < left_dim; ++i) {
            for (std::size_t k = 0; k < bond; ++k) {
                std::complex<double> coeff = current[i * bond + k];
                if (coeff == std::complex<double>{0.0, 0.0}) {
                    continue;
                }
                for (std::size_t p = 0; p < 2; ++p) {
                    for (std::size_t r = 0; r < next_bond; ++r) {
                        next[(i * 2 + p) * next_bond + r] +=
                            coeff * tensor[(k * 2 + p) * next_bond + r];
                    }
                }
            }
        }
        current.swap(next);
        left_dim *= 2;
    }

    const std::size_t final_bond = bond_dims[n];
    std::vector<std::complex<double>> state(left_dim);
    for (std::size_t i = 0; i < left_dim; ++i) {
        state[i] = current[i * final_bond];
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

std::vector<std::complex<double>>
ConversionEngine::dd_to_statevector(const dd::vEdge& edge) const {
    // Use the decision diagram package to export the amplitudes represented by
    // ``edge``.  ``getVector`` returns a flat vector ordered with qubit 0 as the
    // least significant bit.  Normalise the result to guard against numerical
    // imprecision in the underlying DD representation.
    auto vec = dd_pkg->getVector(edge);
    double norm = 0.0;
    for (const auto& amp : vec) {
        norm += std::norm(amp);
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (auto& amp : vec) {
            amp /= norm;
        }
    }
    return vec;
}

std::vector<std::vector<std::complex<double>>>
ConversionEngine::dd_to_mps(const dd::vEdge& edge, std::size_t chi) const {
    auto state = dd_to_statevector(edge);
    return statevector_to_mps(state, chi);
}
#endif

#if defined(QUASAR_USE_MQT) && defined(QUASAR_USE_STIM)
dd::vEdge ConversionEngine::tableau_to_dd(const StimTableau& tableau) const {
    // Produce a dense statevector for the tableau and construct a decision
    // diagram representing the same amplitudes.
    auto vec = tableau_to_statevector(tableau);
    dd::CVec dd_vec;
    dd_vec.reserve(vec.size());
    for (const auto& amp : vec) {
        dd_vec.emplace_back(static_cast<dd::fp>(amp.real()),
                            static_cast<dd::fp>(amp.imag()));
    }
    return dd::makeStateFromVector(dd_vec, *dd_pkg);
}
#endif

#ifdef QUASAR_USE_STIM
std::vector<std::complex<double>>
ConversionEngine::tableau_to_statevector(const StimTableau& tableau) const {
    // Use Stim's TableauSimulator to produce the state vector corresponding to
    // ``tableau`` applied to the |0...0> state.  The simulator requires the
    // inverse tableau as its internal state.  Convert the returned
    // ``std::complex<float>`` amplitudes into ``std::complex<double>`` for
    // consistency with the rest of the engine.
    stim::TableauSimulator<stim::MAX_BITWORD_WIDTH> sim(std::mt19937_64{0},
                                                       tableau.num_qubits);
    sim.inv_state = tableau.inverse(false);
    auto vec_f = sim.to_state_vector(true);
    std::vector<std::complex<double>> vec(vec_f.size());
    for (size_t i = 0; i < vec_f.size(); ++i) {
        vec[i] = static_cast<std::complex<double>>(vec_f[i]);
    }
    return vec;
}

std::vector<std::vector<std::complex<double>>>
ConversionEngine::tableau_to_mps(const StimTableau& tableau, std::size_t chi) const {
    auto state = tableau_to_statevector(tableau);
    return statevector_to_mps(state, chi);
}

StimTableau ConversionEngine::convert_boundary_to_tableau(const SSD& ssd) const {
    // Return an identity tableau of the requested size.
    return StimTableau(ssd.boundary_qubits.size());
}
#endif

std::vector<std::vector<std::complex<double>>>
ConversionEngine::statevector_to_mps(const std::vector<std::complex<double>>& state,
                                    std::size_t chi) const {
    // Perform a left-to-right QR sweep to factor ``state`` into an MPS with
    // optional bond-dimension truncation.  The routine scales as
    // ``O(n*chi^3)`` where ``chi`` is the maximum retained bond dimension.
    const std::size_t dim = state.size();
    if (dim == 0) {
        return {};
    }
    const std::size_t n = static_cast<std::size_t>(std::log2(dim));

    std::vector<std::vector<std::complex<double>>> tensors;
    std::vector<std::complex<double>> current = state;
    std::size_t left_dim = 1;

    for (std::size_t qubit = 0; qubit < n; ++qubit) {
        const std::size_t cols = 1ULL << (n - qubit - 1);
        const std::size_t rows = left_dim * 2;

        std::size_t rank = std::min(rows, cols);
        if (chi != 0) {
            rank = std::min(rank, chi);
        }

        std::vector<std::complex<double>> Q(rows * rank, {0.0, 0.0});
        std::vector<std::complex<double>> R(rank * cols, {0.0, 0.0});

        for (std::size_t k = 0; k < rank; ++k) {
            for (std::size_t r = 0; r < rows; ++r) {
                Q[r * rank + k] = current[r * cols + k];
            }
            for (std::size_t j = 0; j < k; ++j) {
                std::complex<double> dot = {0.0, 0.0};
                for (std::size_t r = 0; r < rows; ++r) {
                    dot += std::conj(Q[r * rank + j]) * Q[r * rank + k];
                }
                for (std::size_t r = 0; r < rows; ++r) {
                    Q[r * rank + k] -= dot * Q[r * rank + j];
                }
                R[j * cols + k] = dot;
            }
            double norm = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                norm += std::norm(Q[r * rank + k]);
            }
            norm = std::sqrt(norm);
            if (norm > 0.0) {
                for (std::size_t r = 0; r < rows; ++r) {
                    Q[r * rank + k] /= norm;
                }
            }
            R[k * cols + k] = norm;

            for (std::size_t c = k + 1; c < cols; ++c) {
                std::complex<double> dot = {0.0, 0.0};
                for (std::size_t r = 0; r < rows; ++r) {
                    dot += std::conj(Q[r * rank + k]) * current[r * cols + c];
                }
                R[k * cols + c] = dot;
                for (std::size_t r = 0; r < rows; ++r) {
                    current[r * cols + c] -= dot * Q[r * rank + k];
                }
            }
        }

        std::vector<std::complex<double>> tensor(left_dim * 2 * rank);
        for (std::size_t l = 0; l < left_dim; ++l) {
            for (std::size_t p = 0; p < 2; ++p) {
                for (std::size_t r = 0; r < rank; ++r) {
                    tensor[(l * 2 + p) * rank + r] = Q[(l * 2 + p) * rank + r];
                }
            }
        }
        tensors.push_back(std::move(tensor));

        current.assign(R.begin(), R.begin() + rank * cols);
        left_dim = rank;
    }

    return tensors;
}

#ifdef QUASAR_USE_STIM
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
        // Stim's C++ API does not expose ``Tableau::from_state_vector`` directly.
        // Convert the stabilizer state vector into a circuit and then into a
        // tableau to mirror the Python ``Tableau.from_state_vector`` utility.
        std::vector<std::complex<float>> v(state.begin(), state.end());
        auto circuit = stim::stabilizer_state_vector_to_circuit(v, true);
        return stim::circuit_to_tableau<stim::MAX_BITWORD_WIDTH>(circuit, false, false, false);
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

    // Check for phase-factorable stabilizer states where each qubit contributes
    // an independent {±1, ±i} phase.  Such states correspond to products of
    // single-qubit Clifford operations applied to |+>^{\otimes n}.
    bool factorable = true;
    std::vector<std::complex<double>> qubit_phase(n, {1.0, 0.0});
    for (std::size_t bit = 0; bit < n && factorable; ++bit) {
        std::complex<double> amp0 = state[0];
        std::complex<double> amp1 = state[1ULL << bit];
        if (std::abs(std::abs(amp0) - target_mag) > 1e-9 ||
            std::abs(std::abs(amp1) - target_mag) > 1e-9) {
            factorable = false;
            break;
        }
        std::complex<double> rel = amp1 / amp0;
        if (!phase_ok(rel)) {
            factorable = false;
            break;
        }
        qubit_phase[bit] = rel;
    }
    if (factorable) {
        for (std::size_t idx = 0; idx < dim && factorable; ++idx) {
            std::complex<double> expected = state[0];
            for (std::size_t bit = 0; bit < n; ++bit) {
                if ((idx >> bit) & 1ULL) {
                    expected *= qubit_phase[bit];
                }
            }
            if (std::abs(state[idx] - expected) > 1e-9) {
                factorable = false;
            }
        }
    }
    if (factorable) {
        return StimTableau(n);
    }

    return std::nullopt;
}
#endif  // QUASAR_USE_STIM

} // namespace quasar

