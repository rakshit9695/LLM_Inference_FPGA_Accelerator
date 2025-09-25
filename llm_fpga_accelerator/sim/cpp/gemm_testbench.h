#ifndef GEMM_TESTBENCH_H
#define GEMM_TESTBENCH_H

#include <vector>
#include <cstdint>
#include <string>

class GEMMTestbench {
public:
    // Generate rows×cols int16 test matrix with pseudo-random values
    std::vector<int16_t> generate_test_matrix(size_t rows,
                                              size_t cols,
                                              uint32_t seed = 123);

    // Compute reference GEMM: A (MxK) × B (KxN) → C (MxN)
    std::vector<int32_t>
    compute_reference_gemm(const std::vector<int16_t>& A,
                           const std::vector<int16_t>& B,
                           size_t M, size_t N, size_t K);

    // Exact integer comparison
    bool verify_results(const std::vector<int32_t>& actual,
                        const std::vector<int32_t>& reference,
                        double tolerance = 0.0);

    // Optional: append results to CSV
    void save_results_csv(const std::string& filename,
                          size_t M, size_t N, size_t K,
                          uint32_t cycles,
                          double gops,
                          bool verified);
};

#endif // GEMM_TESTBENCH_H
