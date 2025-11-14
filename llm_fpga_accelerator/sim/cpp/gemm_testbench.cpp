#include "gemm_testbench.h"
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>

std::vector<int16_t>
GEMMTestbench::generate_test_matrix(size_t rows,
                                    size_t cols,
                                    uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int16_t> dis(-128, 127);
    std::vector<int16_t> matrix;
    matrix.reserve(rows * cols);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix.push_back(static_cast<int16_t>(dis(gen)));
    }
    
    return matrix;
}

std::vector<int32_t>
GEMMTestbench::compute_reference_gemm(const std::vector<int16_t>& A,
                                      const std::vector<int16_t>& B,
                                      size_t M, size_t N, size_t K) {
    std::vector<int32_t> C(M * N, 0);
    
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[m * K + k]) *
                       static_cast<int32_t>(B[k * N + n]);
            }
            C[m * N + n] = acc;
        }
    }
    
    return C;
}

bool
GEMMTestbench::verify_results(const std::vector<int32_t>& actual,
                              const std::vector<int32_t>& reference,
                              double /*tolerance*/) {
    assert(actual.size() == reference.size());
    
    size_t errors = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != reference[i]) {
            if (errors < 10) { // Print first 10 mismatches
                std::cerr << "Mismatch at idx " << i
                          << ": actual=" << actual[i]
                          << ", expected=" << reference[i] << "\n";
            }
            ++errors;
        }
    }
    
    if (errors) {
        std::cerr << "Verification failed: " << errors
                  << "/" << actual.size() << " incorrect\n";
        return false;
    }
    
    return true;
}

void
GEMMTestbench::save_results_csv(const std::string& filename,
                                size_t M, size_t N, size_t K,
                                uint32_t cycles,
                                double gops,
                                bool verified) {
    bool exists = false;
    {
        std::ifstream f(filename);
        exists = f.good();
    }
    
    std::ofstream f(filename, std::ios::app);
    if (!exists) {
        f << "M,N,K,Cycles,GOPS,Verified\n";
    }
    
    f << M << "," << N << "," << K << ","
      << cycles << "," << gops << ","
      << (verified ? "PASS" : "FAIL") << "\n";
}