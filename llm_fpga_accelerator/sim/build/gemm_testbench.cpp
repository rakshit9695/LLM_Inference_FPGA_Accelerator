#include "gemm_testbench.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <climits>
#include <cstdint> // Ensure int16_t, int32_t are available

class GEMMTestbench {
public:
    std::vector<int16_t> generate_test_matrix(size_t rows, size_t cols, int seed);
    std::vector<int32_t> compute_reference_gemm(
        const std::vector<int16_t>& A,
        const std::vector<int16_t>& B,
        size_t M, size_t N, size_t K);
    bool verify_results(
        const std::vector<int32_t>& actual,
        const std::vector<int32_t>& expected,
        double tolerance);
    void save_results_csv(
        const std::string& filename,
        size_t M, size_t N, size_t K,
        uint32_t cycles,
        double gops,
        bool verified);
};

std::vector<int16_t> GEMMTestbench::generate_test_matrix(size_t rows, size_t cols, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(-128, 127);

    std::vector<int16_t> matrix;
    matrix.reserve(rows * cols);

    for (size_t i = 0; i < rows * cols; i++) {
        matrix.push_back(static_cast<int16_t>(dis(gen)));
    }

    return matrix;
}

std::vector<int32_t> GEMMTestbench::compute_reference_gemm(
    const std::vector<int16_t>& A,
    const std::vector<int16_t>& B,
    size_t M, size_t N, size_t K) {

    std::vector<int32_t> C(M * N, 0);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += static_cast<int32_t>(A[i * K + k]) *
                       static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }

    return C;
}

bool GEMMTestbench::verify_results(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected,
    double tolerance) {

    if (actual.size() != expected.size()) {
        std::cout << "Size mismatch: actual=" << actual.size()
                  << ", expected=" << expected.size() << std::endl;
        return false;
    }

    size_t errors = 0;
    for (size_t i = 0; i < actual.size(); i++) {
        int32_t actual_val = actual[i];
        int32_t expected_val = expected[i];

        int32_t diff = std::abs(actual_val - expected_val);
        int32_t max_val = std::max(std::abs(actual_val), std::abs(expected_val));

        double rel_error = (max_val > 0) ? static_cast<double>(diff) / static_cast<double>(max_val) : 0.0;

        if (rel_error > tolerance) {
            if (errors < 10) {
                std::cout << "Error at index " << i << ": actual=" << actual_val
                          << ", expected=" << expected_val
                          << ", error=" << rel_error << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0) {
        std::cout << "Verification failed: " << errors << "/" << actual.size()
                  << " elements incorrect" << std::endl;
        return false;
    } else {
        std::cout << "Verification passed: all " << actual.size()
                  << " elements correct" << std::endl;
        return true;
    }
}

void GEMMTestbench::save_results_csv(
    const std::string& filename,
    size_t M, size_t N, size_t K,
    uint32_t cycles,
    double gops,
    bool verified) {

    bool file_exists = false;
    {
        std::ifstream check_file(filename);
        file_exists = check_file.good();
    }

    std::ofstream file(filename, std::ios::app);
    if (!file_exists) {
        file << "M,N,K,Cycles,GOPS,Verified\n";
    }
    file << M << "," << N << "," << K << "," << cycles << ","
         << gops << "," << (verified ? "PASS" : "FAIL") << "\n";
    file.close();
}