// sim/cpp/sim_main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>  // JSON library

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vgemm_accelerator.h"
#include "Vgemm_accelerator__Syms.h"
#include "axi_driver.h"
#include "memory_model.h"
#include "gemm_testbench.h"

using json = nlohmann::json;
using vluint64_t = unsigned long long;

struct Config {
    int M, N, K;
    uint32_t addrA, addrB, addrC;
    int data_format;
};

static Config load_config(const std::string &path) {
    std::ifstream f(path);
    json j; f >> j;
    return {
        j["matrix_m"], j["matrix_n"], j["matrix_k"],
        j["addr_a_base"], j["addr_b_base"], j["addr_c_base"],
        j["data_format"]
    };
}

int main(int argc, char** argv) {
    bool tracing = true;
    std::string trace_file = "gemm_trace.vcd";
    std::string cfg_file, a_file, b_file, out_file, metrics_file;
    size_t timeout = 30;  // seconds

    for (int i = 1; i < argc; i++) {
        std::string s(argv[i]);
        if (s == "--no-trace") tracing = false;
        else if (s=="--tracefile" && i+1<argc) trace_file = argv[++i];
        else if (s=="--config" && i+1<argc) cfg_file = argv[++i];
        else if (s=="--matrix-a" && i+1<argc) a_file = argv[++i];
        else if (s=="--matrix-b" && i+1<argc) b_file = argv[++i];
        else if (s=="--output" && i+1<argc) out_file = argv[++i];
        else if (s=="--metrics" && i+1<argc) metrics_file = argv[++i];
    }

    if (cfg_file.empty() || a_file.empty() || b_file.empty() || out_file.empty() || metrics_file.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --config CFG.json --matrix-a A.bin"
                  << " --matrix-b B.bin --output C.bin --metrics M.json"
                  << " [--no-trace] [--tracefile FILE]\n";
        return 1;
    }

    Config cfg = load_config(cfg_file);
    GEMMSimulator sim(tracing, trace_file);

    // Load inputs
    auto A = GEMMTestbench::generate_test_matrix(cfg.M, cfg.K, 42);
    auto B = GEMMTestbench::generate_test_matrix(cfg.K, cfg.N, 43);
    // Or read from files if binary format needed

    sim.configure_matrix(cfg.M, cfg.N, cfg.K,
                         cfg.addrA, cfg.addrB, cfg.addrC,
                         cfg.data_format);
    sim.load_matrix_data(cfg.addrA, A);
    sim.load_matrix_data(cfg.addrB, B);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = sim.run_gemm_operation(cfg.M*cfg.N*cfg.K*2);  // or use large cycle cap
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!ok) {
        std::cerr << "Hardware GEMM timed out\n";
        return 1;
    }

    // Read results and verify
    auto C = sim.read_result_matrix(cfg.addrC, cfg.M, cfg.N);
    auto Cref = GEMMTestbench::compute_reference_gemm(A, B, cfg.M, cfg.N, cfg.K);
    bool verified = GEMMTestbench::verify_results(C, Cref);

    // Write C to binary
    std::ofstream fout(out_file, std::ios::binary);
    for (auto &v: C) fout.write(reinterpret_cast<char*>(&v), sizeof(v));
    fout.close();

    // Write metrics JSON
    json m;
    m["cycles"] = sim.get_cycle_count();
    m["sim_us"] = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    m["verified"] = verified;
    std::ofstream mfile(metrics_file);
    mfile << m.dump(2);
    mfile.close();

    std::cout << "Simulation " << (verified? "PASSED":"FAILED")
              << ", cycles=" << m["cycles"]
              << ", time(us)=" << m["sim_us"] << "\n";
    return verified?0:1;
}
