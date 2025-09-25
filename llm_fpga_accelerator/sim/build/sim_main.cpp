// gemm_sim.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

// Verilator headers (adjust include path as needed)
#include <verilated.h>
#include <verilated_vcd_c.h>

// Generated Verilator model headers -- adjust the names/paths if necessary.
// If your model is Vgemm_accelerator, keep as is:
#include "Vgemm_accelerator.h"

// -------------------------------
// Minimal AXI Driver (stubbed)
// -------------------------------
//
// This implements a tiny "register read/write" abstraction that the testbench
// uses to configure the accelerator. If your DUT exposes real s_axi_* signals,
// adapt the methods to drive/check those signals.
//
class AXIDriver {
public:
    AXIDriver(Vgemm_accelerator* dut) : dut(dut) {
        // initialize shadow register map
        regmap.resize(256, 0);
    }

    // Write 32-bit to register offset
    void write_register(uint32_t offset, uint32_t value) {
        // In a real driver you would toggle s_axi_awvalid, s_axi_wvalid, etc.
        // Here we update an internal map and try to reflect to DUT if possible.
        if (offset / 4 < regmap.size()) {
            regmap[offset / 4] = value;
        }

        // Example: if your DUT exposes control registers as inputs,
        // you could assign them here, e.g.:
        // if (offset == 0x00) dut->m_reg_m = value;

        // NOTE: If your generated DUT has named ports for registers, assign them here.
    }

    // Read 32-bit from register offset
    uint32_t read_register(uint32_t offset) {
        if (offset / 4 < regmap.size()) {
            return regmap[offset / 4];
        }
        return 0;
    }

private:
    Vgemm_accelerator* dut;
    std::vector<uint32_t> regmap;
};

// -------------------------------
// Simple in-memory model
// -------------------------------
//
// Very small memory model that supports read/write of int16/32 to a vector<byte>.
// The accelerator DUT should use an external memory model via some memory interface;
// if your DUT drives memory signals (addr/valid/ready/data) adapt update(...) to
// respond to those signals.
//
class MemoryModel {
public:
    MemoryModel(size_t bytes = 1024 * 1024) {
        mem.resize(bytes);
    }

    void write_int16(uint32_t addr, int16_t value) {
        assert(addr + 2 <= mem.size());
        mem[addr + 0] = static_cast<uint8_t>(value & 0xff);
        mem[addr + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
    }

    int16_t read_int16(uint32_t addr) {
        assert(addr + 2 <= mem.size());
        int16_t lo = mem[addr];
        int16_t hi = mem[addr + 1];
        return static_cast<int16_t>((hi << 8) | (lo & 0xff));
    }

    void write_int32(uint32_t addr, int32_t value) {
        assert(addr + 4 <= mem.size());
        mem[addr + 0] = static_cast<uint8_t>(value & 0xff);
        mem[addr + 1] = static_cast<uint8_t>((value >> 8) & 0xff);
        mem[addr + 2] = static_cast<uint8_t>((value >> 16) & 0xff);
        mem[addr + 3] = static_cast<uint8_t>((value >> 24) & 0xff);
    }

    int32_t read_int32(uint32_t addr) {
        assert(addr + 4 <= mem.size());
        uint32_t b0 = mem[addr + 0];
        uint32_t b1 = mem[addr + 1];
        uint32_t b2 = mem[addr + 2];
        uint32_t b3 = mem[addr + 3];
        uint32_t r = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
        return static_cast<int32_t>(r);
    }

    // Called each half-cycle; adapt to your DUT memory interface to respond
    // to read/write requests. For now it's a noop because full AXI handshake
    // would be large.
    void update(Vgemm_accelerator* /*dut*/) {
        // If your DUT drives memory request signals (addr/valid/ready/data),
        // implement those handshakes here and perform memory reads/writes.
    }

private:
    std::vector<uint8_t> mem;
};

// -------------------------------
// GEMM test helper utilities
// -------------------------------

namespace GEMMTestbench {

// Generate M x K matrix of int16 values with a simple pseudo-random pattern
static std::vector<int16_t> generate_test_matrix(size_t rows, size_t cols, uint32_t seed = 0) {
    std::vector<int16_t> out;
    out.reserve(rows * cols);
    uint32_t s = seed ? seed : 12345u;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            // small range to avoid overflow in int32 accumulation
            s = (1103515245u * s + 12345u);
            int16_t v = static_cast<int16_t>((s >> 16) & 0x7FFF);
            v = static_cast<int16_t>( ( (int)v % 8 ) - 4 ); // values -4..3
            out.push_back(v);
        }
    }
    return out;
}

// Compute reference GEMM: C = A * B  (A: MxK, B: KxN) -> C: MxN
static std::vector<int32_t> compute_reference_gemm(const std::vector<int16_t>& A,
                                                   const std::vector<int16_t>& B,
                                                   size_t M, size_t N, size_t K) {
    std::vector<int32_t> C(M * N, 0);
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t k = 0; k < K; ++k) {
                int16_t a = A[m * K + k];
                int16_t b = B[k * N + n];
                acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            }
            C[m * N + n] = acc;
        }
    }
    return C;
}

// Compare actual (int32) against reference within absolute tolerance
static bool verify_results(const std::vector<int32_t>& actual,
                           const std::vector<int32_t>& reference,
                           double /*tolerance*/) {
    if (actual.size() != reference.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != reference[i]) {
            std::cerr << "Mismatch at index " << i << ": actual=" << actual[i]
                      << " expected=" << reference[i] << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace GEMMTestbench

// -------------------------------
// GEMMSimulator class
// -------------------------------
using vluint64_t = unsigned long long;

class GEMMSimulator {
public:
    GEMMSimulator(Vgemm_accelerator* extern_dut,
                  int argc, char** argv,
                  bool enable_tracing = false,
                  const std::string& trace_file = "gemm_trace.vcd")
        : dut(extern_dut),
          sim_time(0),
          trace_enabled(enable_tracing),
          trace_filename(trace_file)
    {
        Verilated::commandArgs(argc, argv);
        Verilated::traceEverOn(enable_tracing);

        if (trace_enabled) {
            tfp = std::make_unique<VerilatedVcdC>();
            dut->trace(tfp.get(), 99);
            tfp->open(trace_filename.c_str());
            std::cout << "Tracing enabled: " << trace_filename << std::endl;
        }

        axi_driver = std::make_unique<AXIDriver>(dut);
        memory_model = std::make_unique<MemoryModel>(1024 * 1024);

        reset();
    }

    ~GEMMSimulator() {
        if (tfp) tfp->close();
        dut->final();
    }

    // Reset the DUT for a few cycles
    void reset() {
        // Set known stable inputs — adapt to the actual port names if different.
        if constexpr (true) {
            // Many generated DUTs use .clk and .rst_n or .reset; adjust as needed.
            // We'll attempt to set commonly-named ports but guard with ifdef-like checks
            // cannot be done in C++ at runtime; so the user must adapt if compilation fails.
        }

        // If your DUT has signals named rst_n/clk/start/done adapt these
        dut->rst_n = 0;
        dut->clk = 0;
        // Some DUTs may not have these signals; if compile fails, change them.

        for (int i = 0; i < 20; ++i) {
            clock_tick();
        }
        dut->rst_n = 1;
        clock_tick();
        std::cout << "Design reset completed" << std::endl;
    }

    // single full clock period (two evals) -- you can call this from tests
    void clock_tick() {
        // rising edge
        dut->clk = 1;
        dut->eval();
        memory_model->update(dut);
        if (trace_enabled && tfp) tfp->dump(sim_time);
        sim_time += 1;

        // falling edge
        dut->clk = 0;
        dut->eval();
        memory_model->update(dut);
        if (trace_enabled && tfp) tfp->dump(sim_time);
        sim_time += 1;
    }

    // configure registers for matrix dims & addresses
    void configure_matrix(uint16_t m, uint16_t n, uint16_t k,
                          uint32_t addr_a, uint32_t addr_b, uint32_t addr_c,
                          uint8_t data_format = 2) {
        std::cout << "Configuring matrix multiplication: " << m << "x" << n << " @ " << k << std::endl;
        axi_driver->write_register(0x00, static_cast<uint32_t>(m));
        axi_driver->write_register(0x04, static_cast<uint32_t>(n));
        axi_driver->write_register(0x08, static_cast<uint32_t>(k));
        axi_driver->write_register(0x10, addr_a);
        axi_driver->write_register(0x14, addr_b);
        axi_driver->write_register(0x18, addr_c);
        uint32_t config = data_format & 0x3;
        axi_driver->write_register(0x0C, config);

        // If your DUT takes registers via direct ports, map them here:
        // e.g. dut->m = m; dut->n = n; ...
    }

    void load_matrix_data(uint32_t base_addr, const std::vector<int16_t>& data) {
        std::cout << "Loading " << data.size() << " elements into memory @ 0x"
                  << std::hex << base_addr << std::dec << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            memory_model->write_int16(base_addr + static_cast<uint32_t>(i * 2), data[i]);
        }
    }

    std::vector<int32_t> read_result_matrix(uint32_t base_addr, size_t rows, size_t cols) {
        std::vector<int32_t> result;
        result.reserve(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            result.push_back(memory_model->read_int32(base_addr + static_cast<uint32_t>(i * 4)));
        }
        return result;
    }

    // Kick off DUT's start signal and wait for done (polling). Adapt port names if needed.
    bool run_gemm_operation(uint32_t timeout_cycles = 100000) {
        std::cout << "Starting GEMM operation..." << std::endl;

        // Drive start if DUT has such a port. Adjust or remove if not present.
        dut->start = 1;
        clock_tick();
        dut->start = 0;
        clock_tick();

        uint32_t cycles = 0;
        while (cycles < timeout_cycles) {
            clock_tick();
            ++cycles;
            // If your DUT exposes 'done' or 'idle' change this test accordingly.
            if (dut->done) {
                std::cout << "GEMM operation completed in " << cycles << " cycles" << std::endl;
                return true;
            }
            if ((cycles & 0x3FFF) == 0) { // occasional status message
                std::cout << " ... waiting, cycles=" << cycles << std::endl;
            }
        }
        std::cout << "GEMM operation timed out after " << timeout_cycles << " cycles" << std::endl;
        return false;
    }

    uint32_t get_cycle_count() {
        return axi_driver->read_register(0x20);
    }

    vluint64_t get_sim_time() const { return sim_time; }

private:
    Vgemm_accelerator* dut; // the Verilated DUT -- lifetime external to simulator
    std::unique_ptr<VerilatedVcdC> tfp;
    std::unique_ptr<AXIDriver> axi_driver;
    std::unique_ptr<MemoryModel> memory_model;
    vluint64_t sim_time;
    bool trace_enabled;
    std::string trace_filename;
};

// -------------------------------
// main
// -------------------------------
int main(int argc, char** argv) {
    std::cout << "GEMM Accelerator Verilator Simulation (refined example)" << std::endl;

    // Basic arg parsing
    bool enable_tracing = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--trace") enable_tracing = true;
        if (a == "--no-trace") enable_tracing = false;
    }

    // Instantiate the DUT (Verilator model)
    Vgemm_accelerator* dut = new Vgemm_accelerator();

    // Recommended: set to 0 to avoid Xs in trace; Verilated::debug(1) if needed.
    Verilated::randReset(2);

    // Create simulator wrapper (passes argc/argv for Verilator)
    GEMMSimulator sim(dut, argc, argv, enable_tracing, "gemm_trace.vcd");

    // Test parameters (tiny case)
    const size_t M = 4, N = 4, K = 4;
    const uint32_t ADDR_A = 0x1000;
    const uint32_t ADDR_B = 0x2000;
    const uint32_t ADDR_C = 0x3000;

    using namespace GEMMTestbench;
    auto matrix_A = generate_test_matrix(M, K, 42);
    auto matrix_B = generate_test_matrix(K, N, 43);
    auto reference_C = compute_reference_gemm(matrix_A, matrix_B, M, N, K);

    // load config & data
    sim.configure_matrix(static_cast<uint16_t>(M),
                         static_cast<uint16_t>(N),
                         static_cast<uint16_t>(K),
                         ADDR_A, ADDR_B, ADDR_C);
    sim.load_matrix_data(ADDR_A, matrix_A);
    sim.load_matrix_data(ADDR_B, matrix_B);

    // Run accelerator
    bool completed = sim.run_gemm_operation(20000);

    if (completed) {
        auto actual_C = sim.read_result_matrix(ADDR_C, M, N);
        bool ok = verify_results(actual_C, reference_C, 0.01);
        std::cout << "Test " << (ok ? "PASSED" : "FAILED") << std::endl;
        delete dut;
        return ok ? 0 : 1;
    } else {
        std::cerr << "Test FAILED: accelerator timed out" << std::endl;
        delete dut;
        return 1;
    }
}
