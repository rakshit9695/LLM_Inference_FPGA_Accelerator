#include <iostream>
#include <memory>
#include <string>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstring>
#include <getopt.h>
#include <verilated.h>
#include <verilated_vcd_c.h>

// JSON parsing - simple implementation for config
// File: sim/sim_main_bridge.cpp
// File: sim/sim_main_bridge.cpp

#include "Vgemm_accelerator.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include "axi_driver.h"
#include <iostream>
#include <cstdint>

void write_and_verify(Vgemm_accelerator* dut, uint32_t addr, uint32_t data) {
    // AXI write address phase
    dut->s_axi_awaddr  = addr;
    dut->s_axi_awvalid = 1;
    while (!dut->s_axi_awready) {
        dut->clk ^= 1;
        dut->eval();
    }
    dut->s_axi_awvalid = 0;

    // AXI write data phase
    dut->s_axi_wdata  = data;
    dut->s_axi_wstrb  = 0xF;
    dut->s_axi_wvalid = 1;
    while (!dut->s_axi_wready) {
        dut->clk ^= 1;
        dut->eval();
    }
    dut->s_axi_wvalid = 0;

    // Wait for write response
    while (!dut->s_axi_bvalid) {
        dut->clk ^= 1;
        dut->eval();
    }
    std::cout << "[AXI] Wrote 0x" << std::hex << data
              << " to addr 0x" << addr
              << " (bvalid=" << dut->s_axi_bvalid << ")" << std::dec << std::endl;

    dut->s_axi_bready = 1;
    dut->clk ^= 1;
    dut->eval();
    dut->s_axi_bready = 0;
}

void run(Vgemm_accelerator* dut) {
    // Reset sequence
    dut->rst_n = 0;
    dut->clk   = 0;
    dut->start = 0;
    for (int i = 0; i < 10; ++i) {
        dut->clk ^= 1;
        dut->eval();
    }
    dut->rst_n = 1;
    dut->clk   ^= 1;
    dut->eval();

    // Before pulsing start
    std::cout << "[RUN] Asserting start, busy=" << (int)dut->busy << std::endl;
    dut->start = 1;
    dut->clk   ^= 1;
    dut->eval();
    dut->start = 0;
    std::cout << "[RUN] Deasserted start, busy=" << (int)dut->busy << std::endl;

    // Wait for done
    while (!dut->done) {
        dut->clk ^= 1;
        dut->eval();
    }
    std::cout << "[RUN] Done received, busy=" << (int)dut->busy << std::endl;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto dut = std::make_unique<Vgemm_accelerator>();

    // Initialize
    dut->clk          = 0;
    dut->rst_n        = 1;
    dut->start        = 0;
    dut->s_axi_awvalid= 0;
    dut->s_axi_wvalid = 0;
    dut->s_axi_bready = 0;

    run(dut.get());

    // Example config writes
    write_and_verify(dut.get(), 0x00, 64);
    write_and_verify(dut.get(), 0x04, 64);
    write_and_verify(dut.get(), 0x08, 64);
    write_and_verify(dut.get(), 0x10, 0x1000);
    write_and_verify(dut.get(), 0x14, 0x5000);
    write_and_verify(dut.get(), 0x18, 0x9000);
    write_and_verify(dut.get(), 0x0C, 2);

    return 0;
}
