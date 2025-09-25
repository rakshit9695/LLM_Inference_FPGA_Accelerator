#include "axi_driver.h"
#include <iostream>
#include <cassert>
#include "Vgemm_accelerator.h"

AXIDriver::AXIDriver(Vgemm_accelerator* dut_) : dut(dut_) {
    // Initialize AXI-lite interface signals (assumes these signals exist)
    // If your DUT uses different namespace/names, adapt accordingly.
    dut->s_axi_awvalid = 0;
    dut->s_axi_wvalid = 0;
    dut->s_axi_bready = 1;  // ready to accept write response
    dut->s_axi_arvalid = 0;
    dut->s_axi_rready = 1;  // ready to accept read data
}

// NOTE: These routines are blocking and simplistic; they toggle clock manually.
// If you have a central clock in the simulator, integrate these clock toggles with it.
void AXIDriver::write_register(uint32_t addr, uint32_t data) {
    std::cout << "AXI Write: addr=0x" << std::hex << addr 
              << ", data=0x" << data << std::dec << std::endl;

    // Address phase
    dut->s_axi_awaddr = addr;
    dut->s_axi_awvalid = 1;

    // Wait until DUT asserts ready
    while (!dut->s_axi_awready) {
        clock_tick();
    }
    dut->s_axi_awvalid = 0;

    // Data phase
    dut->s_axi_wdata = data;
    dut->s_axi_wstrb = 0xf;
    dut->s_axi_wvalid = 1;

    while (!dut->s_axi_wready) {
        clock_tick();
    }
    dut->s_axi_wvalid = 0;
    dut->s_axi_wstrb = 0;

    // Wait for response
    while (!dut->s_axi_bvalid) {
        clock_tick();
    }
    if (dut->s_axi_bresp != 0) {
        std::cout << "Warning: AXI write response error: " << dut->s_axi_bresp << std::endl;
    }
}

uint32_t AXIDriver::read_register(uint32_t addr) {
    std::cout << "AXI Read: addr=0x" << std::hex << addr << std::dec << std::endl;

    // Address phase
    dut->s_axi_araddr = addr;
    dut->s_axi_arvalid = 1;

    // Wait until DUT asserts ready
    while (!dut->s_axi_arready) {
        clock_tick();
    }
    dut->s_axi_arvalid = 0;

    // Data phase
    while (!dut->s_axi_rvalid) {
        clock_tick();
    }

    uint32_t data = dut->s_axi_rdata;
    if (dut->s_axi_rresp != 0) {
        std::cout << "Warning: AXI read response error: " << dut->s_axi_rresp << std::endl;
    }

    std::cout << "AXI Read result: 0x" << std::hex << data << std::dec << std::endl;
    return data;
}

void AXIDriver::clock_tick() {
    // Rising edge
    dut->clk = 1;
    dut->eval();
    
    // Falling edge
    dut->clk = 0;
    dut->eval();
}