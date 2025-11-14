#include "axi_driver.h"
#include <iostream>
#include <iomanip>
#include "Vgemm_accelerator.h"

AXIDriver::AXIDriver(Vgemm_accelerator* dut_) : dut(dut_) {
    // Initialize AXI-lite interface signals
    dut->s_axi_awvalid = 0;
    dut->s_axi_wvalid = 0;
    dut->s_axi_bready = 1; // ready to accept write response
    dut->s_axi_arvalid = 0;
    dut->s_axi_rready = 1; // ready to accept read data
}

// NOTE: This version removes internal clock_tick() calls to avoid conflicts
// with the main simulator's clock. The simulator should call dut->eval() 
// and handle timing externally.

void AXIDriver::write_register(uint32_t addr, uint32_t data) {
    std::cout << "AXI Write: addr=0x" << std::hex << addr
              << ", data=0x" << data << std::dec << std::endl;
    
    // Address phase - set up write address and data simultaneously
    dut->s_axi_awaddr = addr;
    dut->s_axi_awvalid = 1;
    dut->s_axi_wdata = data;
    dut->s_axi_wstrb = 0xf;
    dut->s_axi_wvalid = 1;
    
    // NOTE: In a proper implementation, you would wait for handshakes
    // but without internal clock control, we rely on external simulation
    // to handle the timing and handshake completion.
    
    // Reset signals after one cycle (external sim should handle timing)
    // These will be cleared by the external simulator on next cycle
}

uint32_t AXIDriver::read_register(uint32_t addr) {
    std::cout << "AXI Read: addr=0x" << std::hex << addr << std::dec << std::endl;
    
    // Address phase
    dut->s_axi_araddr = addr;
    dut->s_axi_arvalid = 1;
    
    // NOTE: Similar to write, we set up the transaction and let
    // external simulator handle timing and read data capture
    
    // In practice, the external simulator would:
    // 1. Call this function to set up the read
    // 2. Run simulation cycles until s_axi_rvalid is asserted
    // 3. Capture s_axi_rdata when valid
    // 4. Clear s_axi_arvalid
    
    return 0; // Placeholder - actual read handled by external sim
}

// Simplified tick function - only use if absolutely necessary
// and when not conflicting with main simulator
void AXIDriver::clock_tick() {
    // Rising edge
    dut->clk = 1;
    dut->eval();
    
    // Falling edge  
    dut->clk = 0;
    dut->eval();
}