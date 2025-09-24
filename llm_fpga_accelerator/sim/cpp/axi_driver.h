#ifndef AXI_DRIVER_H
#define AXI_DRIVER_H

#include <cstdint>
#include <functional>
#include "Vgemm_accelerator.h"

class AXIDriver {
public:
    // dut pointer is the Verilated model
    explicit AXIDriver(Vgemm_accelerator* dut_);

    // Basic AXI-lite register writes/reads (blocking, simple)
    void write_register(uint32_t addr, uint32_t data);
    uint32_t read_register(uint32_t addr);

    // Simple tick routine used by this driver if needed
    void clock_tick();

private:
    Vgemm_accelerator* dut;
};

#endif // AXI_DRIVER_H
