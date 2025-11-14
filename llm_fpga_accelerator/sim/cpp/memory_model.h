#ifndef MEMORY_MODEL_H
#define MEMORY_MODEL_H

#include <cstdint>
#include <cstddef>  // For size_t

// Forward declaration
class Vgemm_accelerator;

class MemoryModel {
public:
    explicit MemoryModel(size_t size);
    ~MemoryModel();

    // Called by main simulator each cycle to check DUT memory handshake signals.
    void update(Vgemm_accelerator* dut);

    // Host-side helpers for testbench to pre-load data and read results
    void write_int16(uint32_t addr, int16_t value);
    int16_t read_int16(uint32_t addr);
    void write_int32(uint32_t addr, int32_t value);
    int32_t read_int32(uint32_t addr);
    uint16_t read_uint16(uint32_t addr);

private:
    uint8_t* memory;
    size_t memory_size;
};

#endif // MEMORY_MODEL_H