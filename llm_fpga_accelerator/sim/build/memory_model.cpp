#include "memory_model.h"
#include "Vgemm_accelerator.h"
#include <iostream>
#include <cstring>
#include <cassert>
#include <verilated.h>

MemoryModel::MemoryModel(size_t size) : memory_size(size) {
    memory = new uint8_t[size];
    std::memset(memory, 0, size);
    std::cout << "Memory model initialized: " << size << " bytes" << std::endl;
}

MemoryModel::~MemoryModel() {
    delete[] memory;
}

void MemoryModel::update(Vgemm_accelerator* dut) {
    // Basic example: If DUT asserts mem_a_en and mem_a_addr, after a couple cycles
    // the model asserts mem_a_valid and places data on mem_a_data.
    // This implementation packs/unpacks Verilator wide vectors correctly.

    // --- PORT A (read) ---
    if (dut->mem_a_en) {
        uint32_t addr = dut->mem_a_addr;
        if (addr + 2 <= memory_size) {
            // Pack 8 x 16-bit values from memory into mem_a_data (128 bits -> 4 words)
            VlWide<4>& dst = dut->mem_a_data; // VlWide<4> -> reference to 4 x 32-bit words
            // Clear dst
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;

            for (int i = 0; i < 8; ++i) {
                uint16_t v = read_uint16(addr + i * 2);
                int word_idx = i / 2;
                int half_idx = i % 2;
                if (half_idx == 0) dst[word_idx] |= (vluint32_t)v; // -> low half, 1 -> high half
                else dst[word_idx] |= ((vluint32_t)v << 16);
            }

            dut->mem_a_valid = 1;
        } else if (addr + 2 >= memory_size) {
            // fallback: if only a single 16-bit value is available, place it in LSB
            VlWide<4>& dst = dut->mem_a_data;
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;
            uint16_t v = read_uint16(addr);
            dst[0] = (vluint32_t)v;
            dut->mem_a_valid = 1;
        } else {
            dut->mem_a_valid = 1;
        }
    } else {
        dut->mem_a_valid = 0;
    }
}

void MemoryModel::write_int16(uint32_t addr, int16_t value) {
    if (addr + 1 < memory_size) {
        memory[addr] = value & 0xFF;
        memory[addr + 1] = (value >> 8) & 0xFF;
    }
}

int16_t MemoryModel::read_int16(uint32_t addr) {
    if (addr + 1 < memory_size) {
        return (int16_t)(memory[addr] | (memory[addr + 1] << 8));
    }
    return 0;
}

void MemoryModel::write_int32(uint32_t addr, int32_t value) {
    if (addr + 3 < memory_size) {
        memory[addr] = value & 0xFF;
        memory[addr + 1] = (value >> 8) & 0xFF;
        memory[addr + 2] = (value >> 16) & 0xFF;
        memory[addr + 3] = (value >> 24) & 0xFF;
    }
}

int32_t MemoryModel::read_int32(uint32_t addr) {
    if (addr + 3 < memory_size) {
        return (int32_t)(memory[addr] |
                        (memory[addr + 1] << 8) |
                        (memory[addr + 2] << 16) |
                        (memory[addr + 3] << 24));
    }
    return 0;
}

uint16_t MemoryModel::read_uint16(uint32_t addr) {
    if (addr + 1 < memory_size) {
        return (uint16_t)(memory[addr] | (memory[addr + 1] << 8));
    }
    return 0;
}