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

    // -- PORT A (read) --
    if (dut->mem_a_en) {
        uint32_t addr = dut->mem_a_addr;
        if (addr + (8 * 2) <= memory_size) {
            // Pack 8 x 16-bit values from memory into mem_a_data (128 bits -> 4 words)
            vluint32_t *dst = dut->mem_a_data; // VlWide<4> -> pointer to 4 x 32-bit words
            // Clear dst
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;

            for (int i = 0; i < 8; ++i) {
                uint16_t v = read_uint16(addr + i * 2);
                int word_idx = i / 2;            // two 16-bit values per 32-bit word
                int half_idx = i % 2;            // 0 -> low half, 1 -> high half
                if (half_idx == 0) dst[word_idx] |= (vluint32_t)v;
                else              dst[word_idx] |= ((vluint32_t)v << 16);
            }

            dut->mem_a_valid = 1;
        } else if (addr + 2 <= memory_size) {
            // fallback: if only a single 16-bit value is available, place it in LSB
            vluint32_t *dst = dut->mem_a_data;
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;
            uint16_t v = read_uint16(addr);
            dst[0] = (vluint32_t)v;
            dut->mem_a_valid = 1;
        } else {
            dut->mem_a_valid = 0;
        }
    } else {
        dut->mem_a_valid = 0;
    }

    // -- PORT B (read) --
    if (dut->mem_b_en) {
        uint32_t addr = dut->mem_b_addr;
        if (addr + (8 * 2) <= memory_size) {
            vluint32_t *dst = dut->mem_b_data; // VlWide<4>
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;
            for (int i = 0; i < 8; ++i) {
                uint16_t v = read_uint16(addr + i * 2);
                int word_idx = i / 2;
                int half_idx = i % 2;
                if (half_idx == 0) dst[word_idx] |= (vluint32_t)v;
                else              dst[word_idx] |= ((vluint32_t)v << 16);
            }
            dut->mem_b_valid = 1;
        } else if (addr + 2 <= memory_size) {
            vluint32_t *dst = dut->mem_b_data;
            dst[0] = dst[1] = dst[2] = dst[3] = 0u;
            uint16_t v = read_uint16(addr);
            dst[0] = (vluint32_t)v;
            dut->mem_b_valid = 1;
        } else {
            dut->mem_b_valid = 0;
        }
    } else {
        dut->mem_b_valid = 0;
    }

    // -- PORT C (writeback) --
    if (dut->mem_c_en) {
        if (dut->mem_c_we && dut->mem_c_valid) {
            uint32_t addr = dut->mem_c_addr;
            // mem_c_data is ACC_WIDTH*PE_ROWS*PE_COLS bits. With ACC_WIDTH=32, PE_ROWS=8, PE_COLS=8
            // that's 64 x 32-bit words in the default configuration. Read words and write to memory.
            const vluint32_t *src = dut->mem_c_data; // VlWide<64> -> pointer to 64 words
            const int words = 64; // adjust if your parameters differ

            for (int i = 0; i < words; ++i) {
                uint32_t byte_addr = addr + i * 4u;
                if (byte_addr + 3 < memory_size) {
                    int32_t val = (int32_t)src[i];
                    write_int32(byte_addr, val);
                } else {
                    // out of range: stop or break
                    break;
                }
            }
        }
    }
}

void MemoryModel::write_int16(uint32_t addr, int16_t value) {
    if (addr + 1 < memory_size) {
        memory[addr] = static_cast<uint8_t>(value & 0xFF);
        memory[addr + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    } else {
        std::cerr << "Memory write_int16 out of bounds: " << addr << std::endl;
    }
}

int16_t MemoryModel::read_int16(uint32_t addr) {
    if (addr + 1 < memory_size) {
        return static_cast<int16_t>(memory[addr] | (memory[addr + 1] << 8));
    }
    return 0;
}

void MemoryModel::write_int32(uint32_t addr, int32_t value) {
    if (addr + 3 < memory_size) {
        memory[addr] = static_cast<uint8_t>(value & 0xFF);
        memory[addr + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
        memory[addr + 2] = static_cast<uint8_t>((value >> 16) & 0xFF);
        memory[addr + 3] = static_cast<uint8_t>((value >> 24) & 0xFF);
    } else {
        std::cerr << "Memory write_int32 out of bounds: " << addr << std::endl;
    }
}

int32_t MemoryModel::read_int32(uint32_t addr) {
    if (addr + 3 < memory_size) {
        return static_cast<int32_t>(
            (memory[addr]) |
            (memory[addr + 1] << 8) |
            (memory[addr + 2] << 16) |
            (memory[addr + 3] << 24)
        );
    }
    return 0;
}

uint16_t MemoryModel::read_uint16(uint32_t addr) {
    if (addr + 1 < memory_size) {
        return static_cast<uint16_t>(memory[addr] | (memory[addr + 1] << 8));
    }
    return 0;
}
