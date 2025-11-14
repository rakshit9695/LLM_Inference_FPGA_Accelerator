// File: sim/memory_model.cpp

#include "memory_model.h"
#include "Vgemm_accelerator.h"
#include <iostream>
#include <cstring>

MemoryModel::MemoryModel(size_t size) : memory_size(size) {
    memory = new uint8_t[size];
    std::memset(memory, 0, size);
    std::cout << "Memory model initialized: " << size << " bytes" << std::endl;
}

MemoryModel::~MemoryModel() {
    delete[] memory;
}

void MemoryModel::update(Vgemm_accelerator* dut) {
    // PORT A (read)
    if (dut->mem_a_en) {
        uint32_t addr = dut->mem_a_addr;
        if (addr + 16 <= memory_size) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(&dut->mem_a_data);
            dst[0]=dst[1]=dst[2]=dst[3]=0;
            for (int i=0; i<8; ++i) {
                uint16_t v = read_uint16(addr + i*2);
                dst[i/2] |= uint32_t(v) << ((i%2)*16);
            }
            dut->mem_a_valid = 1;
        } else {
            dut->mem_a_valid = 0;
        }
    } else {
        dut->mem_a_valid = 0;
    }

    // PORT B (read)
    if (dut->mem_b_en) {
        uint32_t addr = dut->mem_b_addr;
        if (addr + 16 <= memory_size) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(&dut->mem_b_data);
            dst[0]=dst[1]=dst[2]=dst[3]=0;
            for (int i=0; i<8; ++i) {
                uint16_t v = read_uint16(addr + i*2);
                dst[i/2] |= uint32_t(v) << ((i%2)*16);
            }
            dut->mem_b_valid = 1;
        } else {
            dut->mem_b_valid = 0;
        }
    } else {
        dut->mem_b_valid = 0;
    }

    // PORT C (writeback)
    if (dut->mem_c_en && dut->mem_c_we && dut->mem_c_valid) {
        uint32_t addr = dut->mem_c_addr;
        if (addr + 4 <= memory_size) {
            uint32_t* src = reinterpret_cast<uint32_t*>(&dut->mem_c_data);
            write_int32(addr, int32_t(src[0]));
        }
    }

    std::cout << "[MEM] A_en=" << dut->mem_a_en
              << " A_valid=" << dut->mem_a_valid
              << " B_en=" << dut->mem_b_en
              << " B_valid=" << dut->mem_b_valid << std::endl;
}

void MemoryModel::write_int16(uint32_t addr, int16_t value) {
    if (addr+1 < memory_size) {
        memory[addr]   = value & 0xFF;
        memory[addr+1] = (value>>8) & 0xFF;
    }
}

int16_t MemoryModel::read_int16(uint32_t addr) {
    if (addr+1 < memory_size) {
        return int16_t(memory[addr] | (memory[addr+1]<<8));
    }
    return 0;
}

void MemoryModel::write_int32(uint32_t addr, int32_t value) {
    if (addr+3 < memory_size) {
        for (int i=0; i<4; ++i) {
            memory[addr+i] = (value>>(8*i)) & 0xFF;
        }
    }
}

int32_t MemoryModel::read_int32(uint32_t addr) {
    if (addr+3 < memory_size) {
        uint32_t r=0;
        for (int i=0; i<4; ++i) {
            r |= uint32_t(memory[addr+i]) << (8*i);
        }
        return int32_t(r);
    }
    return 0;
}

uint16_t MemoryModel::read_uint16(uint32_t addr) {
    if (addr+1 < memory_size) {
        return uint16_t(memory[addr] | (memory[addr+1]<<8));
    }
    return 0;
}
