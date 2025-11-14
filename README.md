# LLM Inference FPGA Accelerator

A high-performance FPGA accelerator for large language model (LLM) inference, featuring a GEMM (General Matrix Multiplication) accelerator with AXI-Lite configuration interface and optimized data paths.

## Project Overview

This project implements an LLM inference accelerator on FPGA with the following components:

- **GEMM Accelerator**: Optimized matrix multiplication engine for neural network computations
- **Processing Elements**: Configurable parallel processing units for high throughput
- **AXI-Lite Interface**: Configuration and control interface for easy integration
- **Memory Model**: Efficient memory hierarchy and data movement
- **C++ Testbench**: Comprehensive simulation and verification framework

## Directory Structure

```
llm_fpga_accelerator/
├── rtl/
│   └── src/
│       ├── gemm_accelerator.v       # Main GEMM accelerator module
│       ├── processing_element.v      # Processing element (PE) implementation
│       └── axi_lite_config.v         # AXI-Lite configuration interface
│
├── sim/
│   ├── cpp/
│   │   ├── sim_main.cpp              # Main simulation entry point
│   │   ├── gemm_testbench.cpp        # GEMM testbench with verification
│   │   ├── gemm_testbench.h          # Testbench header
│   │   ├── axi_driver.cpp            # AXI protocol driver
│   │   ├── axi_driver.h              # AXI driver header
│   │   ├── memory_model.cpp          # Memory simulation model
│   │   └── memory_model.h            # Memory model header
│   │
│   ├── build/
│   │   ├── Makefile                  # Build configuration
│   │   ├── build_sim.sh              # Build and test script
│   │   └── obj_dir/                  # Generated Verilator files
│   │
│   └── results/
│       ├── performance/              # Performance benchmark results
│       ├── traces/                   # VCD waveform traces
│       └── synthesis/                # Synthesis results
│
└── README.md                         # This file
```

## Key Features

### GEMM Accelerator
- Configurable matrix dimensions (M×N×K)
- 16-bit integer input support with 32-bit accumulation
- Pipelined processing for high throughput
- Data reuse optimization for improved performance

### AXI-Lite Interface
- Configuration registers for matrix dimensions
- Memory address configuration for A, B, and C matrices
- Start/Done handshaking protocol
- Busy status monitoring

### C++ Simulation
- Random test matrix generation
- Reference GEMM computation for verification
- Tolerance-based result comparison
- CSV performance logging
- Cycle and GOPS (Giga Operations Per Second) tracking

## Prerequisites

- **Verilator**: SystemVerilog simulator
  ```bash
  brew install verilator  # macOS
  # or for Linux/other systems, follow Verilator installation guide
  ```

- **C++17 or later**: For compilation
- **GTKWave** (optional): For waveform visualization
  ```bash
  brew install gtkwave  # macOS
  ```

- **Make**: Build automation tool

## Building and Running

### 1. Basic Build

```bash
cd llm_fpga_accelerator/sim/build
make clean
make all
```

### 2. Build with Script

```bash
cd llm_fpga_accelerator/sim/build
bash build_sim.sh
```

### 3. Run Tests

```bash
# Basic test
make run

# Performance benchmark
make performance_test

# Full test suite (matrix size sweep + waveform trace)
make size_sweep
make trace
```

### 4. Full Build and Test

```bash
bash build_sim.sh --full
```

## Usage Examples

### Running the Simulator

```bash
cd llm_fpga_accelerator/sim/build
make run
```

Expected output:
```
[RUN] Asserting start, busy=0
[RUN] Deasserted start, busy=1
[RUN] Done received, busy=0
[AXI] Wrote 0x40 to addr 0x0 (bvalid=1)
...
```

### Viewing Waveforms

```bash
cd llm_fpga_accelerator/sim/build
make trace
make view_trace
```

This generates a VCD file that can be viewed in GTKWave.

### Performance Analysis

```bash
make performance_test
```

Results are saved to `sim/results/performance/gemm_results.csv` with columns:
- M, N, K (matrix dimensions)
- Cycles (execution cycles)
- GOPS (performance metric)
- Verified (PASS/FAIL)

## C++ Testbench API

### GEMMTestbench Class

#### `generate_test_matrix(rows, cols, seed)`
Generates random test matrices with int16_t values in range [-128, 127].

```cpp
auto matrix = GEMMTestbench::generate_test_matrix(64, 64, 42);
```

#### `compute_reference_gemm(A, B, M, N, K)`
Computes reference GEMM result for verification.

```cpp
auto reference = GEMMTestbench::compute_reference_gemm(A, B, M, N, K);
```

#### `verify_results(actual, reference, tolerance)`
Compares accelerator output against reference implementation.

```cpp
bool pass = GEMMTestbench::verify_results(actual, reference, 0.0);
```

#### `save_results_csv(filename, M, N, K, cycles, gops, verified)`
Logs performance metrics to CSV file.

```cpp
GEMMTestbench::save_results_csv("results.csv", 64, 64, 64, 1024, 42.5, true);
```

## AXI-Lite Configuration Registers

| Address | Register | Description |
|---------|----------|-------------|
| 0x00    | M        | Number of rows in A and C |
| 0x04    | N        | Number of columns in B and C |
| 0x08    | K        | Number of columns in A / rows in B |
| 0x0C    | START    | Start signal (write 1 to begin) |
| 0x10    | ADDR_A   | Base address for matrix A |
| 0x14    | ADDR_B   | Base address for matrix B |
| 0x18    | ADDR_C   | Base address for matrix C |

## Performance Metrics

The accelerator tracks:
- **Throughput**: Operations per second (GOPS)
- **Latency**: Execution time in clock cycles
- **Utilization**: Active cycles vs. total cycles

## Known Limitations

- RTL files (`gemm_accelerator.v`, `processing_element.v`, `axi_lite_config.v`) need to be implemented or provided
- Currently supports int16 × int16 → int32 operations
- Fixed matrix multiplication only (no other operations)

## Future Enhancements

- [ ] Support for floating-point operations
- [ ] Dynamic batch processing
- [ ] Multi-accelerator support
- [ ] DMA integration for better data movement
- [ ] Synthesis and implementation on real FPGA hardware
- [ ] Performance optimization for larger matrices

## Troubleshooting

### Build Fails with "Missing RTL Files"
**Solution**: Implement or obtain the missing Verilog files:
- `rtl/src/gemm_accelerator.v`
- `rtl/src/processing_element.v`
- `rtl/src/axi_lite_config.v`

### Compilation Errors with "GEMMTestbench Undefined"
**Solution**: Ensure `#include "gemm_testbench.h"` is in your source files.

### Verilator Not Found
**Solution**: Install Verilator following the [official guide](https://verilator.org/guide/latest/install.html)

## Contributing

When contributing to this project:
1. Follow the existing code style and naming conventions
2. Add unit tests for new features
3. Update this README with any new features or changes
4. Ensure all tests pass before submitting

## License

[Add appropriate license information here]

## Contact & Support

For issues, questions, or contributions, please contact the project maintainer.

---

**Last Updated**: November 14, 2025