# CXXFLAGS += -std=c++14 -O3
# CXXFLAGS += -std=c++14
# Fixed Verilator GEMM Accelerator Simulation Makefile
# Verilator executable
VERILATOR := verilator

# Directories - adjust to your project structure
RTL_DIR      = /Users/rakshit9695/Desktop/AI_W_P_RTL/llm_fpga_accelerator/sim/rtl/src
CPP_DIR      = ../cpp
BUILD_DIR    = .
RESULTS_DIR  = ../results

# Verilator flags
VERILATOR_FLAGS := --cc --exe --build --trace
VERILATOR_FLAGS += -Wall -Wno-UNUSED -Wno-UNOPTFLAT -Wno-DECLFILENAME
VERILATOR_FLAGS += --top-module gemm_accelerator

# C++ compiler flags passed via -CFLAGS
VERILATOR_CFLAGS := -std=c++17 -O3

# Source files
RTL_SOURCES := \
    $(RTL_DIR)/gemm_accelerator.v \
    $(RTL_DIR)/processing_element.v \
    $(RTL_DIR)/axi_lite_config.v

CPP_SOURCES := \
    $(CPP_DIR)/sim_main.cpp \
    $(CPP_DIR)/axi_driver.cpp \
    $(CPP_DIR)/memory_model.cpp \
    $(CPP_DIR)/gemm_testbench.cpp

# Header dependencies
CPP_HEADERS := \
    $(CPP_DIR)/axi_driver.h \
    $(CPP_DIR)/memory_model.h \
    $(CPP_DIR)/gemm_testbench.h

# Output executable
EXECUTABLE := Vgemm_accelerator
TRACE_FILE := gemm_trace.vcd

.PHONY: all clean run trace view_trace performance_test size_sweep help info check_files check_structure

# Check file existence
check_files:
	@echo "=== FILE EXISTENCE CHECK ==="
	@echo "RTL_DIR = $(RTL_DIR)"
	@echo "CPP_DIR = $(CPP_DIR)"
	@echo ""
	@echo "RTL files:"
	@for file in $(RTL_SOURCES); do \
		printf "  %-50s " "$$file:"; \
		[ -f "$$file" ] && echo "✓" || echo "✗"; \
	done
	@echo ""
	@echo "C++ source files:"
	@for file in $(CPP_SOURCES); do \
		printf "  %-50s " "$$file:"; \
		[ -f "$$file" ] && echo "✓" || echo "✗"; \
	done
	@echo ""
	@echo "C++ header files:"
	@for file in $(CPP_HEADERS); do \
		printf "  %-50s " "$$file:"; \
		[ -f "$$file" ] && echo "✓" || echo "✗"; \
	done
	@echo ""

# Inspect directory structure
check_structure:
	@echo "=== DIRECTORY STRUCTURE CHECK ==="
	@echo "PWD: $$(pwd)"
	@echo "RTL contents ($(RTL_DIR)):"
	@ls -la $(RTL_DIR) || true
	@echo ""
	@echo "CPP contents ($(CPP_DIR)):"
	@ls -la $(CPP_DIR) || true
	@echo ""

# Default target
all: check_files $(EXECUTABLE)

# Build simulation
$(EXECUTABLE): $(RTL_SOURCES) $(CPP_SOURCES) $(CPP_HEADERS)
	@echo "=== BUILDING GEMM ACCELERATOR SIMULATION ==="
	@echo "RTL sources: $(RTL_SOURCES)"
	@echo "C++ sources: $(CPP_SOURCES)"
	@echo "Headers:     $(CPP_HEADERS)"
	@echo ""
	@echo "Running Verilator:"
	@echo "$(VERILATOR) $(VERILATOR_FLAGS) --Mdir obj_dir -I$(RTL_DIR) $(RTL_SOURCES) $(CPP_SOURCES) -CFLAGS \"$(VERILATOR_CFLAGS)\""
	@$(VERILATOR) $(VERILATOR_FLAGS) \
		--Mdir obj_dir \
		-I$(RTL_DIR) \
		$(RTL_SOURCES) \
		$(CPP_SOURCES) \
		-CFLAGS "$(VERILATOR_CFLAGS)"
	@echo "✓ Build complete: $(EXECUTABLE)"

# Run simulation
run: $(EXECUTABLE)
	@echo "Running basic simulation test..."
	@./obj_dir/$(EXECUTABLE)
	@echo "✓ Simulation complete"

# Generate VCD trace
trace: $(EXECUTABLE)
	@echo "Running simulation with tracing..."
	@./obj_dir/$(EXECUTABLE) --trace $(TRACE_FILE)
	@echo "✓ Trace file: $(TRACE_FILE)"

# View VCD in GTKWave
view_trace: $(TRACE_FILE)
	@echo "Opening $(TRACE_FILE) in GTKWave..."
	@gtkwave $(TRACE_FILE) &

# Performance test
performance_test: $(EXECUTABLE)
	@echo "Running performance test..."
	@mkdir -p $(RESULTS_DIR)/performance
	@./obj_dir/$(EXECUTABLE) --no-trace > $(RESULTS_DIR)/performance/basic_test.log 2>&1 || true
	@echo "✓ Performance log: $(RESULTS_DIR)/performance/basic_test.log"

# Sweep matrix sizes
size_sweep: $(EXECUTABLE)
	@echo "Running size sweep..."
	@mkdir -p $(RESULTS_DIR)/size_sweep
	@for size in 32 64 128 256; do \
		echo "Testing $$size×$$size"; \
		./obj_dir/$(EXECUTABLE) --size $$size --no-trace \
		  > $(RESULTS_DIR)/size_sweep/test_$${size}.log 2>&1 || true; \
	done
	@echo "✓ Size sweep logs in $(RESULTS_DIR)/size_sweep"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf obj_dir
	@rm -f $(TRACE_FILE)
	@rm -rf $(RESULTS_DIR)/performance $(RESULTS_DIR)/size_sweep
	@echo "✓ Clean complete"

# Help
help:
	@echo "Available targets:"
	@echo "  all             - Build simulation"
	@echo "  check_files     - Verify source/header files exist"
	@echo "  check_structure - Inspect directory layout"
	@echo "  run             - Run basic simulation"
	@echo "  trace           - Generate VCD trace"
	@echo "  view_trace      - Open VCD in GTKWave"
	@echo "  performance_test- Run performance test"
	@echo "  size_sweep      - Sweep matrix sizes"
	@echo "  clean           - Remove build artifacts"
	@echo "  help            - Show this message"

# Info
info:
	@echo "RTL_DIR:     $(RTL_DIR)"
	@echo "CPP_DIR:     $(CPP_DIR)"
	@echo "Build Dir:   $(BUILD_DIR)"
	@echo "Results Dir: $(RESULTS_DIR)"
	@echo "Verilator:   $(VERILATOR)"
	@echo "Executable:  $(EXECUTABLE)"
