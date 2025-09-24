// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary model header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef VERILATED_VGEMM_ACCELERATOR_H_
#define VERILATED_VGEMM_ACCELERATOR_H_  // guard

#include "verilated.h"

class Vgemm_accelerator__Syms;
class Vgemm_accelerator___024root;
class VerilatedVcdC;

// This class is the main interface to the Verilated model
class alignas(VL_CACHE_LINE_BYTES) Vgemm_accelerator VL_NOT_FINAL : public VerilatedModel {
  private:
    // Symbol table holding complete model state (owned by this class)
    Vgemm_accelerator__Syms* const vlSymsp;

  public:

    // CONSTEXPR CAPABILITIES
    // Verilated with --trace?
    static constexpr bool traceCapable = true;

    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(&clk,0,0);
    VL_IN8(&rst_n,0,0);
    VL_IN8(&s_axi_awvalid,0,0);
    VL_OUT8(&s_axi_awready,0,0);
    VL_IN8(&s_axi_wstrb,3,0);
    VL_IN8(&s_axi_wvalid,0,0);
    VL_OUT8(&s_axi_wready,0,0);
    VL_OUT8(&s_axi_bresp,1,0);
    VL_OUT8(&s_axi_bvalid,0,0);
    VL_IN8(&s_axi_bready,0,0);
    VL_IN8(&s_axi_arvalid,0,0);
    VL_OUT8(&s_axi_arready,0,0);
    VL_OUT8(&s_axi_rresp,1,0);
    VL_OUT8(&s_axi_rvalid,0,0);
    VL_IN8(&s_axi_rready,0,0);
    VL_OUT8(&mem_a_en,0,0);
    VL_IN8(&mem_a_valid,0,0);
    VL_OUT8(&mem_b_en,0,0);
    VL_IN8(&mem_b_valid,0,0);
    VL_OUT8(&mem_c_en,0,0);
    VL_OUT8(&mem_c_we,0,0);
    VL_OUT8(&mem_c_valid,0,0);
    VL_OUT8(&busy,0,0);
    VL_OUT8(&done,0,0);
    VL_IN8(&start,0,0);
    VL_IN(&s_axi_awaddr,31,0);
    VL_IN(&s_axi_wdata,31,0);
    VL_IN(&s_axi_araddr,31,0);
    VL_OUT(&s_axi_rdata,31,0);
    VL_OUT(&mem_a_addr,31,0);
    VL_INW(&mem_a_data,127,0,4);
    VL_OUT(&mem_b_addr,31,0);
    VL_INW(&mem_b_data,127,0,4);
    VL_OUT(&mem_c_addr,31,0);
    VL_OUTW(&mem_c_data,2047,0,64);

    // CELLS
    // Public to allow access to /* verilator public */ items.
    // Otherwise the application code can consider these internals.

    // Root instance pointer to allow access to model internals,
    // including inlined /* verilator public_flat_* */ items.
    Vgemm_accelerator___024root* const rootp;

    // CONSTRUCTORS
    /// Construct the model; called by application code
    /// If contextp is null, then the model will use the default global context
    /// If name is "", then makes a wrapper with a
    /// single model invisible with respect to DPI scope names.
    explicit Vgemm_accelerator(VerilatedContext* contextp, const char* name = "TOP");
    explicit Vgemm_accelerator(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    virtual ~Vgemm_accelerator();
  private:
    VL_UNCOPYABLE(Vgemm_accelerator);  ///< Copying not allowed

  public:
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    /// Are there scheduled events to handle?
    bool eventsPending();
    /// Returns time at next time slot. Aborts if !eventsPending()
    uint64_t nextTimeSlot();
    /// Trace signals in the model; called by application code
    void trace(VerilatedTraceBaseC* tfp, int levels, int options = 0) { contextp()->trace(tfp, levels, options); }
    /// Retrieve name of this model instance (as passed to constructor).
    const char* name() const;

    // Abstract methods from VerilatedModel
    const char* hierName() const override final;
    const char* modelName() const override final;
    unsigned threads() const override final;
    /// Prepare for cloning the model at the process level (e.g. fork in Linux)
    /// Release necessary resources. Called before cloning.
    void prepareClone() const;
    /// Re-init after cloning the model at the process level (e.g. fork in Linux)
    /// Re-allocate necessary resources. Called after cloning.
    void atClone() const;
    std::unique_ptr<VerilatedTraceConfig> traceConfig() const override final;
  private:
    // Internal functions - trace registration
    void traceBaseModel(VerilatedTraceBaseC* tfp, int levels, int options);
};

#endif  // guard
