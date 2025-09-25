// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VGEMM_ACCELERATOR__SYMS_H_
#define VERILATED_VGEMM_ACCELERATOR__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vgemm_accelerator.h"

// INCLUDE MODULE CLASSES
#include "Vgemm_accelerator___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vgemm_accelerator__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vgemm_accelerator* const __Vm_modelp;
    bool __Vm_activity = false;  ///< Used by trace routines to determine change occurred
    uint32_t __Vm_baseCode = 0;  ///< Used by trace routines when tracing multiple models
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vgemm_accelerator___024root    TOP;

    // CONSTRUCTORS
    Vgemm_accelerator__Syms(VerilatedContext* contextp, const char* namep, Vgemm_accelerator* modelp);
    ~Vgemm_accelerator__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
