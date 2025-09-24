// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vgemm_accelerator.h for the primary calling header

#include "Vgemm_accelerator__pch.h"
#include "Vgemm_accelerator__Syms.h"
#include "Vgemm_accelerator___024root.h"

void Vgemm_accelerator___024root___ctor_var_reset(Vgemm_accelerator___024root* vlSelf);

Vgemm_accelerator___024root::Vgemm_accelerator___024root(Vgemm_accelerator__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vgemm_accelerator___024root___ctor_var_reset(this);
}

void Vgemm_accelerator___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vgemm_accelerator___024root::~Vgemm_accelerator___024root() {
}
