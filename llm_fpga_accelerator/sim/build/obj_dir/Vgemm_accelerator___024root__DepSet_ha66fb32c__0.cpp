// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vgemm_accelerator.h for the primary calling header

#include "Vgemm_accelerator__pch.h"
#include "Vgemm_accelerator__Syms.h"
#include "Vgemm_accelerator___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vgemm_accelerator___024root___dump_triggers__act(Vgemm_accelerator___024root* vlSelf);
#endif  // VL_DEBUG

void Vgemm_accelerator___024root___eval_triggers__act(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_triggers__act\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.setBit(0U, ((IData)(vlSelfRef.clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__clk__0))));
    vlSelfRef.__VactTriggered.setBit(1U, ((~ (IData)(vlSelfRef.rst_n)) 
                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0)));
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vgemm_accelerator___024root___dump_triggers__act(vlSelf);
    }
#endif
}
