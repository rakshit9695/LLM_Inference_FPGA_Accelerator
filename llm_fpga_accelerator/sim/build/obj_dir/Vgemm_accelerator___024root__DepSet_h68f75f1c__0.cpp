// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vgemm_accelerator.h for the primary calling header

#include "Vgemm_accelerator__pch.h"
#include "Vgemm_accelerator___024root.h"

void Vgemm_accelerator___024root___eval_act(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_act\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vgemm_accelerator___024root___nba_sequent__TOP__0(Vgemm_accelerator___024root* vlSelf);

void Vgemm_accelerator___024root___eval_nba(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_nba\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vgemm_accelerator___024root___nba_sequent__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[1U] = 1U;
    }
}

VL_INLINE_OPT void Vgemm_accelerator___024root___nba_sequent__TOP__0(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___nba_sequent__TOP__0\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __Vdly__gemm_accelerator__DOT__cycles_counter;
    __Vdly__gemm_accelerator__DOT__cycles_counter = 0;
    CData/*2:0*/ __Vdly__gemm_accelerator__DOT__state;
    __Vdly__gemm_accelerator__DOT__state = 0;
    // Body
    __Vdly__gemm_accelerator__DOT__cycles_counter = vlSelfRef.gemm_accelerator__DOT__cycles_counter;
    __Vdly__gemm_accelerator__DOT__state = vlSelfRef.gemm_accelerator__DOT__state;
    if (vlSelfRef.rst_n) {
        if (((IData)(vlSelfRef.s_axi_arvalid) & (~ (IData)(vlSelfRef.s_axi_rvalid)))) {
            vlSelfRef.s_axi_arready = 1U;
            vlSelfRef.s_axi_rvalid = 1U;
            vlSelfRef.s_axi_rdata = (((((((((0U == 
                                             (0xffU 
                                              & vlSelfRef.s_axi_araddr)) 
                                            | (4U == 
                                               (0xffU 
                                                & vlSelfRef.s_axi_araddr))) 
                                           | (8U == 
                                              (0xffU 
                                               & vlSelfRef.s_axi_araddr))) 
                                          | (0xcU == 
                                             (0xffU 
                                              & vlSelfRef.s_axi_araddr))) 
                                         | (0x10U == 
                                            (0xffU 
                                             & vlSelfRef.s_axi_araddr))) 
                                        | (0x14U == 
                                           (0xffU & vlSelfRef.s_axi_araddr))) 
                                       | (0x1cU == 
                                          (0xffU & vlSelfRef.s_axi_araddr))) 
                                      | (0x20U == (0xffU 
                                                   & vlSelfRef.s_axi_araddr)))
                                      ? ((0U == (0xffU 
                                                 & vlSelfRef.s_axi_araddr))
                                          ? (IData)(vlSelfRef.gemm_accelerator__DOT__matrix_m)
                                          : ((4U == 
                                              (0xffU 
                                               & vlSelfRef.s_axi_araddr))
                                              ? (IData)(vlSelfRef.gemm_accelerator__DOT__matrix_n)
                                              : ((8U 
                                                  == 
                                                  (0xffU 
                                                   & vlSelfRef.s_axi_araddr))
                                                  ? (IData)(vlSelfRef.gemm_accelerator__DOT__matrix_k)
                                                  : 
                                                 ((0xcU 
                                                   == 
                                                   (0xffU 
                                                    & vlSelfRef.s_axi_araddr))
                                                   ? vlSelfRef.gemm_accelerator__DOT__addr_a_base
                                                   : 
                                                  ((0x10U 
                                                    == 
                                                    (0xffU 
                                                     & vlSelfRef.s_axi_araddr))
                                                    ? vlSelfRef.gemm_accelerator__DOT__addr_b_base
                                                    : 
                                                   ((0x14U 
                                                     == 
                                                     (0xffU 
                                                      & vlSelfRef.s_axi_araddr))
                                                     ? vlSelfRef.gemm_accelerator__DOT__addr_c_base
                                                     : 
                                                    ((0x1cU 
                                                      == 
                                                      (0xffU 
                                                       & vlSelfRef.s_axi_araddr))
                                                      ? 
                                                     (((IData)(vlSelfRef.done) 
                                                       << 1U) 
                                                      | (IData)(vlSelfRef.busy))
                                                      : vlSelfRef.gemm_accelerator__DOT__cycles_counter)))))))
                                      : 0xdeadbeefU);
        } else {
            vlSelfRef.s_axi_arready = 0U;
            if (vlSelfRef.s_axi_rready) {
                vlSelfRef.s_axi_rvalid = 0U;
            }
        }
        __Vdly__gemm_accelerator__DOT__cycles_counter 
            = ((IData)(1U) + vlSelfRef.gemm_accelerator__DOT__cycles_counter);
        if ((0U == (IData)(vlSelfRef.gemm_accelerator__DOT__state))) {
            vlSelfRef.busy = 0U;
            vlSelfRef.done = 0U;
            if (vlSelfRef.start) {
                __Vdly__gemm_accelerator__DOT__state = 2U;
                vlSelfRef.busy = 1U;
            }
        } else if ((2U == (IData)(vlSelfRef.gemm_accelerator__DOT__state))) {
            if ((0x3e8U < vlSelfRef.gemm_accelerator__DOT__cycles_counter)) {
                __Vdly__gemm_accelerator__DOT__state = 4U;
                vlSelfRef.busy = 0U;
                vlSelfRef.done = 1U;
            }
        } else if ((4U == (IData)(vlSelfRef.gemm_accelerator__DOT__state))) {
            __Vdly__gemm_accelerator__DOT__state = 0U;
            vlSelfRef.done = 0U;
            __Vdly__gemm_accelerator__DOT__cycles_counter = 0U;
        } else {
            __Vdly__gemm_accelerator__DOT__state = 0U;
        }
        if ((((IData)(vlSelfRef.s_axi_awvalid) & (IData)(vlSelfRef.s_axi_wvalid)) 
             & (~ (IData)(vlSelfRef.s_axi_bvalid)))) {
            vlSelfRef.s_axi_awready = 1U;
            vlSelfRef.s_axi_wready = 1U;
            vlSelfRef.s_axi_bvalid = 1U;
            if ((0U == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__matrix_m 
                    = (0xffffU & vlSelfRef.s_axi_wdata);
            } else if ((4U == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__matrix_n 
                    = (0xffffU & vlSelfRef.s_axi_wdata);
            } else if ((8U == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__matrix_k 
                    = (0xffffU & vlSelfRef.s_axi_wdata);
            } else if ((0xcU == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__data_format 
                    = (3U & vlSelfRef.s_axi_wdata);
                vlSelfRef.gemm_accelerator__DOT__accumulate_mode 
                    = (1U & (vlSelfRef.s_axi_wdata 
                             >> 2U));
            } else if ((0x10U == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__addr_a_base 
                    = vlSelfRef.s_axi_wdata;
            } else if ((0x14U == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__addr_b_base 
                    = vlSelfRef.s_axi_wdata;
            } else if ((0x18U == (0xffU & vlSelfRef.s_axi_awaddr))) {
                vlSelfRef.gemm_accelerator__DOT__addr_c_base 
                    = vlSelfRef.s_axi_wdata;
            }
        } else {
            vlSelfRef.s_axi_awready = 0U;
            vlSelfRef.s_axi_wready = 0U;
            if (vlSelfRef.s_axi_bready) {
                vlSelfRef.s_axi_bvalid = 0U;
            }
        }
    } else {
        vlSelfRef.s_axi_arready = 0U;
        vlSelfRef.s_axi_rvalid = 0U;
        vlSelfRef.s_axi_rdata = 0U;
        __Vdly__gemm_accelerator__DOT__state = 0U;
        __Vdly__gemm_accelerator__DOT__cycles_counter = 0U;
        vlSelfRef.busy = 0U;
        vlSelfRef.done = 0U;
        vlSelfRef.s_axi_awready = 0U;
        vlSelfRef.s_axi_wready = 0U;
        vlSelfRef.s_axi_bvalid = 0U;
    }
    vlSelfRef.gemm_accelerator__DOT__cycles_counter 
        = __Vdly__gemm_accelerator__DOT__cycles_counter;
    vlSelfRef.gemm_accelerator__DOT__state = __Vdly__gemm_accelerator__DOT__state;
}

void Vgemm_accelerator___024root___eval_triggers__act(Vgemm_accelerator___024root* vlSelf);

bool Vgemm_accelerator___024root___eval_phase__act(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_phase__act\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<2> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vgemm_accelerator___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vgemm_accelerator___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vgemm_accelerator___024root___eval_phase__nba(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_phase__nba\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vgemm_accelerator___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vgemm_accelerator___024root___dump_triggers__nba(Vgemm_accelerator___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vgemm_accelerator___024root___dump_triggers__act(Vgemm_accelerator___024root* vlSelf);
#endif  // VL_DEBUG

void Vgemm_accelerator___024root___eval(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY(((0x64U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vgemm_accelerator___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("/Users/rakshit9695/Desktop/AI_W_P_RTL/llm_fpga_accelerator/sim/rtl/src/gemm_accelerator.v", 5, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vgemm_accelerator___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("/Users/rakshit9695/Desktop/AI_W_P_RTL/llm_fpga_accelerator/sim/rtl/src/gemm_accelerator.v", 5, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vgemm_accelerator___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vgemm_accelerator___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vgemm_accelerator___024root___eval_debug_assertions(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_debug_assertions\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((vlSelfRef.clk & 0xfeU)))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY(((vlSelfRef.rst_n & 0xfeU)))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY(((vlSelfRef.s_axi_awvalid & 0xfeU)))) {
        Verilated::overWidthError("s_axi_awvalid");}
    if (VL_UNLIKELY(((vlSelfRef.s_axi_wstrb & 0xf0U)))) {
        Verilated::overWidthError("s_axi_wstrb");}
    if (VL_UNLIKELY(((vlSelfRef.s_axi_wvalid & 0xfeU)))) {
        Verilated::overWidthError("s_axi_wvalid");}
    if (VL_UNLIKELY(((vlSelfRef.s_axi_bready & 0xfeU)))) {
        Verilated::overWidthError("s_axi_bready");}
    if (VL_UNLIKELY(((vlSelfRef.s_axi_arvalid & 0xfeU)))) {
        Verilated::overWidthError("s_axi_arvalid");}
    if (VL_UNLIKELY(((vlSelfRef.s_axi_rready & 0xfeU)))) {
        Verilated::overWidthError("s_axi_rready");}
    if (VL_UNLIKELY(((vlSelfRef.mem_a_valid & 0xfeU)))) {
        Verilated::overWidthError("mem_a_valid");}
    if (VL_UNLIKELY(((vlSelfRef.mem_b_valid & 0xfeU)))) {
        Verilated::overWidthError("mem_b_valid");}
    if (VL_UNLIKELY(((vlSelfRef.start & 0xfeU)))) {
        Verilated::overWidthError("start");}
}
#endif  // VL_DEBUG
