// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vgemm_accelerator.h for the primary calling header

#include "Vgemm_accelerator__pch.h"
#include "Vgemm_accelerator___024root.h"

VL_ATTR_COLD void Vgemm_accelerator___024root___eval_static(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_static\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
    vlSelfRef.__Vtrigprevexpr___TOP__rst_n__0 = vlSelfRef.rst_n;
}

VL_ATTR_COLD void Vgemm_accelerator___024root___eval_initial__TOP(Vgemm_accelerator___024root* vlSelf);
VL_ATTR_COLD void Vgemm_accelerator___024root____Vm_traceActivitySetAll(Vgemm_accelerator___024root* vlSelf);

VL_ATTR_COLD void Vgemm_accelerator___024root___eval_initial(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_initial\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vgemm_accelerator___024root___eval_initial__TOP(vlSelf);
    Vgemm_accelerator___024root____Vm_traceActivitySetAll(vlSelf);
}

extern const VlWide<64>/*2047:0*/ Vgemm_accelerator__ConstPool__CONST_h6be9aa18_0;

VL_ATTR_COLD void Vgemm_accelerator___024root___eval_initial__TOP(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_initial__TOP\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.gemm_accelerator__DOT__state = 0U;
    vlSelfRef.gemm_accelerator__DOT__matrix_m = 0x40U;
    vlSelfRef.gemm_accelerator__DOT__matrix_n = 0x40U;
    vlSelfRef.gemm_accelerator__DOT__matrix_k = 0x40U;
    vlSelfRef.gemm_accelerator__DOT__data_format = 2U;
    vlSelfRef.gemm_accelerator__DOT__accumulate_mode = 0U;
    vlSelfRef.gemm_accelerator__DOT__cycles_counter = 0U;
    vlSelfRef.busy = 0U;
    vlSelfRef.done = 0U;
    vlSelfRef.s_axi_awready = 0U;
    vlSelfRef.s_axi_wready = 0U;
    vlSelfRef.s_axi_bresp = 0U;
    vlSelfRef.s_axi_bvalid = 0U;
    vlSelfRef.s_axi_arready = 0U;
    vlSelfRef.s_axi_rdata = 0U;
    vlSelfRef.s_axi_rresp = 0U;
    vlSelfRef.s_axi_rvalid = 0U;
    vlSelfRef.mem_a_addr = 0U;
    vlSelfRef.mem_a_en = 0U;
    vlSelfRef.mem_b_addr = 0U;
    vlSelfRef.mem_b_en = 0U;
    vlSelfRef.mem_c_addr = 0U;
    vlSelfRef.mem_c_en = 0U;
    vlSelfRef.mem_c_we = 0U;
    IData/*31:0*/ __Vilp1;
    __Vilp1 = 0U;
    while ((__Vilp1 <= 0x3fU)) {
        vlSelfRef.mem_c_data[__Vilp1] = Vgemm_accelerator__ConstPool__CONST_h6be9aa18_0[__Vilp1];
        __Vilp1 = ((IData)(1U) + __Vilp1);
    }
    vlSelfRef.mem_c_valid = 0U;
    vlSelfRef.gemm_accelerator__DOT__addr_a_base = 0U;
    vlSelfRef.gemm_accelerator__DOT__addr_b_base = 0U;
    vlSelfRef.gemm_accelerator__DOT__addr_c_base = 0U;
}

VL_ATTR_COLD void Vgemm_accelerator___024root___eval_final(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_final\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vgemm_accelerator___024root___eval_settle(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___eval_settle\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vgemm_accelerator___024root___dump_triggers__act(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___dump_triggers__act\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vgemm_accelerator___024root___dump_triggers__nba(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___dump_triggers__nba\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vgemm_accelerator___024root____Vm_traceActivitySetAll(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root____Vm_traceActivitySetAll\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vm_traceActivity[0U] = 1U;
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
}

VL_ATTR_COLD void Vgemm_accelerator___024root___ctor_var_reset(Vgemm_accelerator___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root___ctor_var_reset\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16707436170211756652ull);
    vlSelf->rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1638864771569018232ull);
    vlSelf->s_axi_awaddr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7303631981020876172ull);
    vlSelf->s_axi_awvalid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13986037914296269070ull);
    vlSelf->s_axi_awready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14099717354022636468ull);
    vlSelf->s_axi_wdata = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11311253403970331505ull);
    vlSelf->s_axi_wstrb = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 18112015138521062007ull);
    vlSelf->s_axi_wvalid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12168530306759773544ull);
    vlSelf->s_axi_wready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17608475915581965368ull);
    vlSelf->s_axi_bresp = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15162762900795686431ull);
    vlSelf->s_axi_bvalid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9334582144896637853ull);
    vlSelf->s_axi_bready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15653039750784194130ull);
    vlSelf->s_axi_araddr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8722301305194254610ull);
    vlSelf->s_axi_arvalid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17746383479076595557ull);
    vlSelf->s_axi_arready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17791137924766170856ull);
    vlSelf->s_axi_rdata = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12866136205313389248ull);
    vlSelf->s_axi_rresp = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 14929039895447920609ull);
    vlSelf->s_axi_rvalid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15026938065200214434ull);
    vlSelf->s_axi_rready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1794163653381394343ull);
    vlSelf->mem_a_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11793667553881918372ull);
    vlSelf->mem_a_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8322568018556304361ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->mem_a_data, __VscopeHash, 3320846067764280536ull);
    vlSelf->mem_a_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11130434001373939598ull);
    vlSelf->mem_b_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 802523056536400890ull);
    vlSelf->mem_b_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1922258814587538930ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->mem_b_data, __VscopeHash, 14201032999268337065ull);
    vlSelf->mem_b_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15055632728724341405ull);
    vlSelf->mem_c_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10729416870701509833ull);
    vlSelf->mem_c_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13527072633511619737ull);
    vlSelf->mem_c_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2631849456972047353ull);
    VL_SCOPED_RAND_RESET_W(2048, vlSelf->mem_c_data, __VscopeHash, 4820739187672600363ull);
    vlSelf->mem_c_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11890536670619750239ull);
    vlSelf->busy = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6386567572483775230ull);
    vlSelf->done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10296494685231209730ull);
    vlSelf->start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9867861323841650631ull);
    vlSelf->gemm_accelerator__DOT__matrix_m = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2521754157154077165ull);
    vlSelf->gemm_accelerator__DOT__matrix_n = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17310182414779931441ull);
    vlSelf->gemm_accelerator__DOT__matrix_k = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 16021232150182954848ull);
    vlSelf->gemm_accelerator__DOT__data_format = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11538779260885849526ull);
    vlSelf->gemm_accelerator__DOT__accumulate_mode = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12732849993826949611ull);
    vlSelf->gemm_accelerator__DOT__cycles_counter = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7825228887879877403ull);
    vlSelf->gemm_accelerator__DOT__addr_a_base = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 300953333034537185ull);
    vlSelf->gemm_accelerator__DOT__addr_b_base = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6405254566425656308ull);
    vlSelf->gemm_accelerator__DOT__addr_c_base = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9901672064099012851ull);
    vlSelf->gemm_accelerator__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 5148082335104190239ull);
    vlSelf->gemm_accelerator__DOT__next_state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 17437193826217005996ull);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9526919608049418986ull);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14803524876191471008ull);
    for (int __Vi0 = 0; __Vi0 < 2; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
