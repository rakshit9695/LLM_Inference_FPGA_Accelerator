// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vgemm_accelerator__Syms.h"


void Vgemm_accelerator___024root__trace_chg_0_sub_0(Vgemm_accelerator___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vgemm_accelerator___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root__trace_chg_0\n"); );
    // Init
    Vgemm_accelerator___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vgemm_accelerator___024root*>(voidSelf);
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vgemm_accelerator___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vgemm_accelerator___024root__trace_chg_0_sub_0(Vgemm_accelerator___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root__trace_chg_0_sub_0\n"); );
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[1U]))) {
        bufp->chgSData(oldp+0,(vlSelfRef.gemm_accelerator__DOT__matrix_m),16);
        bufp->chgSData(oldp+1,(vlSelfRef.gemm_accelerator__DOT__matrix_n),16);
        bufp->chgSData(oldp+2,(vlSelfRef.gemm_accelerator__DOT__matrix_k),16);
        bufp->chgCData(oldp+3,(vlSelfRef.gemm_accelerator__DOT__data_format),2);
        bufp->chgBit(oldp+4,(vlSelfRef.gemm_accelerator__DOT__accumulate_mode));
        bufp->chgIData(oldp+5,(vlSelfRef.gemm_accelerator__DOT__cycles_counter),32);
        bufp->chgIData(oldp+6,(vlSelfRef.gemm_accelerator__DOT__addr_a_base),32);
        bufp->chgIData(oldp+7,(vlSelfRef.gemm_accelerator__DOT__addr_b_base),32);
        bufp->chgIData(oldp+8,(vlSelfRef.gemm_accelerator__DOT__addr_c_base),32);
        bufp->chgCData(oldp+9,(vlSelfRef.gemm_accelerator__DOT__state),3);
    }
    bufp->chgBit(oldp+10,(vlSelfRef.clk));
    bufp->chgBit(oldp+11,(vlSelfRef.rst_n));
    bufp->chgIData(oldp+12,(vlSelfRef.s_axi_awaddr),32);
    bufp->chgBit(oldp+13,(vlSelfRef.s_axi_awvalid));
    bufp->chgBit(oldp+14,(vlSelfRef.s_axi_awready));
    bufp->chgIData(oldp+15,(vlSelfRef.s_axi_wdata),32);
    bufp->chgCData(oldp+16,(vlSelfRef.s_axi_wstrb),4);
    bufp->chgBit(oldp+17,(vlSelfRef.s_axi_wvalid));
    bufp->chgBit(oldp+18,(vlSelfRef.s_axi_wready));
    bufp->chgCData(oldp+19,(vlSelfRef.s_axi_bresp),2);
    bufp->chgBit(oldp+20,(vlSelfRef.s_axi_bvalid));
    bufp->chgBit(oldp+21,(vlSelfRef.s_axi_bready));
    bufp->chgIData(oldp+22,(vlSelfRef.s_axi_araddr),32);
    bufp->chgBit(oldp+23,(vlSelfRef.s_axi_arvalid));
    bufp->chgBit(oldp+24,(vlSelfRef.s_axi_arready));
    bufp->chgIData(oldp+25,(vlSelfRef.s_axi_rdata),32);
    bufp->chgCData(oldp+26,(vlSelfRef.s_axi_rresp),2);
    bufp->chgBit(oldp+27,(vlSelfRef.s_axi_rvalid));
    bufp->chgBit(oldp+28,(vlSelfRef.s_axi_rready));
    bufp->chgIData(oldp+29,(vlSelfRef.mem_a_addr),32);
    bufp->chgBit(oldp+30,(vlSelfRef.mem_a_en));
    bufp->chgWData(oldp+31,(vlSelfRef.mem_a_data),128);
    bufp->chgBit(oldp+35,(vlSelfRef.mem_a_valid));
    bufp->chgIData(oldp+36,(vlSelfRef.mem_b_addr),32);
    bufp->chgBit(oldp+37,(vlSelfRef.mem_b_en));
    bufp->chgWData(oldp+38,(vlSelfRef.mem_b_data),128);
    bufp->chgBit(oldp+42,(vlSelfRef.mem_b_valid));
    bufp->chgIData(oldp+43,(vlSelfRef.mem_c_addr),32);
    bufp->chgBit(oldp+44,(vlSelfRef.mem_c_en));
    bufp->chgBit(oldp+45,(vlSelfRef.mem_c_we));
    bufp->chgWData(oldp+46,(vlSelfRef.mem_c_data),2048);
    bufp->chgBit(oldp+110,(vlSelfRef.mem_c_valid));
    bufp->chgBit(oldp+111,(vlSelfRef.busy));
    bufp->chgBit(oldp+112,(vlSelfRef.done));
    bufp->chgBit(oldp+113,(vlSelfRef.start));
}

void Vgemm_accelerator___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vgemm_accelerator___024root__trace_cleanup\n"); );
    // Init
    Vgemm_accelerator___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vgemm_accelerator___024root*>(voidSelf);
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
}
