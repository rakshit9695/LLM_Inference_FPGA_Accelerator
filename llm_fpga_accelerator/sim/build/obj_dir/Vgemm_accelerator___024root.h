// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vgemm_accelerator.h for the primary calling header

#ifndef VERILATED_VGEMM_ACCELERATOR___024ROOT_H_
#define VERILATED_VGEMM_ACCELERATOR___024ROOT_H_  // guard

#include "verilated.h"


class Vgemm_accelerator__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vgemm_accelerator___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(s_axi_awvalid,0,0);
    VL_OUT8(s_axi_awready,0,0);
    VL_IN8(s_axi_wstrb,3,0);
    VL_IN8(s_axi_wvalid,0,0);
    VL_OUT8(s_axi_wready,0,0);
    VL_OUT8(s_axi_bresp,1,0);
    VL_OUT8(s_axi_bvalid,0,0);
    VL_IN8(s_axi_bready,0,0);
    VL_IN8(s_axi_arvalid,0,0);
    VL_OUT8(s_axi_arready,0,0);
    VL_OUT8(s_axi_rresp,1,0);
    VL_OUT8(s_axi_rvalid,0,0);
    VL_IN8(s_axi_rready,0,0);
    VL_OUT8(mem_a_en,0,0);
    VL_IN8(mem_a_valid,0,0);
    VL_OUT8(mem_b_en,0,0);
    VL_IN8(mem_b_valid,0,0);
    VL_OUT8(mem_c_en,0,0);
    VL_OUT8(mem_c_we,0,0);
    VL_OUT8(mem_c_valid,0,0);
    VL_OUT8(busy,0,0);
    VL_OUT8(done,0,0);
    VL_IN8(start,0,0);
    CData/*1:0*/ gemm_accelerator__DOT__data_format;
    CData/*0:0*/ gemm_accelerator__DOT__accumulate_mode;
    CData/*2:0*/ gemm_accelerator__DOT__state;
    CData/*2:0*/ gemm_accelerator__DOT__next_state;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ gemm_accelerator__DOT__matrix_m;
    SData/*15:0*/ gemm_accelerator__DOT__matrix_n;
    SData/*15:0*/ gemm_accelerator__DOT__matrix_k;
    VL_IN(s_axi_awaddr,31,0);
    VL_IN(s_axi_wdata,31,0);
    VL_IN(s_axi_araddr,31,0);
    VL_OUT(s_axi_rdata,31,0);
    VL_OUT(mem_a_addr,31,0);
    VL_INW(mem_a_data,127,0,4);
    VL_OUT(mem_b_addr,31,0);
    VL_INW(mem_b_data,127,0,4);
    VL_OUT(mem_c_addr,31,0);
    VL_OUTW(mem_c_data,2047,0,64);
    IData/*31:0*/ gemm_accelerator__DOT__cycles_counter;
    IData/*31:0*/ gemm_accelerator__DOT__addr_a_base;
    IData/*31:0*/ gemm_accelerator__DOT__addr_b_base;
    IData/*31:0*/ gemm_accelerator__DOT__addr_c_base;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<CData/*0:0*/, 2> __Vm_traceActivity;
    VlTriggerVec<2> __VactTriggered;
    VlTriggerVec<2> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vgemm_accelerator__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vgemm_accelerator___024root(Vgemm_accelerator__Syms* symsp, const char* v__name);
    ~Vgemm_accelerator___024root();
    VL_UNCOPYABLE(Vgemm_accelerator___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
