// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vgemm_accelerator__pch.h"
#include "verilated_vcd_c.h"

//============================================================
// Constructors

Vgemm_accelerator::Vgemm_accelerator(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vgemm_accelerator__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , s_axi_awvalid{vlSymsp->TOP.s_axi_awvalid}
    , s_axi_awready{vlSymsp->TOP.s_axi_awready}
    , s_axi_wstrb{vlSymsp->TOP.s_axi_wstrb}
    , s_axi_wvalid{vlSymsp->TOP.s_axi_wvalid}
    , s_axi_wready{vlSymsp->TOP.s_axi_wready}
    , s_axi_bresp{vlSymsp->TOP.s_axi_bresp}
    , s_axi_bvalid{vlSymsp->TOP.s_axi_bvalid}
    , s_axi_bready{vlSymsp->TOP.s_axi_bready}
    , s_axi_arvalid{vlSymsp->TOP.s_axi_arvalid}
    , s_axi_arready{vlSymsp->TOP.s_axi_arready}
    , s_axi_rresp{vlSymsp->TOP.s_axi_rresp}
    , s_axi_rvalid{vlSymsp->TOP.s_axi_rvalid}
    , s_axi_rready{vlSymsp->TOP.s_axi_rready}
    , mem_a_en{vlSymsp->TOP.mem_a_en}
    , mem_a_valid{vlSymsp->TOP.mem_a_valid}
    , mem_b_en{vlSymsp->TOP.mem_b_en}
    , mem_b_valid{vlSymsp->TOP.mem_b_valid}
    , mem_c_en{vlSymsp->TOP.mem_c_en}
    , mem_c_we{vlSymsp->TOP.mem_c_we}
    , mem_c_valid{vlSymsp->TOP.mem_c_valid}
    , busy{vlSymsp->TOP.busy}
    , done{vlSymsp->TOP.done}
    , start{vlSymsp->TOP.start}
    , s_axi_awaddr{vlSymsp->TOP.s_axi_awaddr}
    , s_axi_wdata{vlSymsp->TOP.s_axi_wdata}
    , s_axi_araddr{vlSymsp->TOP.s_axi_araddr}
    , s_axi_rdata{vlSymsp->TOP.s_axi_rdata}
    , mem_a_addr{vlSymsp->TOP.mem_a_addr}
    , mem_a_data{vlSymsp->TOP.mem_a_data}
    , mem_b_addr{vlSymsp->TOP.mem_b_addr}
    , mem_b_data{vlSymsp->TOP.mem_b_data}
    , mem_c_addr{vlSymsp->TOP.mem_c_addr}
    , mem_c_data{vlSymsp->TOP.mem_c_data}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
    contextp()->traceBaseModelCbAdd(
        [this](VerilatedTraceBaseC* tfp, int levels, int options) { traceBaseModel(tfp, levels, options); });
}

Vgemm_accelerator::Vgemm_accelerator(const char* _vcname__)
    : Vgemm_accelerator(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vgemm_accelerator::~Vgemm_accelerator() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vgemm_accelerator___024root___eval_debug_assertions(Vgemm_accelerator___024root* vlSelf);
#endif  // VL_DEBUG
void Vgemm_accelerator___024root___eval_static(Vgemm_accelerator___024root* vlSelf);
void Vgemm_accelerator___024root___eval_initial(Vgemm_accelerator___024root* vlSelf);
void Vgemm_accelerator___024root___eval_settle(Vgemm_accelerator___024root* vlSelf);
void Vgemm_accelerator___024root___eval(Vgemm_accelerator___024root* vlSelf);

void Vgemm_accelerator::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vgemm_accelerator::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vgemm_accelerator___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_activity = true;
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vgemm_accelerator___024root___eval_static(&(vlSymsp->TOP));
        Vgemm_accelerator___024root___eval_initial(&(vlSymsp->TOP));
        Vgemm_accelerator___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vgemm_accelerator___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vgemm_accelerator::eventsPending() { return false; }

uint64_t Vgemm_accelerator::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vgemm_accelerator::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vgemm_accelerator___024root___eval_final(Vgemm_accelerator___024root* vlSelf);

VL_ATTR_COLD void Vgemm_accelerator::final() {
    Vgemm_accelerator___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vgemm_accelerator::hierName() const { return vlSymsp->name(); }
const char* Vgemm_accelerator::modelName() const { return "Vgemm_accelerator"; }
unsigned Vgemm_accelerator::threads() const { return 1; }
void Vgemm_accelerator::prepareClone() const { contextp()->prepareClone(); }
void Vgemm_accelerator::atClone() const {
    contextp()->threadPoolpOnClone();
}
std::unique_ptr<VerilatedTraceConfig> Vgemm_accelerator::traceConfig() const {
    return std::unique_ptr<VerilatedTraceConfig>{new VerilatedTraceConfig{false, false, false}};
};

//============================================================
// Trace configuration

void Vgemm_accelerator___024root__trace_decl_types(VerilatedVcd* tracep);

void Vgemm_accelerator___024root__trace_init_top(Vgemm_accelerator___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD static void trace_init(void* voidSelf, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    Vgemm_accelerator___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vgemm_accelerator___024root*>(voidSelf);
    Vgemm_accelerator__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (!vlSymsp->_vm_contextp__->calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
            "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->pushPrefix(std::string{vlSymsp->name()}, VerilatedTracePrefixType::SCOPE_MODULE);
    Vgemm_accelerator___024root__trace_decl_types(tracep);
    Vgemm_accelerator___024root__trace_init_top(vlSelf, tracep);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vgemm_accelerator___024root__trace_register(Vgemm_accelerator___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vgemm_accelerator::traceBaseModel(VerilatedTraceBaseC* tfp, int levels, int options) {
    (void)levels; (void)options;
    VerilatedVcdC* const stfp = dynamic_cast<VerilatedVcdC*>(tfp);
    if (VL_UNLIKELY(!stfp)) {
        vl_fatal(__FILE__, __LINE__, __FILE__,"'Vgemm_accelerator::trace()' called on non-VerilatedVcdC object;"
            " use --trace-fst with VerilatedFst object, and --trace-vcd with VerilatedVcd object");
    }
    stfp->spTrace()->addModel(this);
    stfp->spTrace()->addInitCb(&trace_init, &(vlSymsp->TOP));
    Vgemm_accelerator___024root__trace_register(&(vlSymsp->TOP), stfp->spTrace());
}
