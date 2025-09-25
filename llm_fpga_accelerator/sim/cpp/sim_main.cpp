#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cassert>

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vgemm_accelerator.h"
#include "axi_driver.h"
#include "memory_model.h"
#include "gemm_testbench.h"

using vluint64_t = unsigned long long;

class GEMMSimulator {
public:
    GEMMSimulator(bool trace=false, const char* vcd="trace.vcd") {
        Verilated::traceEverOn(trace);
        dut = std::make_unique<Vgemm_accelerator>();
        if (trace) {
            tfp = std::make_unique<VerilatedVcdC>();
            dut->trace(tfp.get(),99);
            tfp->open(vcd);
        }
        axi = std::make_unique<AXIDriver>(dut.get());
        mem = std::make_unique<MemoryModel>(1<<20);
        reset();
    }
    void reset() {
        dut->rst_n=0; dut->clk=0;
        for(int i=0;i<10;i++) tick();
        dut->rst_n=1; tick();
        std::cout<<"Reset complete\n";
    }
    void tick() {
        dut->clk=1; dut->eval(); mem->update(dut.get());
        if(tfp) tfp->dump(time);
        time++;
        dut->clk=0; dut->eval(); mem->update(dut.get());
        if(tfp) tfp->dump(time);
        time++;
    }
    void config(int M,int N,int K,uint32_t a,uint32_t b,uint32_t c) {
        axi->write_register(0x00,M);
        axi->write_register(0x04,N);
        axi->write_register(0x08,K);
        axi->write_register(0x10,a);
        axi->write_register(0x14,b);
        axi->write_register(0x18,c);
        axi->write_register(0x0C,2);
    }
    void load(uint32_t base,const std::vector<int16_t>& A){
        for(size_t i=0;i<A.size();++i) mem->write_int16(base+i*2,A[i]);
    }
    std::vector<int32_t> read(uint32_t base,int M,int N){
        std::vector<int32_t> C; C.reserve(M*N);
        for(int i=0;i<M*N;++i) C.push_back(mem->read_int32(base+i*4));
        return C;
    }
    bool run(int timeout=10000){
        dut->start=1; tick(); dut->start=0; tick();
        for(int i=0;i<timeout;++i){
            tick();
            if(dut->done){ std::cout<<"Done in "<<i<<" cycles\n"; return true; }
        }
        std::cerr<<"Timeout\n"; return false;
    }
private:
    std::unique_ptr<Vgemm_accelerator> dut;
    std::unique_ptr<VerilatedVcdC> tfp;
    std::unique_ptr<AXIDriver> axi;
    std::unique_ptr<MemoryModel> mem;
    vluint64_t time=0;
};

int main(int argc,char**argv){
    bool trace=false;
    if(argc>1 && std::string(argv[1])=="--trace") trace=true;
    GEMMSimulator sim(trace,"trace.vcd");
    const int M=8,N=8,K=8;
    const uint32_t A=0x1000,B=0x2000,C=0x3000;
    GEMMTestbench tb;
    auto Adata=tb.generate_test_matrix(M,K,42);
    auto Bdata=tb.generate_test_matrix(K,N,43);
    auto Cref = tb.compute_reference_gemm(Adata,Bdata,M,N,K);
    sim.config(M,N,K,A,B,C);
    sim.load(A,Adata); sim.load(B,Bdata);
    if(!sim.run()) return 1;
    auto Cact=sim.read(C,M,N);
    if(!tb.verify_results(Cact,Cref,0.01)) return 1;
    std::cout<<"PASS\n"; return 0;
}
