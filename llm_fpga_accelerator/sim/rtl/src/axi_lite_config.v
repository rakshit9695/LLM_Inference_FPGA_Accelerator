/*
Just a configuration script acts as a interface bw the CPU and the GEMM Accelerator

Purpose
    Provides a CPU-accessible register interface to:
    Set matrix dimensions.
    Set base addresses of matrices in memory.
    Control operation modes.
    Read status like busy/done/cycles.
    Acts as a register bank connected to the top-level gemm_acc.v.
*/


`timescale 1ns / 1ps

module axi_lite_config #(
    parameter ADDR_WIDTH = 32
) (
    input wire clk,
    input wire rst_n,
    
    // AXI4-Lite Write Address Channel
    input wire [ADDR_WIDTH-1:0] s_axi_awaddr,
    input wire s_axi_awvalid,
    output reg s_axi_awready,
    
    // AXI4-Lite Write Data Channel
    input wire [31:0] s_axi_wdata,
    input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid,
    output reg s_axi_wready,
    
    // AXI4-Lite Write Response Channel
    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input wire s_axi_bready,
    
    // AXI4-Lite Read Address Channel
    input wire [ADDR_WIDTH-1:0] s_axi_araddr,
    input wire s_axi_arvalid,
    output reg s_axi_arready,
    
    // AXI4-Lite Read Data Channel
    output reg [31:0] s_axi_rdata,
    output reg [1:0] s_axi_rresp,
    output reg s_axi_rvalid,
    input wire s_axi_rready,
    
    // Configuration Registers
    output reg [15:0] matrix_m,
    output reg [15:0] matrix_n,
    output reg [15:0] matrix_k,
    output reg [1:0] data_format,
    output reg accumulate_mode,
    output reg [31:0] addr_a_base,
    output reg [31:0] addr_b_base,
    output reg [31:0] addr_c_base,
    
    // Status Inputs
    input wire [31:0] cycles_counter,
    input wire busy,
    input wire done
);

    // Initialize configuration registers and AXI signals
    initial begin
        matrix_m        = 16'd64;
        matrix_n        = 16'd64;
        matrix_k        = 16'd64;
        data_format     = 2'b10;  // INT16 default
        accumulate_mode = 1'b0;
        addr_a_base     = 32'd0;
        addr_b_base     = 32'd0;
        addr_c_base     = 32'd0;

        s_axi_awready   = 1'b0;
        s_axi_wready    = 1'b0;
        s_axi_bvalid    = 1'b0;
        s_axi_bresp     = 2'b00;
        s_axi_arready   = 1'b0;
        s_axi_rvalid    = 1'b0;
        s_axi_rresp     = 2'b00;
        s_axi_rdata     = 32'd0;
    end

    // AXI4-Lite Read Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= 32'd0;
        end else begin
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= {16'd0, matrix_m};
                    8'h04: s_axi_rdata <= {16'd0, matrix_n};
                    8'h08: s_axi_rdata <= {16'd0, matrix_k};
                    8'h0C: s_axi_rdata <= {29'd0, accumulate_mode, data_format};
                    8'h10: s_axi_rdata <= addr_a_base;
                    8'h14: s_axi_rdata <= addr_b_base;
                    8'h18: s_axi_rdata <= addr_c_base;
                    8'h1C: s_axi_rdata <= {30'd0, done, busy};
                    8'h20: s_axi_rdata <= cycles_counter;
                    default: s_axi_rdata <= 32'hDEADBEEF;
                endcase
            end else begin
                s_axi_arready <= 1'b0;
                if (s_axi_rready) s_axi_rvalid <= 1'b0;
            end
        end
    end

    // AXI4-Lite Write Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
        end else begin
            if (s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
                s_axi_bvalid  <= 1'b1;
                s_axi_bresp   <= 2'b00; // OKAY
                
                case (s_axi_awaddr[7:0])
                    8'h00: matrix_m        <= s_axi_wdata[15:0];
                    8'h04: matrix_n        <= s_axi_wdata[15:0];
                    8'h08: matrix_k        <= s_axi_wdata[15:0];
                    8'h0C: begin
                        data_format     <= s_axi_wdata[1:0];
                        accumulate_mode <= s_axi_wdata[2];
                    end
                    8'h10: addr_a_base     <= s_axi_wdata;
                    8'h14: addr_b_base     <= s_axi_wdata;
                    8'h18: addr_c_base     <= s_axi_wdata;
                    default: ; // no-op
                endcase
            end else begin
                s_axi_awready <= 1'b0;
                s_axi_wready  <= 1'b0;
                if (s_axi_bready) s_axi_bvalid <= 1'b0;
            end
        end
    end

endmodule
