/*
This module represents a single Processing Element (PE), which is the core computation unit in your GEMM accelerator.

Purpose
    Performs a multiply-accumulate (MAC) operation for GEMM.
    Can pipeline inputs and accumulate results over time.
    Designed to be instantiated in arrays (PE_ROWS × PE_COLS).

Module Parameters
    DATA_WIDTH = 16 → Width of input data a_in and b_in.
    ACC_WIDTH = 32 → Width of the accumulator and output c_out.
    This allows the PE to handle 16-bit input matrices but accumulate in a wider 32-bit space to prevent overflow.
    I/O Ports
        Port	Type	Description
        clk	input	Clock signal for sequential operations
        rst_n	input	Active-low reset
        enable	input	Enables the PE operation; if 0, pipeline is stalled
        a_in, b_in	input	Input data values from matrix A and B
        a_out, b_out	output	Registered outputs (for pipelining to next PE)
        c_out	output	Output accumulator result
        valid_in	input	Indicates that a_in and b_in are valid
        valid_out	output	Indicates that c_out is valid
        accumulate	input	If 1, adds a*b to existing accumulator
        data_format	input	Placeholder for future support of INT16/FP16, etc.

    Internal Logic
        Multiplication
            always @(*) begin
                mult_result = $signed(a_in) * $signed(b_in);
            end
            Computes a_in * b_in combinationally.
            Signed multiplication ensures negative numbers are handled.
        
        Pipeline & Accumulation
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    accumulator <= 0;
                    a_out <= 0;
                    b_out <= 0;
                    c_out <= 0;
                    valid_out <= 0;
                    valid_delay <= 0;
                end else if (enable) begin
                    a_out <= a_in;
                    b_out <= b_in;
                    valid_delay <= valid_in;

                    if (valid_delay) begin
                        if (accumulate) accumulator <= accumulator + mult_result;
                        else accumulator <= mult_result;
                    end

                    c_out <= accumulator;
                    valid_out <= valid_delay;
                end
            end
            Stage 1: Registers inputs (a_out, b_out) and delays valid_in.
            Stage 2: Performs accumulation if valid_delay=1.
            Outputs c_out and valid_out one cycle later.
            Reset clears all registers and accumulator.

Key Points
    Can be chained into arrays for GEMM computation.
    Supports both single multiply or accumulate mode.
    Pipelined to allow simultaneous computation across multiple PEs.
    Currently only integer multiplication is implemented ($signed).
*/


`timescale 1ns / 1ps

module processing_element #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 32
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // Data inputs/outputs
    input wire [DATA_WIDTH-1:0] a_in,
    input wire [DATA_WIDTH-1:0] b_in,
    output reg [DATA_WIDTH-1:0] a_out,
    output reg [DATA_WIDTH-1:0] b_out,
    output reg [ACC_WIDTH-1:0] c_out,
    
    // Control signals
    input wire valid_in,
    output reg valid_out,
    input wire accumulate,
    input wire [1:0] data_format
);

    // Internal registers
    reg [ACC_WIDTH-1:0] accumulator;
    reg [ACC_WIDTH-1:0] mult_result;
    reg valid_delay;
    
    // Simple multiply logic (signed integer)
    always @(*) begin
        mult_result = $signed(a_in) * $signed(b_in);
    end
    
    // Accumulation & pipelining
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            a_out <= 0;
            b_out <= 0;
            c_out <= 0;
            valid_out <= 0;
            valid_delay <= 0;
        end else if (enable) begin
            // Stage 1: register inputs
            a_out <= a_in;
            b_out <= b_in;
            valid_delay <= valid_in;
            
            // Stage 2: accumulate
            if (valid_delay) begin
                if (accumulate) begin
                    accumulator <= accumulator + mult_result;
                end else begin
                    accumulator <= mult_result;
                end
            end
            
            c_out <= accumulator;
            valid_out <= valid_delay;
        end
    end

endmodule
