// Simple Processing Element for testing
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
    
    // Simple multiply-accumulate logic
    always @(*) begin
        // Simple integer multiplication for testing
        mult_result = $signed(a_in) * $signed(b_in);
    end
    
    // Accumulation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            a_out <= 0;
            b_out <= 0;
            c_out <= 0;
            valid_out <= 0;
            valid_delay <= 0;
        end else if (enable) begin
            // Pipeline stage 1: Register inputs
            a_out <= a_in;
            b_out <= b_in;
            valid_delay <= valid_in;
            
            // Pipeline stage 2: Accumulate
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