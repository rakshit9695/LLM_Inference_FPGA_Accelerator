// File: gemm_accelerator.v

`timescale 1ns / 1ps

module gemm_accelerator #(
    parameter PE_ROWS    = 8,
    parameter PE_COLS    = 8,
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 32,
    parameter ACC_WIDTH  = 32,
    parameter CYCLES_PER_TILE = 100
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire                      s_axi_awvalid,
    output reg                       s_axi_awready,
    input  wire [31:0]               s_axi_wdata,
    input  wire [3:0]                s_axi_wstrb,
    input  wire                      s_axi_wvalid,
    output reg                       s_axi_wready,
    output reg [1:0]                 s_axi_bresp,
    output reg                       s_axi_bvalid,
    input  wire                      s_axi_bready,
    input  wire [ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire                      s_axi_arvalid,
    output reg                       s_axi_arready,
    output reg [31:0]                s_axi_rdata,
    output reg [1:0]                 s_axi_rresp,
    output reg                       s_axi_rvalid,
    input  wire                      s_axi_rready,
    output reg [ADDR_WIDTH-1:0]      mem_a_addr,
    output reg                       mem_a_en,
    input  wire [DATA_WIDTH*PE_COLS-1:0] mem_a_data,
    input  wire                      mem_a_valid,
    output reg [ADDR_WIDTH-1:0]      mem_b_addr,
    output reg                       mem_b_en,
    input  wire [DATA_WIDTH*PE_ROWS-1:0] mem_b_data,
    input  wire                      mem_b_valid,
    output reg [ADDR_WIDTH-1:0]      mem_c_addr,
    output reg                       mem_c_en,
    output reg                       mem_c_we,
    output reg [ACC_WIDTH*PE_ROWS*PE_COLS-1:0] mem_c_data,
    output reg                       mem_c_valid,
    output reg                       busy,
    output reg                       done,
    input  wire                      start
);

    // Configuration regs
    reg [15:0] matrix_m, matrix_n, matrix_k;
    reg [1:0]  data_format;
    reg        accumulate_mode;
    reg [31:0] cycles_counter;
    reg [ADDR_WIDTH-1:0] addr_a_base, addr_b_base, addr_c_base;

    // FSM states
    localparam IDLE    = 3'b000,
               LOAD    = 3'b001,
               COMPUTE = 3'b010,
               STORE   = 3'b011,
               DONE_ST = 3'b100;
    reg [2:0] state;

    localparam TILE_M = PE_ROWS, TILE_N = PE_COLS, TILE_K = PE_COLS;
    reg [15:0] tiles_m, tiles_n, tiles_k;
    reg [15:0] tile_i, tile_j, tile_k_idx;
    reg [31:0] tile_compute_counter;
    localparam ELEMENT_BYTES = DATA_WIDTH/8;

    // Address calc
    wire [ADDR_WIDTH-1:0] next_addr_a = addr_a_base + (((tile_i*TILE_M)*matrix_k + tile_k_idx*TILE_K)*ELEMENT_BYTES)[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] next_addr_b = addr_b_base + (((tile_k_idx*TILE_K)*matrix_n + tile_j*TILE_N)*ELEMENT_BYTES)[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] next_addr_c = addr_c_base + (((tile_i*TILE_M)*matrix_n + tile_j*TILE_N)*ELEMENT_BYTES)[ADDR_WIDTH-1:0];

    initial begin
        state = IDLE;
        matrix_m = 64; matrix_n = 64; matrix_k = 64;
        data_format = 2; accumulate_mode = 0;
        cycles_counter = 0;
        busy = 0; done = 0;
        s_axi_awready = 0; s_axi_wready = 0; s_axi_bvalid = 0;
        s_axi_arready = 0; s_axi_rvalid = 0; mem_a_en = 0;
        mem_b_en = 0; mem_c_en = 0; mem_c_we = 0; mem_c_valid = 0;
        addr_a_base = 0; addr_b_base = 0; addr_c_base = 0;
        tiles_m = 0; tiles_n = 0; tiles_k = 0;
        tile_i = 0; tile_j = 0; tile_k_idx = 0; tile_compute_counter = 0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            busy  <= 0;
            done  <= 0;
            cycles_counter <= 0;
            tiles_m <= 0; tiles_n <= 0; tiles_k <= 0;
            tile_i <= 0; tile_j <= 0; tile_k_idx <= 0; tile_compute_counter <= 0;
            mem_a_en <= 0; mem_b_en <= 0; mem_c_en <= 0; mem_c_we <= 0; mem_c_valid <= 0;
        end else begin
            cycles_counter <= cycles_counter + 1;
            $display("FSM state=%0d i=%0d j=%0d k=%0d", state, tile_i, tile_j, tile_k_idx);
            case (state)
                IDLE: begin
                    busy <= 0; done <= 0;
                    if (start) begin
                        tiles_m <= matrix_m/TILE_M;
                        tiles_n <= matrix_n/TILE_N;
                        tiles_k <= matrix_k/TILE_K;
                        tile_i <= 0; tile_j <= 0; tile_k_idx <= 0;
                        state <= LOAD; busy <= 1;
                    end
                end
                LOAD: begin
                    mem_a_addr <= next_addr_a; mem_b_addr <= next_addr_b;
                    mem_a_en <= 1; mem_b_en <= 1;
                    if (mem_a_valid && mem_b_valid) begin
                        mem_a_en <= 0; mem_b_en <= 0;
                        tile_compute_counter <= 0;
                        state <= COMPUTE;
                    end
                end
                COMPUTE: begin
                    tile_compute_counter <= tile_compute_counter + 1;
                    if (tile_compute_counter >= CYCLES_PER_TILE-1) begin
                        mem_c_data <= {ACC_WIDTH*PE_ROWS*PE_COLS{1'b0}};
                        mem_c_addr <= next_addr_c;
                        state <= STORE;
                    end
                end
                STORE: begin
                    mem_c_en <= 1; mem_c_we <= 1; mem_c_valid <= 1;
                    mem_c_en <= 0; mem_c_we <= 0; mem_c_valid <= 0;
                    if (tile_k_idx+1 < tiles_k) begin
                        tile_k_idx <= tile_k_idx+1; state <= LOAD;
                    end else begin
                        tile_k_idx <= 0;
                        if (tile_j+1 < tiles_n) begin
                            tile_j <= tile_j+1; state <= LOAD;
                        end else if (tile_i+1 < tiles_m) begin
                            tile_j <= 0; tile_i <= tile_i+1; state <= LOAD;
                        end else begin
                            state <= DONE_ST;
                        end
                    end
                end
                DONE_ST: begin
                    busy <= 0; done <= 1;
                    state <= IDLE; done <= 0;
                end
            endcase
        end
    end

    always @(posedge clk) begin
        $display("[MEM] A_en=%b A_valid=%b B_en=%b B_valid=%b",
                 mem_a_en, mem_a_valid, mem_b_en, mem_b_valid);
    end

    // =====================================================
    // AXI4-Lite read interface with debug registers
    // =====================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= 32'h0;
        end else begin
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid <= 1'b1;
                
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= {16'h0, matrix_m};
                    8'h04: s_axi_rdata <= {16'h0, matrix_n};
                    8'h08: s_axi_rdata <= {16'h0, matrix_k};
                    8'h0C: s_axi_rdata <= addr_a_base;
                    8'h10: s_axi_rdata <= addr_b_base;
                    8'h14: s_axi_rdata <= addr_c_base;
                    8'h18: s_axi_rdata <= {16'h0, tile_i};        // debug: current tile_i
                    8'h1A: s_axi_rdata <= {16'h0, tile_j};        // debug: current tile_j
                    8'h1C: s_axi_rdata <= {16'h0, tile_k_idx};    // debug: current tile_k
                    8'h20: s_axi_rdata <= cycles_counter;
                    8'h24: s_axi_rdata <= {16'h0, tiles_m};
                    8'h28: s_axi_rdata <= {16'h0, tiles_n};
                    8'h2C: s_axi_rdata <= {16'h0, tiles_k};
                    default: s_axi_rdata <= 32'hDEADBEEF;
                endcase
            end else begin
                s_axi_arready <= 1'b0;
                if (s_axi_rready) begin
                    s_axi_rvalid <= 1'b0;
                end
            end
        end
    end
    
    // =====================================================
    // AXI4-Lite write interface for configuration
    // =====================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
        end else begin
            if (s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready <= 1'b1;
                s_axi_bvalid <= 1'b1;
                
                case (s_axi_awaddr[7:0])
                8'h00: matrix_m <= s_axi_wdata[15:0];
                8'h04: matrix_n <= s_axi_wdata[15:0];
                8'h08: matrix_k <= s_axi_wdata[15:0];
                8'h0C: begin
                    data_format     <= s_axi_wdata[1:0];
                    accumulate_mode <= s_axi_wdata[2];
                end
                8'h10: addr_a_base  <= s_axi_wdata;
                8'h14: addr_b_base  <= s_axi_wdata;
                8'h18: addr_c_base  <= s_axi_wdata;
                default: /* no-op */;
                endcase

            end else begin
                s_axi_awready <= 1'b0;
                s_axi_wready <= 1'b0;
                if (s_axi_bready) begin
                    s_axi_bvalid <= 1'b0;
                end
            end
        end
    end

endmodule
