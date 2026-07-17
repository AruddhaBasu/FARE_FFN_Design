`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/17/2026 02:37:56 PM
// Design Name: 
// Module Name: ffn_2048
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


//============================================================================
// defines.v - Common parameters and macros for the Pipelined FFN
//============================================================================
// Default parameters:
//   D        = 256   : Input/output vector dimension
//   M        = 16    : Tile dimension
//   DATA_W   = 16    : Fixed-point data width (signed)
//   FRAC_W   = 8     : Fractional bits in fixed-point representation
//   AXI_DATA_W = 128 : AXI data bus width (must be >= DATA_W)
//   ADDR_W   = 32    : AXI address width
//
// Derived:
//   NUM_TILES_D  = D/M     : Number of tiles along D dimension
//   NUM_TILES_4D = 4*D/M   : Number of tiles along 4D dimension
//   PROD_W       = 2*DATA_W: Multiplier product width
//   ACC_UP_W     = PROD_W + $clog2(M) + 2 : Up-proj accumulator width
//   ACC_DOWN_W   = PROD_W + $clog2(M) + 2 : Down-proj accumulator width
//============================================================================

// Macro to index into a flat 1×M tile vector
// tile[k] = tile_flat[k*DATA_W +: DATA_W]
`define TILE_IDX(flat, k, DW) flat[k*DW +: DW]

// Macro to index into a flat M×M weight tile vector
// weight[k][j] = weight_flat[(k*M+j)*DW +: DW]
`define WEIGHT_IDX(flat, k, j, M, DW) flat[(k*(M)+(j))*(DW) +: (DW)]
 
 //============================================================================
// tm_proj_stage.v - Time-Multiplexed Projection Stage
//============================================================================
// Replaces up_projection / down_projection with a single parameterized
// module that processes NUM_COLS output columns per sub-cycle instead of
// all M columns at once. This trades DSP utilization for clock cycles.
//
// With NUM_COLS=14, M=32:
//   DSP per stage: 14×32 = 448   (vs 1024 for full parallel)
//   Sub-cycles per inner iteration: ceil(32/14) = 3
//   Throughput: 1/3× of full parallel
//
// The weight_tile is latched on entry and held stable across sub-cycles.
// In each sub-cycle, a different group of NUM_COLS weight columns is
// extracted and fed to the mul_col instances.
//
// The adder tree pipeline (TREE_DEPTH=5 cycles) is flushed between
// sub-cycles for simplicity. A more aggressive design could overlap
// sub-cycles in the pipeline for higher throughput.
//============================================================================

module tm_proj_stage #(
    parameter D         = 2048,
    parameter M         = 32,
    parameter NUM_COLS  = 14,       // Parallel columns (DSP = NUM_COLS × M per stage)
    parameter MAX_INNER = 256,      // Max inner iterations = max(D/M, 4D/M)
    parameter DATA_W    = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Input from Fetch Stage ----
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [M*M*DATA_W-1:0]         weight_tile,
    input  wire [$clog2(MAX_INNER)-1:0]   tile_col,
    input  wire [$clog2(MAX_INNER)-1:0]   inner_idx,
    input  wire                           is_last,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- Output ----
    output reg  [M*DATA_W-1:0]           result_tile,
    output reg  [$clog2(MAX_INNER)-1:0]   result_col,
    output reg                            valid_out,
    input  wire                           ready_in
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_SUB    = (M + NUM_COLS - 1) / NUM_COLS;  // Sub-cycles per iteration
    localparam TREE_DEPTH = $clog2(M);
    localparam PROD_W     = 2 * DATA_W;
    localparam SUM_W      = PROD_W + $clog2(M);
    localparam ACC_W      = SUM_W + $clog2(MAX_INNER) + 2;
    localparam MAX_POS    = (1 << (DATA_W - 1)) - 1;
    localparam MIN_NEG    = -(1 << (DATA_W - 1));

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE   = 3'd0;  // Waiting for input
    localparam S_FEED   = 3'd1;  // Feed column group to mul_cols
    localparam S_PIPE   = 3'd2;  // Wait for adder tree pipeline
    localparam S_ACC    = 3'd3;  // Accumulate results
    localparam S_OUT    = 3'd4;  // Output final result

    reg [2:0]            state;
    reg [$clog2(NUM_SUB)-1:0] sub_cycle;
    reg [TREE_DEPTH:0]   pipe_cnt;

    // -------------------------------------------------------------------
    // Latched inputs (held stable across sub-cycles)
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0]          input_tile_r;
    reg [M*M*DATA_W-1:0]        weight_tile_r;
    reg [$clog2(MAX_INNER)-1:0]  tile_col_r;
    reg [$clog2(MAX_INNER)-1:0]  inner_idx_r;
    reg                          is_last_r;

    // -------------------------------------------------------------------
    // NUM_COLS mul_col instances (time-shared across sub-cycles)
    // -------------------------------------------------------------------
    wire [NUM_COLS*SUM_W-1:0]  col_results;
    wire [NUM_COLS-1:0]         col_valid_out;

    // Weight column extraction: dynamic MUX based on sub_cycle
    // For mul_col[gj], the actual column index = sub_cycle * NUM_COLS + gj
    // wcol[k] = weight_tile_r[(k*M + col_idx)*DATA_W +: DATA_W]
    reg [NUM_COLS*M*DATA_W-1:0] weight_group;

    integer gj, k;
    always @(*) begin
        weight_group = '0;
        for (gj = 0; gj < NUM_COLS; gj = gj + 1) begin
            for (k = 0; k < M; k = k + 1) begin
                if (sub_cycle * NUM_COLS + gj < M) begin
                    weight_group[(gj*M+k)*DATA_W +: DATA_W] =
                        weight_tile_r[(k*M + sub_cycle*NUM_COLS + gj)*DATA_W +: DATA_W];
                end
            end
        end
    end

    genvar gc;
    generate
        for (gc = 0; gc < NUM_COLS; gc = gc + 1) begin : gen_col
            mul_col #(
                .M      (M),
                .DATA_W (DATA_W),
                .PIPE   (1)
            ) u_mul_col (
                .clk        (clk),
                .rst_n      (rst_n),
                .input_tile (input_tile_r),
                .weight_col (weight_group[gc*M*DATA_W +: M*DATA_W]),
                .valid_in   (state == S_FEED),
                .result     (col_results[gc*SUM_W +: SUM_W]),
                .result_valid(col_valid_out[gc])
            );
        end
    endgenerate

    // -------------------------------------------------------------------
    // M×ACC_W accumulator (one per output element)
    // -------------------------------------------------------------------
    reg [M*ACC_W-1:0] acc;
    reg                acc_valid;
    reg [$clog2(MAX_INNER)-1:0] acc_col;
    reg signed [ACC_W-1:0] acc_val;

    integer j;
    // Number of active columns in current sub-cycle
    reg [$clog2(NUM_COLS)-1:0] active_cols;

    // -------------------------------------------------------------------
    // Main state machine
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            sub_cycle    <= 0;
            pipe_cnt     <= 0;
            ready_out    <= 1'b1;
            valid_out    <= 1'b0;
            input_tile_r <= '0;
            weight_tile_r<= '0;
            tile_col_r   <= '0;
            inner_idx_r  <= '0;
            is_last_r    <= 1'b0;
            acc          <= '0;
            acc_valid    <= 1'b0;
            acc_col      <= '0;
            result_tile  <= '0;
            result_col   <= '0;
            active_cols  <= 0;
        end else begin
            valid_out <= 1'b0;

            case (state)
                // =================================================
                S_IDLE: begin
                    acc_valid <= 1'b0;
                    if (valid_in && ready_out) begin
                        // Latch all inputs
                        input_tile_r  <= input_tile;
                        weight_tile_r <= weight_tile;
                        tile_col_r    <= tile_col;
                        inner_idx_r   <= inner_idx;
                        is_last_r     <= is_last;
                        sub_cycle     <= 0;
                        ready_out     <= 1'b0;

                        // Compute active columns for sub-cycle 0
                        active_cols <= (M < NUM_COLS) ? M[$clog2(NUM_COLS)-1:0] :
                                       NUM_COLS[$clog2(NUM_COLS)-1:0];
                        state <= S_FEED;
                    end
                end

                // =================================================
                S_FEED: begin
                    // Data is being fed to mul_cols (combinational)
                    // Start pipeline counter
                    pipe_cnt <= TREE_DEPTH;
                    state    <= S_PIPE;
                end

                // =================================================
                S_PIPE: begin
                    if (pipe_cnt == 0) begin
                        state <= S_ACC;
                    end else begin
                        pipe_cnt <= pipe_cnt - 1;
                    end
                end

                // =================================================
                S_ACC: begin
                    // Accumulate results from mul_cols into the
                    // corresponding accumulator entries
                    for (j = 0; j < NUM_COLS; j = j + 1) begin
                        if (sub_cycle * NUM_COLS + j < M) begin
                            if (inner_idx_r == 0) begin
                                // First inner iteration: initialize ALL sub-cycles
                                // (each sub-cycle handles different output columns)
                                acc[(sub_cycle*NUM_COLS+j)*ACC_W +: ACC_W] <=
                                    {{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W - 1]}},
                                     col_results[j*SUM_W +: SUM_W]};
                            end else begin
                                // Accumulate
                                acc[(sub_cycle*NUM_COLS+j)*ACC_W +: ACC_W] <=
                                    $signed(acc[(sub_cycle*NUM_COLS+j)*ACC_W +: ACC_W]) +
                                    $signed({{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W - 1]}},
                                             col_results[j*SUM_W +: SUM_W]});
                            end
                        end
                    end
                    acc_col <= tile_col_r;

                    // Advance to next sub-cycle or finish
                    if (sub_cycle >= NUM_SUB - 1) begin
                        // All sub-cycles done for this inner iteration
                        if (is_last_r) begin
                            acc_valid <= 1'b1;
                            state     <= S_OUT;
                        end else begin
                            ready_out <= 1'b1;
                            state     <= S_IDLE;
                        end
                    end else begin
                        // Next sub-cycle
                        sub_cycle <= sub_cycle + 1;
                        // Compute active columns for next sub-cycle
                        if ((sub_cycle + 1) * NUM_COLS + NUM_COLS > M)
                            active_cols <= (M - (sub_cycle + 1) * NUM_COLS);
                        else
                            active_cols <= NUM_COLS[$clog2(NUM_COLS)-1:0];
                        state <= S_FEED;
                    end
                end

                // =================================================
                S_OUT: begin
                    if (ready_in) begin
                        // Output with saturation
                        for (j = 0; j < M; j = j + 1) begin
                            acc_val = $signed(acc[j*ACC_W +: ACC_W]);
                            if (acc_val > MAX_POS)
                                result_tile[j*DATA_W +: DATA_W] <= MAX_POS[DATA_W-1:0];
                            else if (acc_val < MIN_NEG)
                                result_tile[j*DATA_W +: DATA_W] <= MIN_NEG[DATA_W-1:0];
                            else
                                result_tile[j*DATA_W +: DATA_W] <= acc[j*ACC_W +: DATA_W];
                        end
                        result_col <= acc_col;
                        valid_out  <= 1'b1;
                        acc_valid  <= 1'b0;
                        ready_out  <= 1'b1;
                        state      <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule



//============================================================================
// mul_col.v - Multiply-Add Column: M multipliers + 1 adder tree
//============================================================================
// Computes one output element of a tile partial product:
//   result = Σ_{k=0}^{M-1} input_tile[k] * weight_col[k]
//
// This hierarchical module replaces the flat M×M multiplier array.
// Instead of one massive 32768-bit product bus, we have M independent
// column modules, each with M multipliers and one adder tree.
//
// Benefits:
//   - No M*M*PROD_W flat bus (was 32768 bits for M=32)
//   - Only M*DATA_W weight input (per column, not the full tile)
//   - Vivado can place & route each column independently
//   - Fixes HDConfig::lookup() BelGrid crash
//============================================================================
(* keep_hierarchy = "yes" *)
module mul_col #(
    parameter M      = 32,
    parameter DATA_W = 16,
    parameter PIPE   = 1     // 1 = pipelined adder tree (recommended for M>8)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // 1×M input tile
    input  wire [M*DATA_W-1:0]           input_tile,

    // 1×M weight column (k-th element of each row for this output column)
    input  wire [M*DATA_W-1:0]           weight_col,

    // Adder tree valid input
    input  wire                          valid_in,

    // Result
    output wire [DATA_W + $clog2(M) + DATA_W-1:0]  result,
    output wire                          result_valid
);

    localparam PROD_W = 2 * DATA_W;
    localparam SUM_W  = PROD_W + $clog2(M);

    // -------------------------------------------------------------------
    // M multipliers: products[k] = input_tile[k] * weight_col[k]
    // -------------------------------------------------------------------
    wire [M*PROD_W-1:0] products;

    genvar k;
    generate
        for (k = 0; k < M; k = k + 1) begin : gen_mul
            wire signed [DATA_W-1:0] a;
            wire signed [DATA_W-1:0] b;
            wire signed [PROD_W-1:0] p;

            assign a = input_tile[k*DATA_W +: DATA_W];
            assign b = weight_col[k*DATA_W +: DATA_W];
            assign p = a * b;

            assign products[k*PROD_W +: PROD_W] = p;
        end
    endgenerate

    // -------------------------------------------------------------------
    // Adder tree: M products → 1 sum
    // -------------------------------------------------------------------
    adder_tree #(
        .N     (M),
        .IN_W  (PROD_W),
        .PIPE  (PIPE)
    ) u_adder_tree (
        .clk       (clk),
        .rst_n     (rst_n),
        .data_in   (products),
        .valid_in  (valid_in),
        .data_out  (result),
        .valid_out (result_valid)
    );

endmodule


//============================================================================
// axi_read_master.v - Simplified AXI4 Read Master
//============================================================================
// Accepts a read request (base address + burst length), issues AXI4 AR
// channel transactions, collects R channel data, and presents results
// through a simple output interface.
//
// The module handles one burst at a time. A new request is accepted only
// after the previous burst completes.
//
// AXI4 Signals used:
//   AR channel: araddr, arlen, arsize, arburst, arvalid, arready
//   R  channel: rdata, rlast, rvalid, rready
//
// Parameters:
//   AXI_DATA_W : Width of the AXI data bus (e.g., 128)
//   AXI_ADDR_W : Width of the AXI address bus (e.g., 32)
//============================================================================

module axi_read_master #(
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Request interface (from fetch stage) ----
    input  wire [AXI_ADDR_W-1:0]        req_addr,      // Burst start address
    input  wire [7:0]                   req_len,        // Burst length (number of beats - 1)
    input  wire [2:0]                   req_size,       // Transfer size (bytes per beat: 3=8bytes, 4=16bytes, etc.)
    input  wire                          req_valid,      // Request is valid
    output wire                          req_ready,      // Can accept a new request

    // ---- Response interface (to fetch stage) ----
    output wire [AXI_DATA_W-1:0]        resp_data,      // Data beat
    output wire                          resp_last,      // Last beat of burst
    output wire                          resp_valid,     // Data beat is valid
    input  wire                          resp_ready,     // Consumer can accept data

    // ---- AXI4 Read Address Channel ----
    output reg  [AXI_ADDR_W-1:0]        araddr,
    output reg  [7:0]                   arlen,
    output reg  [2:0]                   arsize,
    output reg  [1:0]                   arburst,
    output reg                           arvalid,
    input  wire                          arready,

    // ---- AXI4 Read Data Channel ----
    input  wire [AXI_DATA_W-1:0]        rdata,
    input  wire                          rlast,
    input  wire                          rvalid,
    output reg                           rready
);

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE    = 2'd0;
    localparam S_AR      = 2'd1;  // Driving AR channel
    localparam S_R       = 2'd2;  // Receiving R channel

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            arvalid  <= 1'b0;
            rready   <= 1'b0;
            araddr   <= '0;
            arlen    <= '0;
            arsize   <= '0;
            arburst  <= 2'b01; // INCR
        end else begin
            case (state)
                S_IDLE: begin
                    arvalid <= 1'b0;
                    rready  <= 1'b0;
                    if (req_valid) begin
                        araddr   <= req_addr;
                        arlen    <= req_len;
                        arsize   <= req_size;
                        arburst  <= 2'b01; // INCR burst
                        arvalid  <= 1'b1;
                        state    <= S_AR;
                    end
                end

                S_AR: begin
                    if (arready && arvalid) begin
                        arvalid <= 1'b0;
                        rready  <= 1'b1;
                        state   <= S_R;
                    end
                end

                S_R: begin
                    if (rvalid && rready) begin
                        if (rlast) begin
                            rready <= 1'b0;
                            state  <= S_IDLE;
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Request can be accepted only in IDLE
    assign req_ready = (state == S_IDLE);

    // Response data: directly from AXI R channel
    assign resp_data  = rdata;
    assign resp_last  = rlast;
    assign resp_valid = rvalid && (state == S_R);

endmodule

//============================================================================
// bram_dp.v - Simple Dual-Port BRAM (1 write port, 1 read port)
//============================================================================
// Port A: Write only
// Port B: Read only (synchronous read, 1-cycle latency)
//
// Parameters:
//   DATA_W : Bit width of each word
//   DEPTH  : Number of words
//   ADDR_W : Address width ($clog2(DEPTH))
//============================================================================

module bram_dp #(
    parameter DATA_W = 16,
    parameter DEPTH  = 256,
    parameter ADDR_W = $clog2(DEPTH)
)(
    input  wire                  clk,
    // Port A - Write
    input  wire                  wr_en,
    input  wire [ADDR_W-1:0]    wr_addr,
    input  wire [DATA_W-1:0]    wr_data,
    // Port B - Read
    input  wire                  rd_en,
    input  wire [ADDR_W-1:0]    rd_addr,
    output wire [DATA_W-1:0]    rd_data
);

    // Storage array
    (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Port A: Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Port B: Synchronous read (1-cycle latency)
    reg [DATA_W-1:0] rd_reg;
    always @(posedge clk) begin
        if (rd_en) begin
            rd_reg <= mem[rd_addr];
        end
    end

    assign rd_data = rd_reg;

endmodule


//============================================================================
// bram_dp_wide - Wide-word Dual-Port BRAM for tile storage
//============================================================================
// Stores one tile per address (word width = TILE_SIZE × DATA_W)
// This allows reading/writing an entire 1×M tile in one cycle.
//
// Parameters:
//   DATA_W    : Bit width of each element
//   TILE_SIZE : Number of elements per tile (= M)
//   DEPTH     : Number of tiles
//============================================================================

module bram_dp_wide #(
    parameter DATA_W    = 16,
    parameter TILE_SIZE = 16,
    parameter DEPTH     = 64,
    parameter ADDR_W    = $clog2(DEPTH)
)(
    input  wire                          clk,
    // Port A - Write
    input  wire                          wr_en,
    input  wire [ADDR_W-1:0]            wr_addr,
    input  wire [TILE_SIZE*DATA_W-1:0]   wr_data,
    // Port B - Read
    input  wire                          rd_en,
    input  wire [ADDR_W-1:0]            rd_addr,
    output wire [TILE_SIZE*DATA_W-1:0]   rd_data
);

    localparam WORD_W = TILE_SIZE * DATA_W;

    (* ram_style = "block" *) reg [WORD_W-1:0] mem [0:DEPTH-1];

    // Port A: Write
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Port B: Synchronous read
    reg [WORD_W-1:0] rd_reg;
    always @(posedge clk) begin
        if (rd_en) begin
            rd_reg <= mem[rd_addr];
        end
    end

    assign rd_data = rd_reg;

endmodule

//============================================================================
// adder_tree.v - Parameterized Pipelined Adder Tree (Vivado-Safe)
//============================================================================
// Reduces N inputs to a single sum using a binary tree.
//
// This version uses a FLAT structure with explicit wires at each level
// instead of large 2D arrays. This avoids creating massive intermediate
// buses that crash Vivado's HDConfig/BelGrid database.
//
// For N=32, PIPE=1: 5 pipeline stages, 5-cycle latency.
// Output width = IN_W + $clog2(N)
//============================================================================

module adder_tree #(
    parameter N     = 16,
    parameter IN_W  = 32,
    parameter PIPE  = 0
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire  [N*IN_W-1:0]          data_in,
    input  wire                        valid_in,
    output wire  [IN_W+$clog2(N)-1:0]  data_out,
    output wire                        valid_out
);

    localparam LEVELS = $clog2(N);
    localparam OUT_W  = IN_W + LEVELS;

    // ===================================================================
    // Stage 0: Sign-extend all N inputs from IN_W to OUT_W
    // ===================================================================
    wire [N*OUT_W-1:0] stage0;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_se0
            assign stage0[i*OUT_W +: OUT_W] =
                {{LEVELS{data_in[i*IN_W + IN_W - 1]}}, data_in[i*IN_W +: IN_W]};
        end
    endgenerate

    // ===================================================================
    // Build each level as a flat bus of half the entries
    //
    // At level l:  inputs  = N/(2^l)  values, each (IN_W+l) bits
    //              outputs = N/(2^(l+1)) values, each (IN_W+l+1) bits
    //
    // We pad each value to OUT_W for consistent indexing.
    // Active bits at level l: (IN_W+l) per entry.
    // ===================================================================

    // We need intermediate buses between levels.
    // For Vivado compatibility, we declare them with maximum width
    // and only use the first (count * OUT_W) bits at each level.

    // Stage buses: stage_l_out has (N >> (l+1)) entries padded to OUT_W
    // We use a generate chain: each level reads from the previous stage.

    // Pipeline registers for each level (used when PIPE=1)
    // We declare them as flat buses too.

    // Chain: stage0 → (level 0 logic) → stage1 → (level 1 logic) → ... → result

    // For clean Vivado synthesis, we build a Generate chain where
    // each level instantiates its own local wires and regs.

    wire [N*OUT_W-1:0]  level_bus [0:LEVELS];
    reg  [N*OUT_W-1:0]  level_pipe [0:LEVELS];
    reg                  level_valid_pipe [0:LEVELS];

    assign level_bus[0] = stage0;

    genvar l;
    generate
        for (l = 0; l < LEVELS; l = l + 1) begin : gen_level

            localparam COUNT_IN  = N >> l;
            localparam COUNT_OUT = N >> (l + 1);
            localparam W_CUR     = IN_W + l;        // Active width at this level
            localparam W_NEXT    = IN_W + l + 1;    // Active width after addition

            // Source: either pipelined or combinational from previous level
            wire [N*OUT_W-1:0] src;
            if (PIPE && l > 0) begin : gen_src_pipe
                assign src = level_pipe[l];
            end else begin : gen_src_comb
                assign src = level_bus[l];
            end

            // Generate (COUNT_OUT) adder pairs
            genvar j;
            for (j = 0; j < COUNT_OUT; j = j + 1) begin : gen_add

                wire signed [W_CUR-1:0] a;
                wire signed [W_CUR-1:0] b;
                wire signed [W_NEXT-1:0] sum;

                // Extract pair from source bus (each entry padded to OUT_W)
                assign a   = src[(2*j)*OUT_W +: W_CUR];
                assign b   = src[(2*j+1)*OUT_W +: W_CUR];
                assign sum = $signed(a) + $signed(b);

                if (PIPE) begin : gen_pipelined
                    // Register the sum
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n)
                            level_pipe[l+1][j*OUT_W +: OUT_W] <= '0;
                        else if (level_valid_pipe[l])
                            level_pipe[l+1][j*OUT_W +: OUT_W] <=
                                {{(OUT_W - W_NEXT){1'b0}}, sum};
                    end
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        level_pipe[l+1][j*OUT_W +: OUT_W];
                end else begin : gen_combinational
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        {{(OUT_W - W_NEXT){1'b0}}, sum};
                end
            end

            // Fill unused entries with zero (for clean simulation)
            for (j = COUNT_OUT; j < N; j = j + 1) begin : gen_pad
                if (PIPE) begin : gen_pad_pipe
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        level_pipe[l+1][j*OUT_W +: OUT_W];
                end else begin : gen_pad_comb
                    assign level_bus[l+1][j*OUT_W +: OUT_W] = '0;
                end
            end
        end
    endgenerate

    // ===================================================================
    // Valid signal pipeline
    // ===================================================================
    integer vp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (vp = 0; vp <= LEVELS; vp = vp + 1)
                level_valid_pipe[vp] <= 1'b0;
        end else begin
            level_valid_pipe[0] <= valid_in;
            for (vp = 1; vp <= LEVELS; vp = vp + 1)
                level_valid_pipe[vp] <= level_valid_pipe[vp-1];
        end
    end

    // ===================================================================
    // Output
    // ===================================================================
    assign data_out  = level_bus[LEVELS][OUT_W-1:0];
    assign valid_out = PIPE ? level_valid_pipe[LEVELS] : valid_in;

endmodule

//============================================================================
// fetch_addr_gen.v - Stage 1: Fetch & Address Generation (REVISED)
//============================================================================
// FIX: Registered the AXI request valid signal so that request data
// (addr, len, size) is guaranteed stable when valid is asserted.
// The combinational valid led to a one-cycle race where the AXI master
// captured stale address/size data.
//
// Two-phase operation:
//   UP_PHASE   : Fetches input tiles + up-projection weight tiles → Stage 2
//   DOWN_PHASE : Reads ReLU tiles from local BRAM + fetches down-projection
//                weight tiles from external memory → Stage 4
//============================================================================

module fetch_addr_gen #(
    parameter D          = 256,
    parameter M          = 16,
    parameter DATA_W     = 16,
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output reg                           done,
    output reg                           phase_up,
    output reg                           phase_down,

    // ---- AXI Read Master Interface ----
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,

    // ---- ReLU BRAM Read Interface ----
    output reg                           relu_rd_en,
    output reg  [$clog2(4*D/M)-1:0]     relu_rd_addr,
    input  wire [M*DATA_W-1:0]          relu_rd_data,

    // ---- To Up Projection Stage ----
    output reg  [M*DATA_W-1:0]          up_input_tile,
    output reg  [M*M*DATA_W-1:0]        up_weight_tile,
    output reg  [$clog2(4*D/M)-1:0]     up_tile_col,
    output reg  [$clog2(D/M)-1:0]       up_inner_idx,
    output reg                           up_is_last,
    output reg                           up_valid,
    input  wire                          up_ready,

    // ---- To Down Projection Stage ----
    output reg  [M*DATA_W-1:0]          down_relu_tile,
    output reg  [M*M*DATA_W-1:0]        down_weight_tile,
    output reg  [$clog2(D/M)-1:0]       down_tile_col,
    output reg  [$clog2(4*D/M)-1:0]     down_inner_idx,
    output reg                           down_is_last,
    output reg                           down_valid,
    input  wire                          down_ready
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;
    localparam WORDS_PER_BEAT = AXI_DATA_W / DATA_W;

    localparam INPUT_BASE   = 32'h0000_0000;
    localparam WUP_BASE     = 32'h1000_0000;
    localparam WDOWN_BASE   = 32'h2000_0000;
    localparam TILE_BYTES   = M * (DATA_W / 8);
    localparam WTILE_BYTES  = M * M * (DATA_W / 8);

    localparam INPUT_BEATS  = (M + WORDS_PER_BEAT - 1) / WORDS_PER_BEAT;
    localparam WEIGHT_BEATS = (M * M + WORDS_PER_BEAT - 1) / WORDS_PER_BEAT;

    // AXI size: log2(AXI_DATA_W/8)
    function [2:0] axi_size;
        input integer bytes;
        integer s;
        begin
            s = 0;
            while ((1 << s) < bytes) s = s + 1;
            axi_size = s[2:0];
        end
    endfunction

    localparam AXI_SZ = axi_size(AXI_DATA_W / 8);

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE           = 4'd0;
    localparam S_UP_REQ_INPUT   = 4'd1;
    localparam S_UP_WAIT_INPUT  = 4'd2;
    localparam S_UP_REQ_WEIGHT  = 4'd3;
    localparam S_UP_WAIT_WEIGHT = 4'd4;
    localparam S_UP_SEND        = 4'd5;
    localparam S_DOWN_REQ_RELU  = 4'd6;
    localparam S_DOWN_RELAY     = 4'd7;  // Wait for BRAM read latency
    localparam S_DOWN_WAIT_RELU = 4'd8;
    localparam S_DOWN_REQ_WGT   = 4'd9;
    localparam S_DOWN_WAIT_WGT  = 4'd10;
    localparam S_DOWN_SEND      = 4'd11;
    localparam S_DONE           = 4'd12;

    reg [3:0] state;

    // -------------------------------------------------------------------
    // Iteration counters
    // -------------------------------------------------------------------
    reg [$clog2(NUM_TILES_4D)-1:0] tile_col;
    reg [$clog2(NUM_TILES_D)-1:0]  inner_idx;
    reg [$clog2(NUM_TILES_4D)-1:0] inner_idx_wide;

    // -------------------------------------------------------------------
    // Tile data buffers
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0]     input_tile_buf;
    reg [M*M*DATA_W-1:0]   weight_tile_buf;
    reg [M*DATA_W-1:0]     relu_tile_buf;
    reg [$clog2(M*M)-1:0]  beat_cnt;

    integer k;

    // -------------------------------------------------------------------
    // axi_resp_ready: combinational - high when in a wait state
    // -------------------------------------------------------------------
    assign axi_resp_ready = (state == S_UP_WAIT_INPUT) ||
                            (state == S_UP_WAIT_WEIGHT) ||
                            (state == S_DOWN_WAIT_WGT);

    // -------------------------------------------------------------------
    // Main state machine
    // -------------------------------------------------------------------
    // KEY FIX: Request signals (addr, len, size, valid) are all set with
    // non-blocking assignments BEFORE the AXI master can see them.
    // The pattern is:
    //   Cycle N  : Set req_addr/len/size AND state <= REQ_STATE
    //   Cycle N+1: Request signals are now stable; assert req_valid
    //   Cycle N+2: AXI master sees valid=1 with correct data
    //
    // We use a registered req_valid that is asserted one cycle after
    // entering a request state, ensuring the request data is stable.
    // -------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            phase_up        <= 1'b0;
            phase_down      <= 1'b0;
            tile_col        <= '0;
            inner_idx       <= '0;
            inner_idx_wide  <= '0;
            input_tile_buf  <= '0;
            weight_tile_buf <= '0;
            relu_tile_buf   <= '0;
            beat_cnt        <= '0;
            axi_req_addr    <= '0;
            axi_req_len     <= '0;
            axi_req_size    <= '0;
            axi_req_valid   <= 1'b0;
            up_input_tile   <= '0;
            up_weight_tile  <= '0;
            up_tile_col     <= '0;
            up_inner_idx    <= '0;
            up_is_last      <= 1'b0;
            up_valid        <= 1'b0;
            down_relu_tile  <= '0;
            down_weight_tile<= '0;
            down_tile_col   <= '0;
            down_inner_idx  <= '0;
            down_is_last    <= 1'b0;
            down_valid      <= 1'b0;
            relu_rd_en      <= 1'b0;
            relu_rd_addr    <= '0;
        end else begin
            // Default: deassert one-shot signals
            up_valid     <= 1'b0;
            down_valid   <= 1'b0;
            relu_rd_en   <= 1'b0;
            axi_req_valid <= 1'b0;  // Default: deassert valid each cycle

            case (state)
                // =====================================================
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        phase_up   <= 1'b1;
                        phase_down <= 1'b0;
                        tile_col   <= '0;
                        inner_idx  <= '0;
                        // Pre-set request signals for first input request
                        axi_req_addr  <= INPUT_BASE;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;  // Will be stable next cycle
                        state         <= S_UP_REQ_INPUT;
                    end
                end

                // =====================================================
                // UP PHASE: Request input tile
                // =====================================================
                S_UP_REQ_INPUT: begin
                    // Request signals were set in previous cycle; keep asserting valid
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_UP_WAIT_INPUT;
                    end
                end

                S_UP_WAIT_INPUT: begin
                    if (axi_resp_valid) begin
                        for (k = 0; k < WORDS_PER_BEAT; k = k + 1) begin
                            if (beat_cnt * WORDS_PER_BEAT + k < M) begin
                                input_tile_buf[(beat_cnt * WORDS_PER_BEAT + k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                            end
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            // Pre-set request signals for weight request
                            axi_req_addr  <= WUP_BASE +
                                             (inner_idx * NUM_TILES_4D + tile_col) * WTILE_BYTES;
                            axi_req_len   <= WEIGHT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_UP_REQ_WEIGHT;
                        end
                    end
                end

                // =====================================================
                // UP PHASE: Request weight tile
                // =====================================================
                S_UP_REQ_WEIGHT: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_UP_WAIT_WEIGHT;
                    end
                end

                S_UP_WAIT_WEIGHT: begin
                    if (axi_resp_valid) begin
                        for (k = 0; k < WORDS_PER_BEAT; k = k + 1) begin
                            if (beat_cnt * WORDS_PER_BEAT + k < M * M) begin
                                weight_tile_buf[(beat_cnt * WORDS_PER_BEAT + k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                            end
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            state <= S_UP_SEND;
                        end
                    end
                end

                // =====================================================
                // UP PHASE: Send tile pair to up projection
                // =====================================================
                S_UP_SEND: begin
                    if (up_ready) begin
                        up_input_tile   <= input_tile_buf;
                        up_weight_tile  <= weight_tile_buf;
                        up_tile_col     <= tile_col[$clog2(NUM_TILES_4D)-1:0];
                        up_inner_idx    <= inner_idx;
                        up_is_last      <= (inner_idx == NUM_TILES_D - 1);
                        up_valid        <= 1'b1;

                        // Advance counters and pre-set next request
                        if (inner_idx == NUM_TILES_D - 1) begin
                            if (tile_col == NUM_TILES_4D - 1) begin
                                phase_up   <= 1'b0;
                                phase_down <= 1'b1;
                                tile_col   <= '0;
                                inner_idx_wide <= '0;
                                state <= S_DOWN_REQ_RELU;
                            end else begin
                                tile_col   <= tile_col + 1;
                                inner_idx  <= '0;
                                // Pre-set next input request
                                axi_req_addr  <= INPUT_BASE;
                                axi_req_len   <= INPUT_BEATS - 1;
                                axi_req_size  <= AXI_SZ;
                                axi_req_valid <= 1'b1;
                                state         <= S_UP_REQ_INPUT;
                            end
                        end else begin
                            inner_idx <= inner_idx + 1;
                            // Pre-set next input request
                            axi_req_addr  <= INPUT_BASE + (inner_idx + 1) * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_UP_REQ_INPUT;
                        end
                    end
                end

                // =====================================================
                // DOWN PHASE: Read ReLU tile from local BRAM
                // =====================================================
                S_DOWN_REQ_RELU: begin
                    relu_rd_addr <= inner_idx_wide[$clog2(NUM_TILES_4D)-1:0];
                    relu_rd_en   <= 1'b1;
                    state        <= S_DOWN_RELAY;
                end

                S_DOWN_RELAY: begin
                    // Wait one cycle for BRAM synchronous read to complete.
                    // rd_en was asserted in S_DOWN_REQ_RELU (takes effect at
                    // end of that cycle).  The BRAM sees rd_en at the NEXT
                    // posedge (this cycle) and latches the data into rd_reg.
                    // The data will be available in S_DOWN_WAIT_RELU.
                    state <= S_DOWN_WAIT_RELU;
                end

                S_DOWN_WAIT_RELU: begin
                    relu_tile_buf <= relu_rd_data;
                    // Pre-set down weight request
                    axi_req_addr  <= WDOWN_BASE +
                                    (inner_idx_wide * NUM_TILES_D + tile_col) * WTILE_BYTES;
                    axi_req_len   <= WEIGHT_BEATS - 1;
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state         <= S_DOWN_REQ_WGT;
                end

                // =====================================================
                // DOWN PHASE: Request down-projection weight tile
                // =====================================================
                S_DOWN_REQ_WGT: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_DOWN_WAIT_WGT;
                    end
                end

                S_DOWN_WAIT_WGT: begin
                    if (axi_resp_valid) begin
                        for (k = 0; k < WORDS_PER_BEAT; k = k + 1) begin
                            if (beat_cnt * WORDS_PER_BEAT + k < M * M) begin
                                weight_tile_buf[(beat_cnt * WORDS_PER_BEAT + k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                            end
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            state <= S_DOWN_SEND;
                        end
                    end
                end

                // =====================================================
                // DOWN PHASE: Send tile pair to down projection
                // =====================================================
                S_DOWN_SEND: begin
                    if (down_ready) begin
                        down_relu_tile   <= relu_tile_buf;
                        down_weight_tile <= weight_tile_buf;
                        down_tile_col    <= tile_col[$clog2(NUM_TILES_D)-1:0];
                        down_inner_idx   <= inner_idx_wide[$clog2(NUM_TILES_4D)-1:0];
                        down_is_last     <= (inner_idx_wide == NUM_TILES_4D - 1);
                        down_valid       <= 1'b1;

                        if (inner_idx_wide == NUM_TILES_4D - 1) begin
                            if (tile_col == NUM_TILES_D - 1) begin
                                phase_down <= 1'b0;
                                done       <= 1'b1;
                                state      <= S_DONE;
                            end else begin
                                tile_col       <= tile_col + 1;
                                inner_idx_wide <= '0;
                                state          <= S_DOWN_REQ_RELU;
                            end
                        end else begin
                            inner_idx_wide <= inner_idx_wide + 1;
                            state          <= S_DOWN_REQ_RELU;
                        end
                    end
                end

                // =====================================================
                S_DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

//============================================================================
// up_projection.v - Stage 2: Up Projection (d × 4d) - Vivado-Safe
//============================================================================
// Uses hierarchical mul_col modules instead of flat M×M multiplier array.
//
// Architecture:
//   M mul_col modules, each containing M multipliers + 1 adder tree.
//   Total: M×M multipliers, M adder trees - same as before,
//   but now Vivado sees M manageable submodules instead of 1024 leaf cells.
//
// The weight_tile input (M×M×DATA_W = 16384 bits for M=32) is split
// into M weight columns, each M×DATA_W = 512 bits.
//============================================================================

module up_projection #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [M*M*DATA_W-1:0]         weight_tile,
    input  wire [$clog2(4*D/M)-1:0]      tile_col,
    input  wire [$clog2(D/M)-1:0]        inner_idx,
    input  wire                           is_last,
    input  wire                           valid_in,
    output reg                            ready_out,

    output reg  [M*DATA_W-1:0]           result_tile,
    output reg  [$clog2(4*D/M)-1:0]      result_col,
    output reg                            valid_out,
    input  wire                           ready_in
);

    localparam PROD_W  = 2 * DATA_W;
    localparam SUM_W   = PROD_W + $clog2(M);
    localparam ACC_W   = SUM_W + $clog2(D/M) + 2;
    localparam MAX_POS = (1 << (DATA_W - 1)) - 1;
    localparam MIN_NEG = -(1 << (DATA_W - 1));
    localparam TREE_DEPTH = $clog2(M);

    integer j;
    reg signed [ACC_W-1:0] acc_val;

    // -------------------------------------------------------------------
    // M mul_col instances (hierarchical - Vivado-safe)
    // -------------------------------------------------------------------
    // Each mul_col computes: result[j] = Σ_k input_tile[k] * weight_tile[k][j]
    // Weight column j: {weight_tile[0][j], weight_tile[1][j], ..., weight_tile[M-1][j]}
    // Stored in row-major order: weight_tile[(k*M+j)*DATA_W +: DATA_W]

    wire [M*SUM_W-1:0]   col_results;
    wire [M-1:0]          col_valid;

    genvar gj;
    generate
        for (gj = 0; gj < M; gj = gj + 1) begin : gen_col

            // Extract weight column j from the flat weight tile
            // weight_tile[(k*M + j) * DATA_W +: DATA_W] for k=0..M-1
            wire [M*DATA_W-1:0] wcol;
            genvar k;
            for (k = 0; k < M; k = k + 1) begin : gen_wcol
                assign wcol[k*DATA_W +: DATA_W] =
                    weight_tile[(k*M + gj)*DATA_W +: DATA_W];
            end

            mul_col #(
                .M      (M),
                .DATA_W (DATA_W),
                .PIPE   (1)    // Pipelined adder tree
            ) u_mul_col (
                .clk        (clk),
                .rst_n      (rst_n),
                .input_tile (input_tile),
                .weight_col (wcol),
                .valid_in   (valid_in & ready_out),
                .result     (col_results[gj*SUM_W +: SUM_W]),
                .result_valid(col_valid[gj])
            );
        end
    endgenerate

    // -------------------------------------------------------------------
    // Pipeline valid and control signals through adder tree
    // -------------------------------------------------------------------
    reg [TREE_DEPTH:0] valid_pipe;
    reg [$clog2(D/M)-1:0]    inner_idx_pipe [0:TREE_DEPTH];
    reg                       is_last_pipe   [0:TREE_DEPTH];
    reg [$clog2(4*D/M)-1:0]  tile_col_pipe  [0:TREE_DEPTH];

    integer p;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= '0;
            for (p = 0; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= '0;
                is_last_pipe[p]   <= '0;
                tile_col_pipe[p]  <= '0;
            end
        end else begin
            valid_pipe[0] <= valid_in & ready_out;
            valid_pipe[TREE_DEPTH:1] <= valid_pipe[TREE_DEPTH-1:0];
            inner_idx_pipe[0] <= inner_idx;
            is_last_pipe[0]   <= is_last;
            tile_col_pipe[0]  <= tile_col;
            for (p = 1; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= inner_idx_pipe[p-1];
                is_last_pipe[p]   <= is_last_pipe[p-1];
                tile_col_pipe[p]  <= tile_col_pipe[p-1];
            end
        end
    end

    // -------------------------------------------------------------------
    // Internal Accumulator
    // -------------------------------------------------------------------
    reg [M*ACC_W-1:0]    acc;
    reg                  acc_valid;
    reg [$clog2(4*D/M)-1:0] acc_col;

    wire tree_out_valid = valid_pipe[TREE_DEPTH];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            acc         <= '0;
            acc_valid   <= 1'b0;
            acc_col     <= '0;
            result_tile <= '0;
            result_col  <= '0;
        end else begin
            valid_out <= 1'b0;

            // Accumulate adder tree outputs
            if (tree_out_valid) begin
                for (j = 0; j < M; j = j + 1) begin
                    if (inner_idx_pipe[TREE_DEPTH] == 0) begin
                        acc[j*ACC_W +: ACC_W] <=
                            {{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W - 1]}},
                             col_results[j*SUM_W +: SUM_W]};
                    end else begin
                        acc[j*ACC_W +: ACC_W] <=
                            $signed(acc[j*ACC_W +: ACC_W]) +
                            $signed({{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W - 1]}},
                                     col_results[j*SUM_W +: SUM_W]});
                    end
                end
                acc_col <= tile_col_pipe[TREE_DEPTH];

                if (is_last_pipe[TREE_DEPTH]) begin
                    acc_valid <= 1'b1;
                end
            end

            // Output with saturation
            if (acc_valid && ready_in) begin
                for (j = 0; j < M; j = j + 1) begin
                    acc_val = $signed(acc[j*ACC_W +: ACC_W]);
                    if (acc_val > MAX_POS)
                        result_tile[j*DATA_W +: DATA_W] <= MAX_POS[DATA_W-1:0];
                    else if (acc_val < MIN_NEG)
                        result_tile[j*DATA_W +: DATA_W] <= MIN_NEG[DATA_W-1:0];
                    else
                        result_tile[j*DATA_W +: DATA_W] <= acc[j*ACC_W +: DATA_W];
                end
                result_col <= acc_col;
                valid_out  <= 1'b1;
                acc_valid  <= 1'b0;
            end

            ready_out <= !(acc_valid && !ready_in);
        end
    end

endmodule

//============================================================================
// relu_stage.v - Stage 3: ReLU Activation
//============================================================================
// Applies ReLU (max(0, x)) element-wise to each element of the 1×M tile.
// All M elements are processed in parallel using M comparators.
//
// The ReLU output is written to a local BRAM (relu_bram) so the DOWN
// phase can read it later. Simultaneously, it passes data through
// boundary registers for pipelined operation.
//
// Parameters:
//   D, M, DATA_W
//============================================================================

module relu_stage #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Input from Up Projection Stage ----
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [$clog2(4*D/M)-1:0]      tile_col,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- ReLU BRAM Write Interface ----
    output reg                           relu_wr_en,
    output reg  [$clog2(4*D/M)-1:0]     relu_wr_addr,
    output reg  [M*DATA_W-1:0]          relu_wr_data,

    // ---- Output (pass-through for pipeline continuity) ----
    output reg  [M*DATA_W-1:0]          result_tile,
    output reg  [$clog2(4*D/M)-1:0]     result_col,
    output reg                           valid_out,
    input  wire                          ready_in
);

    // -------------------------------------------------------------------
    // ReLU: max(0, x) for each element - all M in parallel
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0] relu_output;

    integer j;
    always @(*) begin
        for (j = 0; j < M; j = j + 1) begin
            if (input_tile[j*DATA_W + DATA_W - 1]) begin
                // Negative → output 0
                relu_output[j*DATA_W +: DATA_W] = {DATA_W{1'b0}};
            end else begin
                // Non-negative → pass through
                relu_output[j*DATA_W +: DATA_W] = input_tile[j*DATA_W +: DATA_W];
            end
        end
    end

    // -------------------------------------------------------------------
    // Pipeline register + BRAM write
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            result_tile <= '0;
            result_col  <= '0;
            relu_wr_en  <= 1'b0;
            relu_wr_addr<= '0;
            relu_wr_data<= '0;
        end else begin
            // Default: deassert one-shot signals
            valid_out  <= 1'b0;
            relu_wr_en <= 1'b0;

            if (valid_in && ready_out) begin
                // Latch ReLU output into boundary register
                result_tile <= relu_output;
                result_col  <= tile_col;
                valid_out   <= 1'b1;

                // Write to ReLU BRAM simultaneously
                relu_wr_en   <= 1'b1;
                relu_wr_addr <= tile_col;
                relu_wr_data <= relu_output;
            end

            // Backpressure handling
            ready_out <= 1'b1;  // ReLU is always ready (combinational, no backpressure needed internally)
        end
    end

endmodule

//============================================================================
// down_projection.v - Stage 4: Down Projection (4d × d) - Vivado-Safe
//============================================================================
// Same hierarchical structure as up_projection using mul_col modules.
//============================================================================

module down_projection #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire [M*DATA_W-1:0]           relu_tile,
    input  wire [M*M*DATA_W-1:0]         weight_tile,
    input  wire [$clog2(D/M)-1:0]        tile_col,
    input  wire [$clog2(4*D/M)-1:0]      inner_idx,
    input  wire                           is_last,
    input  wire                           valid_in,
    output reg                            ready_out,

    output reg  [M*DATA_W-1:0]           result_tile,
    output reg  [$clog2(D/M)-1:0]        result_col,
    output reg                            valid_out,
    input  wire                           ready_in
);

    localparam PROD_W  = 2 * DATA_W;
    localparam SUM_W   = PROD_W + $clog2(M);
    localparam ACC_W   = SUM_W + $clog2(4*D/M) + 2;
    localparam MAX_POS = (1 << (DATA_W - 1)) - 1;
    localparam MIN_NEG = -(1 << (DATA_W - 1));
    localparam TREE_DEPTH = $clog2(M);

    integer j;
    reg signed [ACC_W-1:0] acc_val;

    // -------------------------------------------------------------------
    // M mul_col instances (hierarchical - Vivado-safe)
    // -------------------------------------------------------------------
    wire [M*SUM_W-1:0]   col_results;
    wire [M-1:0]          col_valid;

    genvar gj;
    generate
        for (gj = 0; gj < M; gj = gj + 1) begin : gen_col
            wire [M*DATA_W-1:0] wcol;
            genvar k;
            for (k = 0; k < M; k = k + 1) begin : gen_wcol
                assign wcol[k*DATA_W +: DATA_W] =
                    weight_tile[(k*M + gj)*DATA_W +: DATA_W];
            end

            mul_col #(
                .M      (M),
                .DATA_W (DATA_W),
                .PIPE   (1)
            ) u_mul_col (
                .clk        (clk),
                .rst_n      (rst_n),
                .input_tile (relu_tile),
                .weight_col (wcol),
                .valid_in   (valid_in & ready_out),
                .result     (col_results[gj*SUM_W +: SUM_W]),
                .result_valid(col_valid[gj])
            );
        end
    endgenerate

    // -------------------------------------------------------------------
    // Pipeline valid and control signals
    // -------------------------------------------------------------------
    reg [TREE_DEPTH:0] valid_pipe;
    reg [$clog2(4*D/M)-1:0] inner_idx_pipe [0:TREE_DEPTH];
    reg                       is_last_pipe   [0:TREE_DEPTH];
    reg [$clog2(D/M)-1:0]    tile_col_pipe  [0:TREE_DEPTH];

    integer p;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= '0;
            for (p = 0; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= '0;
                is_last_pipe[p]   <= '0;
                tile_col_pipe[p]  <= '0;
            end
        end else begin
            valid_pipe[0] <= valid_in & ready_out;
            valid_pipe[TREE_DEPTH:1] <= valid_pipe[TREE_DEPTH-1:0];
            inner_idx_pipe[0] <= inner_idx;
            is_last_pipe[0]   <= is_last;
            tile_col_pipe[0]  <= tile_col;
            for (p = 1; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= inner_idx_pipe[p-1];
                is_last_pipe[p]   <= is_last_pipe[p-1];
                tile_col_pipe[p]  <= tile_col_pipe[p-1];
            end
        end
    end

    // -------------------------------------------------------------------
    // Internal Accumulator
    // -------------------------------------------------------------------
    reg [M*ACC_W-1:0]        acc;
    reg                      acc_valid;
    reg [$clog2(D/M)-1:0]    acc_col;

    wire tree_out_valid = valid_pipe[TREE_DEPTH];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            acc         <= '0;
            acc_valid   <= 1'b0;
            acc_col     <= '0;
            result_tile <= '0;
            result_col  <= '0;
        end else begin
            valid_out <= 1'b0;

            if (tree_out_valid) begin
                for (j = 0; j < M; j = j + 1) begin
                    if (inner_idx_pipe[TREE_DEPTH] == 0) begin
                        acc[j*ACC_W +: ACC_W] <=
                            {{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W - 1]}},
                             col_results[j*SUM_W +: SUM_W]};
                    end else begin
                        acc[j*ACC_W +: ACC_W] <=
                            $signed(acc[j*ACC_W +: ACC_W]) +
                            $signed({{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W - 1]}},
                                     col_results[j*SUM_W +: SUM_W]});
                    end
                end
                acc_col <= tile_col_pipe[TREE_DEPTH];

                if (is_last_pipe[TREE_DEPTH]) begin
                    acc_valid <= 1'b1;
                end
            end

            if (acc_valid && ready_in) begin
                for (j = 0; j < M; j = j + 1) begin
                    acc_val = $signed(acc[j*ACC_W +: ACC_W]);
                    if (acc_val > MAX_POS)
                        result_tile[j*DATA_W +: DATA_W] <= MAX_POS[DATA_W-1:0];
                    else if (acc_val < MIN_NEG)
                        result_tile[j*DATA_W +: DATA_W] <= MIN_NEG[DATA_W-1:0];
                    else
                        result_tile[j*DATA_W +: DATA_W] <= acc[j*ACC_W +: DATA_W];
                end
                result_col <= acc_col;
                valid_out  <= 1'b1;
                acc_valid  <= 1'b0;
            end

            ready_out <= !(acc_valid && !ready_in);
        end
    end

endmodule

//============================================================================
// accumulator.v - Stage 5: Output Accumulation (REVISED v2)
//============================================================================
// Receives completed 1×M tiles from the Down Projection stage.
// Decodes the tile_col index to determine where in the 1×D output
// vector each tile should be written.
//
// After all tiles are written, a readout FSM streams them out through
// output_tile_data / output_tile_addr / output_tile_valid with proper
// BRAM latency alignment.
//
// Timing analysis (critical path):
//
//   Because out_rd_en and out_rd_addr are registered outputs (non-blocking
//   in this always block), and the BRAM also has a registered read:
//
//   Cycle N   : FSM in RD_ISSUE  → sets out_rd_en<=1, out_rd_addr<=idx
//   Cycle N+1 : out_rd_en=1 reaches BRAM → BRAM latches mem[idx] into rd_reg
//   Cycle N+2 : out_rd_data = mem[idx]  → FSM in RD_CAPTURE reads it
//
//   Total: 2 cycles from RD_ISSUE to valid data.
//   FSM path: RD_ISSUE → RD_RELAY → RD_CAPTURE → (next RD_ISSUE or RD_DONE)
//
//============================================================================

module accumulator #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16,
    parameter DISABLE_READOUT = 0   // 1 = disable internal readout FSM;
                                    //     external module controls BRAM reads
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output reg                           done,

    // ---- Input from Down Projection Stage ----
    input  wire [M*DATA_W-1:0]           result_tile,
    input  wire [$clog2(D/M)-1:0]        result_col,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- Output BRAM Write Interface ----
    output reg                           out_wr_en,
    output reg  [$clog2(D/M)-1:0]       out_wr_addr,
    output reg  [M*DATA_W-1:0]          out_wr_data,

    // ---- Output BRAM Read Interface ----
    output reg                           out_rd_en,
    output reg  [$clog2(D/M)-1:0]       out_rd_addr,
    input  wire [M*DATA_W-1:0]          out_rd_data,

    // ---- Streaming output port (latency-aligned) ----
    output reg  [M*DATA_W-1:0]          output_tile_data,
    output reg  [$clog2(D/M)-1:0]       output_tile_addr,
    output reg                           output_tile_valid,

    // ---- Legacy single-element output (kept for compatibility) ----
    output reg  [DATA_W-1:0]            output_data,
    output reg                           output_valid
);

    localparam NUM_TILES = D / M;

    // -------------------------------------------------------------------
    // Write-side state
    // -------------------------------------------------------------------
    reg [$clog2(NUM_TILES)-1:0] tile_cnt;

    // -------------------------------------------------------------------
    // Readout FSM states
    // -------------------------------------------------------------------
    localparam RD_IDLE    = 3'd0;
    localparam RD_ISSUE   = 3'd1;  // Assert rd_en for current index (registered)
    localparam RD_RELAY   = 3'd2;  // Wait for registered rd_en to reach BRAM
    localparam RD_CAPTURE = 3'd3;  // BRAM data is now valid - capture & output
    localparam RD_DONE    = 3'd4;  // All tiles streamed out

    reg [2:0] rd_state;
    reg [$clog2(NUM_TILES)-1:0] rd_idx;
    reg [$clog2(NUM_TILES)-1:0] rd_idx_pipe;  // Pipelined address (aligns with data)
    reg        rd_done_flag;

    // -------------------------------------------------------------------
    // Main sequential logic
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out       <= 1'b1;
            done            <= 1'b0;
            tile_cnt        <= 0;
            out_wr_en       <= 1'b0;
            out_wr_addr     <= 0;
            out_wr_data     <= 0;
            out_rd_en       <= 1'b0;
            out_rd_addr     <= 0;
            output_tile_data  <= 0;
            output_tile_addr  <= 0;
            output_tile_valid <= 1'b0;
            output_data     <= 0;
            output_valid    <= 1'b0;
            rd_state        <= RD_IDLE;
            rd_idx          <= 0;
            rd_idx_pipe     <= 0;
            rd_done_flag    <= 1'b0;
        end else begin
            // Defaults: one-cycle pulses
            out_wr_en        <= 1'b0;
            out_rd_en        <= 1'b0;
            output_tile_valid <= 1'b0;
            output_valid     <= 1'b0;

            // -----------------------------------------------------------
            // Reset on new start
            // -----------------------------------------------------------
            if (start) begin
                done         <= 1'b0;
                tile_cnt     <= 0;
                rd_state     <= RD_IDLE;
                rd_done_flag <= 1'b0;
            end

            // -----------------------------------------------------------
            // Write path: accept result tile from down projection
            // -----------------------------------------------------------
            if (valid_in && ready_out) begin
                out_wr_en   <= 1'b1;
                out_wr_addr <= result_col;
                out_wr_data <= result_tile;

                tile_cnt <= tile_cnt + 1;

                if (tile_cnt == NUM_TILES - 1) begin
                    done <= 1'b1;
                end
            end

            // -----------------------------------------------------------
            // Readout FSM: stream output tiles after computation done
            // -----------------------------------------------------------
            case (rd_state)
                RD_IDLE: begin
                    if (!DISABLE_READOUT && done && !rd_done_flag) begin
                        rd_idx    <= 0;
                        rd_state  <= RD_ISSUE;
                    end
                end

                RD_ISSUE: begin
                    // Issue BRAM read for rd_idx.
                    // out_rd_en and out_rd_addr are non-blocking, so they
                    // take effect at the END of this cycle.
                    out_rd_en   <= 1'b1;
                    out_rd_addr <= rd_idx;
                    // Pipeline the address to align with data arrival
                    rd_idx_pipe <= rd_idx;
                    rd_state    <= RD_RELAY;
                end

                RD_RELAY: begin
                    // Cycle after RD_ISSUE: out_rd_en is now 1 at the BRAM
                    // input. The BRAM latches mem[rd_idx] into rd_reg at the
                    // end of THIS cycle. Data will be on out_rd_data next cycle.
                    // Deassert rd_en (already default 0).
                    rd_state <= RD_CAPTURE;
                end

                RD_CAPTURE: begin
                    // out_rd_data now holds mem[rd_idx_pipe].
                    // Capture and present on the output port.
                    output_tile_data  <= out_rd_data;
                    output_tile_addr  <= rd_idx_pipe;
                    output_tile_valid <= 1'b1;
                    // Legacy single-element output
                    output_data  <= out_rd_data[0*DATA_W +: DATA_W];
                    output_valid <= 1'b1;

                    if (rd_idx < NUM_TILES - 1) begin
                        rd_idx   <= rd_idx + 1;
                        rd_state <= RD_ISSUE;
                    end else begin
                        rd_done_flag <= 1'b1;
                        rd_state     <= RD_DONE;
                    end
                end

                RD_DONE: begin
                    // Stay here until next start resets us
                    rd_done_flag <= 1'b1;
                end

                default: rd_state <= RD_IDLE;
            endcase

            ready_out <= 1'b1;
        end
    end

endmodule

//============================================================================
// ffn_top_zynq.v - Zynq-7000 Optimized FFN Top Module
//============================================================================
// Optimized for xc7z045ffv900-1 (900 DSP, 218K LUT, 362 IO, 545 BRAM36)
//
// Key optimizations vs ffn_top_2048:
//   1. TIME-MULTIPLEXED PROJECTION: NUM_COLS=14 instead of M=32
//      → DSP: 896 (fits!) vs 2048 (overflows to LUTs)
//      → LUT: ~80K (fits!) vs 449K (overflows)
//   2. NARROW AXI BUS: AXI_DATA_W=64 instead of 512
//      → IOB: ~189 pins (fits!) vs 1085 (overflows)
//   3. NARROW OUTPUT: 64-bit streamed output instead of 512-bit parallel
//      → IOB: further reduction
//
// Output serializer:
//   After computation completes (done=1), the serializer reads each 1×M
//   tile from the output BRAM and streams it as AXI_DATA_W-bit beats.
//   BEATS_PER_TILE = M / (AXI_DATA_W / DATA_W) beats per tile.
//   The accumulator's internal readout FSM is disabled (DISABLE_READOUT=1)
//   and the serializer controls the output BRAM read port directly.
//
// Resource estimates:
//   DSP48E1  : 896  (2 stages × 14 cols × 32 muls)
//   LUT      : ~80K (adder trees, accumulators, control, MUXes)
//   FF       : ~40K (pipeline regs, accumulators, state machines)
//   BRAM36   : ~30  (ReLU BRAM + Output BRAM + weight tile buffers)
//   IOB      : ~189 (64-bit AXI + 64-bit output + control)
//============================================================================

module ffn_top_zynq #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter NUM_COLS   = 14,
    parameter DATA_W     = 16,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output wire                          done,

    // ---- AXI4 Read Master ----
    output wire [AXI_ADDR_W-1:0]        araddr,
    output wire [7:0]                   arlen,
    output wire [2:0]                   arsize,
    output wire [1:0]                   arburst,
    output wire                          arvalid,
    input  wire                          arready,

    input  wire [AXI_DATA_W-1:0]        rdata,
    input  wire                          rlast,
    input  wire                          rvalid,
    output wire                          rready,

    // ---- Streamed Output ----
    output wire [AXI_DATA_W-1:0]        out_data,
    output wire [$clog2(D/M)-1:0]       out_addr,      // Tile address
    output wire [2:0]                   out_offset,     // Beat offset within tile (0..7)
    output wire                          out_valid,      // Beat is valid
    output wire                          out_last        // Last beat of last tile
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;
    localparam MAX_INNER    = (NUM_TILES_4D > NUM_TILES_D) ? NUM_TILES_4D : NUM_TILES_D;

    localparam ELEMS_PER_BEAT = AXI_DATA_W / DATA_W;   // Elements per AXI beat
    localparam BEATS_PER_TILE = M / ELEMS_PER_BEAT;     // Beats per tile

    // -------------------------------------------------------------------
    // Internal wires - Fetch ↔ Pipeline Stages
    // -------------------------------------------------------------------
    wire [M*DATA_W-1:0]           up_input_tile;
    wire [M*M*DATA_W-1:0]         up_weight_tile;
    wire [$clog2(MAX_INNER)-1:0]  up_tile_col;
    wire [$clog2(MAX_INNER)-1:0]  up_inner_idx;
    wire                           up_is_last;
    wire                           up_valid;
    wire                           up_ready;

    wire [M*DATA_W-1:0]           relu_input_tile;
    wire [$clog2(MAX_INNER)-1:0]  relu_tile_col;
    wire                           relu_valid_in;
    wire                           relu_ready_out;

    wire                           relu_wr_en;
    wire [$clog2(NUM_TILES_4D)-1:0] relu_wr_addr;
    wire [M*DATA_W-1:0]           relu_wr_data;

    wire                           relu_rd_en;
    wire [$clog2(NUM_TILES_4D)-1:0] relu_rd_addr;
    wire [M*DATA_W-1:0]           relu_rd_data;

    wire [M*DATA_W-1:0]           down_relu_tile;
    wire [M*M*DATA_W-1:0]         down_weight_tile;
    wire [$clog2(MAX_INNER)-1:0]  down_tile_col;
    wire [$clog2(MAX_INNER)-1:0]  down_inner_idx;
    wire                           down_is_last;
    wire                           down_valid;
    wire                           down_ready;

    wire [M*DATA_W-1:0]           acc_result_tile;
    wire [$clog2(MAX_INNER)-1:0]  acc_result_col;
    wire                           acc_valid_in;
    wire                           acc_ready_out;

    // Accumulator BRAM write interface
    wire                           out_wr_en;
    wire [$clog2(NUM_TILES_D)-1:0] out_wr_addr;
    wire [M*DATA_W-1:0]           out_wr_data;

    // Output BRAM read interface - controlled by serializer
    wire                           out_rd_en;
    wire [$clog2(NUM_TILES_D)-1:0] out_rd_addr;
    wire [M*DATA_W-1:0]           out_rd_data;

    // AXI request/response
    wire [AXI_ADDR_W-1:0]        axi_req_addr;
    wire [7:0]                   axi_req_len;
    wire [2:0]                   axi_req_size;
    wire                          axi_req_valid;
    wire                          axi_req_ready;
    wire [AXI_DATA_W-1:0]        axi_resp_data;
    wire                          axi_resp_last;
    wire                          axi_resp_valid;
    wire                          axi_resp_ready;

    wire                          phase_up;
    wire                          phase_down;

    // ===================================================================
    // AXI Read Master
    // ===================================================================
    axi_read_master #(
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_axi_read_master (
        .clk        (clk),
        .rst_n      (rst_n),
        .req_addr   (axi_req_addr),
        .req_len    (axi_req_len),
        .req_size   (axi_req_size),
        .req_valid  (axi_req_valid),
        .req_ready  (axi_req_ready),
        .resp_data  (axi_resp_data),
        .resp_last  (axi_resp_last),
        .resp_valid (axi_resp_valid),
        .resp_ready (axi_resp_ready),
        .araddr     (araddr),
        .arlen      (arlen),
        .arsize     (arsize),
        .arburst    (arburst),
        .arvalid    (arvalid),
        .arready    (arready),
        .rdata      (rdata),
        .rlast      (rlast),
        .rvalid     (rvalid),
        .rready     (rready)
    );

    // ===================================================================
    // Stage 1: Fetch & Address Generation
    // ===================================================================
    fetch_addr_gen #(
        .D          (D),
        .M          (M),
        .DATA_W     (DATA_W),
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_fetch_addr_gen (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .done           (),
        .phase_up       (phase_up),
        .phase_down     (phase_down),
        .axi_req_addr   (axi_req_addr),
        .axi_req_len    (axi_req_len),
        .axi_req_size   (axi_req_size),
        .axi_req_valid  (axi_req_valid),
        .axi_req_ready  (axi_req_ready),
        .axi_resp_data  (axi_resp_data),
        .axi_resp_last  (axi_resp_last),
        .axi_resp_valid (axi_resp_valid),
        .axi_resp_ready (axi_resp_ready),
        .relu_rd_en     (relu_rd_en),
        .relu_rd_addr   (relu_rd_addr),
        .relu_rd_data   (relu_rd_data),
        .up_input_tile  (up_input_tile),
        .up_weight_tile (up_weight_tile),
        .up_tile_col    (up_tile_col),
        .up_inner_idx   (up_inner_idx),
        .up_is_last     (up_is_last),
        .up_valid       (up_valid),
        .up_ready       (up_ready),
        .down_relu_tile   (down_relu_tile),
        .down_weight_tile (down_weight_tile),
        .down_tile_col    (down_tile_col),
        .down_inner_idx   (down_inner_idx),
        .down_is_last     (down_is_last),
        .down_valid       (down_valid),
        .down_ready       (down_ready)
    );

    // ===================================================================
    // Stage 2: Up Projection - Time-Multiplexed
    // ===================================================================
    tm_proj_stage #(
        .D         (D),
        .M         (M),
        .NUM_COLS  (NUM_COLS),
        .MAX_INNER (MAX_INNER),     // FIX: use max(NUM_TILES_D, NUM_TILES_4D)
        .DATA_W    (DATA_W)
    ) u_up_projection (
        .clk        (clk),
        .rst_n      (rst_n),
        .input_tile (up_input_tile),
        .weight_tile(up_weight_tile),
        .tile_col   (up_tile_col),
        .inner_idx  (up_inner_idx),
        .is_last    (up_is_last),
        .valid_in   (up_valid),
        .ready_out  (up_ready),
        .result_tile(relu_input_tile),
        .result_col (relu_tile_col),
        .valid_out  (relu_valid_in),
        .ready_in   (relu_ready_out)
    );

    // ===================================================================
    // Stage 3: ReLU
    // ===================================================================
    relu_stage #(
        .D      (D),
        .M      (M),
        .DATA_W (DATA_W)
    ) u_relu_stage (
        .clk        (clk),
        .rst_n      (rst_n),
        .input_tile (relu_input_tile),
        .tile_col   (relu_tile_col),
        .valid_in   (relu_valid_in),
        .ready_out  (relu_ready_out),
        .relu_wr_en   (relu_wr_en),
        .relu_wr_addr (relu_wr_addr),
        .relu_wr_data (relu_wr_data),
        .result_tile (),
        .result_col  (),
        .valid_out   (),
        .ready_in    (1'b1)
    );

    // ===================================================================
    // ReLU BRAM
    // ===================================================================
    bram_dp_wide #(
        .DATA_W    (DATA_W),
        .TILE_SIZE (M),
        .DEPTH     (NUM_TILES_4D)
    ) u_relu_bram (
        .clk      (clk),
        .wr_en    (relu_wr_en),
        .wr_addr  (relu_wr_addr),
        .wr_data  (relu_wr_data),
        .rd_en    (relu_rd_en),
        .rd_addr  (relu_rd_addr),
        .rd_data  (relu_rd_data)
    );

    // ===================================================================
    // Stage 4: Down Projection - Time-Multiplexed
    // ===================================================================
    tm_proj_stage #(
        .D         (D),
        .M         (M),
        .NUM_COLS  (NUM_COLS),
        .MAX_INNER (MAX_INNER),     // FIX: use max(NUM_TILES_D, NUM_TILES_4D)
        .DATA_W    (DATA_W)
    ) u_down_projection (
        .clk        (clk),
        .rst_n      (rst_n),
        .input_tile (down_relu_tile),
        .weight_tile(down_weight_tile),
        .tile_col   (down_tile_col),
        .inner_idx  (down_inner_idx),
        .is_last    (down_is_last),
        .valid_in   (down_valid),
        .ready_out  (down_ready),
        .result_tile(acc_result_tile),
        .result_col (acc_result_col),
        .valid_out  (acc_valid_in),
        .ready_in   (acc_ready_out)
    );

    // ===================================================================
    // Stage 5: Accumulator (readout FSM DISABLED - serializer controls BRAM)
    // ===================================================================
    accumulator #(
        .D              (D),
        .M              (M),
        .DATA_W         (DATA_W),
        .DISABLE_READOUT(1)        // Serializer controls output BRAM reads
    ) u_accumulator (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .done        (done),
        .result_tile (acc_result_tile),
        .result_col  (acc_result_col),
        .valid_in    (acc_valid_in),
        .ready_out   (acc_ready_out),
        .out_wr_en   (out_wr_en),
        .out_wr_addr (out_wr_addr),
        .out_wr_data (out_wr_data),
        // Read port - NOT connected to BRAM (serializer controls reads)
        .out_rd_en   (),
        .out_rd_addr (),
        .out_rd_data (out_rd_data),  // from BRAM, but unused by accumulator
        // Streaming outputs - unused (serializer provides them)
        .output_tile_data  (),
        .output_tile_addr  (),
        .output_tile_valid (),
        .output_data (),
        .output_valid()
    );

    // ===================================================================
    // Output BRAM
    // ===================================================================
    bram_dp_wide #(
        .DATA_W    (DATA_W),
        .TILE_SIZE (M),
        .DEPTH     (NUM_TILES_D)
    ) u_output_bram (
        .clk      (clk),
        .wr_en    (out_wr_en),
        .wr_addr  (out_wr_addr),
        .wr_data  (out_wr_data),
        .rd_en    (out_rd_en),
        .rd_addr  (out_rd_addr),
        .rd_data  (out_rd_data)
    );

    // ===================================================================
    // Output Serializer: M×DATA_W-bit BRAM tile → AXI_DATA_W-bit beats
    //
    // Reads each output tile from BRAM and streams it as BEATS_PER_TILE
    // beats of AXI_DATA_W bits each.
    //
    // BRAM read timing (synchronous read, 1-cycle latency):
    //   Cycle N  : serializer sets rd_en<=1, rd_addr<=idx (non-blocking)
    //   Cycle N+1: BRAM sees rd_en=1, latches mem[idx] into rd_reg
    //   Cycle N+2: out_rd_data = mem[idx] → capture in SER_CAPTURE
    //
    // FSM states:
    //   SER_IDLE    - Wait for done=1 from accumulator
    //   SER_WAIT    - BRAM read in progress (1 cycle)
    //   SER_CAPTURE - BRAM data available, capture into ser_tile_data
    //   SER_STREAM  - Output BEATS_PER_TILE beats of AXI_DATA_W bits
    //   SER_DONE    - All tiles streamed
    //======================================================================

    localparam SER_IDLE    = 3'd0;
    localparam SER_WAIT    = 3'd1;
    localparam SER_CAPTURE = 3'd2;
    localparam SER_STREAM  = 3'd3;
    localparam SER_DONE    = 3'd4;

    reg [2:0]                                 ser_state;
    reg [$clog2(NUM_TILES_D)-1:0]            ser_tile_idx;
    reg [2:0]                                 ser_beat_cnt;   // 3 bits: up to 8 beats/tile
    reg [M*DATA_W-1:0]                       ser_tile_data;
    reg                                       ser_bram_rd_en;
    reg [$clog2(NUM_TILES_D)-1:0]            ser_bram_rd_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ser_state       <= SER_IDLE;
            ser_tile_idx    <= '0;
            ser_beat_cnt    <= '0;
            ser_tile_data   <= '0;
            ser_bram_rd_en  <= 1'b0;
            ser_bram_rd_addr<= '0;
        end else begin
            ser_bram_rd_en <= 1'b0;    // default: one-cycle pulse

            // Reset serializer on new start
            if (start) begin
                ser_state    <= SER_IDLE;
            end else begin
                case (ser_state)
                    //================================================
                    SER_IDLE: begin
                        if (done) begin
                            // Computation complete - issue first BRAM read
                            ser_tile_idx    <= 0;
                            ser_bram_rd_en  <= 1'b1;
                            ser_bram_rd_addr<= 0;
                            ser_state       <= SER_WAIT;
                        end
                    end

                    //================================================
                    SER_WAIT: begin
                        // BRAM read in progress; data arrives next cycle
                        ser_state <= SER_CAPTURE;
                    end

                    //================================================
                    SER_CAPTURE: begin
                        // BRAM data is now valid on out_rd_data
                        ser_tile_data <= out_rd_data;
                        ser_beat_cnt  <= 0;
                        ser_state     <= SER_STREAM;
                    end

                    //================================================
                    SER_STREAM: begin
                        // Output current beat (out_valid=1 via assign)
                        if (ser_beat_cnt == BEATS_PER_TILE - 1) begin
                            // Last beat of this tile
                            if (ser_tile_idx == NUM_TILES_D - 1) begin
                                // All tiles streamed
                                ser_state <= SER_DONE;
                            end else begin
                                // Issue read for next tile
                                ser_tile_idx    <= ser_tile_idx + 1;
                                ser_bram_rd_en  <= 1'b1;
                                ser_bram_rd_addr<= ser_tile_idx + 1;
                                ser_state       <= SER_WAIT;
                            end
                        end else begin
                            ser_beat_cnt <= ser_beat_cnt + 1;
                        end
                    end

                    //================================================
                    SER_DONE: begin
                        // Stay here until start resets
                        ser_state <= SER_DONE;
                    end

                    //================================================
                    default: ser_state <= SER_IDLE;
                endcase
            end
        end
    end

    // -------------------------------------------------------------------
    // Serializer drives BRAM read port
    // -------------------------------------------------------------------
    assign out_rd_en   = ser_bram_rd_en;
    assign out_rd_addr = ser_bram_rd_addr;

    // -------------------------------------------------------------------
    // Output port assignments
    // -------------------------------------------------------------------
    // Select AXI_DATA_W bits from the captured tile based on beat count
    assign out_data   = ser_tile_data[ser_beat_cnt * AXI_DATA_W +: AXI_DATA_W];
    assign out_addr   = ser_tile_idx;
    assign out_offset = ser_beat_cnt;
    assign out_valid  = (ser_state == SER_STREAM);
    assign out_last   = (ser_state == SER_STREAM) &&
                        (ser_beat_cnt == BEATS_PER_TILE - 1) &&
                        (ser_tile_idx == NUM_TILES_D - 1);

endmodule
