//============================================================================
// tm_proj_stage.v — Time-Multiplexed Projection Stage
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
    parameter DATA_W    = 16,
    parameter FRAC_W    = 0         // 0 = plain integer, 8 = Q8.8 fixed-point
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Input from Fetch Stage ----
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [M*M*DATA_W-1:0]         weight_tile,
    input  wire [`CLOG2_MIN1(MAX_INNER)-1:0]   tile_col,
    input  wire [`CLOG2_MIN1(MAX_INNER)-1:0]   inner_idx,
    input  wire                           is_last,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- Output ----
    output reg  [M*DATA_W-1:0]           result_tile,
    output reg  [`CLOG2_MIN1(MAX_INNER)-1:0]   result_col,
    output reg                            valid_out,
    input  wire                           ready_in
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_SUB    = (M + NUM_COLS - 1) / NUM_COLS;  // Sub-cycles per iteration
    localparam TREE_DEPTH = `CLOG2_MIN1(M);
    localparam PROD_W     = 2 * DATA_W;
    localparam SUM_W      = PROD_W + `CLOG2_MIN1(M);
    localparam ACC_W      = SUM_W + `CLOG2_MIN1(MAX_INNER) + 2;
    // -------------------------------------------------------------------
    // Fixed-point output extraction and saturation
    // When FRAC_W > 0, the accumulator holds Q8.8 products accumulated
    // with additional integer bits from inner accumulation. The Q8.8 result
    // occupies bits [FRAC_W+DATA_W-1 : FRAC_W] of the ACC_W-wide value.
    // Saturation must compare the full ACC_W value against wide thresholds.
    //
    // When FRAC_W = 0 (plain integer mode), behavior is unchanged:
    //   MAX_THRESH = MAX_POS, MIN_THRESH = MIN_NEG, extraction = bottom DATA_W bits
    //
    // When FRAC_W = 8 (Q8.8 mode):
    //   MAX_THRESH = 8388351 (32767 << 8), extraction = bits [23:8]
    // -------------------------------------------------------------------
    localparam signed [2*DATA_W-1:0] MAX_THRESH = (1 << (DATA_W-1+FRAC_W)) - 1;
    localparam signed [2*DATA_W-1:0] MIN_THRESH = -(1 << (DATA_W-1+FRAC_W));
    localparam signed [DATA_W-1:0]   MAX_OUT    = (1 << (DATA_W-1)) - 1;
    localparam signed [DATA_W-1:0]   MIN_OUT    = -(1 << (DATA_W-1));

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE   = 3'd0;  // Waiting for input
    localparam S_FEED   = 3'd1;  // Feed column group to mul_cols
    localparam S_PIPE   = 3'd2;  // Wait for adder tree pipeline
    localparam S_ACC    = 3'd3;  // Accumulate results
    localparam S_OUT    = 3'd4;  // Output final result

    reg [2:0]            state;
    reg [`CLOG2_MIN1(NUM_SUB)-1:0] sub_cycle;
    reg [TREE_DEPTH:0]   pipe_cnt;

    // -------------------------------------------------------------------
    // Latched inputs (held stable across sub-cycles)
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0]          input_tile_r;
    reg [M*M*DATA_W-1:0]        weight_tile_r;
    reg [`CLOG2_MIN1(MAX_INNER)-1:0]  tile_col_r;
    reg [`CLOG2_MIN1(MAX_INNER)-1:0]  inner_idx_r;
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
    reg [`CLOG2_MIN1(MAX_INNER)-1:0] acc_col;
    reg signed [ACC_W-1:0] acc_val;

    integer j;
    // Number of active columns in current sub-cycle
    reg [`CLOG2_MIN1(NUM_COLS)-1:0] active_cols;

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
                        active_cols <= (M < NUM_COLS) ? M[`CLOG2_MIN1(NUM_COLS)-1:0] :
                                       NUM_COLS[`CLOG2_MIN1(NUM_COLS)-1:0];
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
                            active_cols <= NUM_COLS[`CLOG2_MIN1(NUM_COLS)-1:0];
                        state <= S_FEED;
                    end
                end

                // =================================================
                S_OUT: begin
                    if (ready_in) begin
                        // Output with saturation using Q8.8-aware extraction
                        for (j = 0; j < M; j = j + 1) begin
                            acc_val = $signed(acc[j*ACC_W +: ACC_W]);
                            if (acc_val > MAX_THRESH)
                                result_tile[j*DATA_W +: DATA_W] <= MAX_OUT[DATA_W-1:0];
                            else if (acc_val < MIN_THRESH)
                                result_tile[j*DATA_W +: DATA_W] <= MIN_OUT[DATA_W-1:0];
                            else
                                // Extract Q8.8 result: bits [FRAC_W+DATA_W-1 : FRAC_W]
                                result_tile[j*DATA_W +: DATA_W] <= acc[(j*ACC_W + FRAC_W) +: DATA_W];
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
