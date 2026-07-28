//============================================================================
// tm_proj_stage.v — Time-Multiplexed Tile MatMul Projection
//============================================================================
// Computes result[tile_col] = Σ_inner input_tile × weight_tile over inner
// iterations, processing NUM_COLS columns per sub-cycle.  Result is
// accumulated across inner iterations and emitted as a complete 1×M Q8.8
// tile with saturation when is_last is seen.
//
// FIXES vs original:
//   1. Width-safe multipliers (operands sign-extended before multiply).
//   2. PIPE_WAIT = TREE_DEPTH + 1 to match adder_tree valid latency.
//   3. Saturation thresholds expressed as ACC_W-wide constants so they
//      compare correctly against the wide accumulator.
//   4. Proper backpressure via ready_out only 1 in IDLE.
//============================================================================

module tm_proj_stage #(
    parameter D         = 2048,
    parameter M         = 32,
    parameter NUM_COLS  = 14,
    parameter MAX_INNER = 256,
    parameter DATA_W    = 16,
    parameter FRAC_W    = 8     // 0 = integer, 8 = Q8.8
)(
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire [M*DATA_W-1:0]             input_tile,
    input  wire [M*M*DATA_W-1:0]           weight_tile,
    input  wire [$clog2(MAX_INNER)-1:0]    tile_col,
    input  wire [$clog2(MAX_INNER)-1:0]    inner_idx,
    input  wire                            is_last,
    input  wire                            valid_in,
    output reg                             ready_out,
    output reg  [M*DATA_W-1:0]             result_tile,
    output reg  [$clog2(MAX_INNER)-1:0]    result_col,
    output reg                             valid_out,
    input  wire                            ready_in
);

    localparam NUM_SUB    = (M + NUM_COLS - 1) / NUM_COLS;
    localparam TREE_DEPTH = $clog2(M);
    localparam PROD_W     = 2*DATA_W;
    localparam SUM_W      = PROD_W + $clog2(M);
    localparam ACC_W      = SUM_W + $clog2(MAX_INNER) + 2;

    // Saturation thresholds in ACC_W to compare against accumulator directly.
    // Output range is ± (2^(DATA_W-1) - 1) in QFRAC_W, which in accumulator
    // Q(SUM_W integer + extra).FRAC_W lands at bits [FRAC_W+DATA_W-1 : FRAC_W].
    localparam signed [ACC_W-1:0] MAX_ACC =
        (1 << (DATA_W-1+FRAC_W)) - 1;   // = 0x007FFF00 >> 0
    localparam signed [ACC_W-1:0] MIN_ACC =
        -(1 << (DATA_W-1+FRAC_W));      // = -0x00800000
    localparam signed [DATA_W-1:0] MAX_OUT = (1 << (DATA_W-1)) - 1;
    localparam signed [DATA_W-1:0] MIN_OUT = -(1 << (DATA_W-1));

    localparam S_IDLE = 3'd0, S_FEED = 3'd1, S_PIPE = 3'd2,
               S_ACC  = 3'd3, S_OUT  = 3'd4;
    reg [2:0] state;
    reg [$clog2(NUM_SUB)-1:0] sub_cycle;
    reg [TREE_DEPTH:0]        pipe_cnt;

    // Latched inputs
    reg [M*DATA_W-1:0]           input_tile_r;
    reg [M*M*DATA_W-1:0]         weight_tile_r;
    reg [$clog2(MAX_INNER)-1:0]  tile_col_r, inner_idx_r;
    reg                          is_last_r;

    // NUM_COLS multiply columns
    wire [NUM_COLS*SUM_W-1:0]  col_results;
    wire [NUM_COLS-1:0]        col_valid_out;

    // Dynamic weight mux
    reg [NUM_COLS*M*DATA_W-1:0] weight_group;
    integer gj, k;
    always @(*) begin
        weight_group = '0;
        for (gj = 0; gj < NUM_COLS; gj = gj + 1)
            for (k = 0; k < M; k = k + 1)
                if (sub_cycle * NUM_COLS + gj < M)
                    weight_group[(gj*M+k)*DATA_W +: DATA_W] =
                        weight_tile_r[(k*M + sub_cycle*NUM_COLS + gj)*DATA_W +: DATA_W];
    end

    genvar gc;
    generate
        for (gc = 0; gc < NUM_COLS; gc = gc + 1) begin : gen_col
            mul_col #(.M(M), .DATA_W(DATA_W), .PIPE(1)) u_mul_col (
                .clk(clk), .rst_n(rst_n),
                .input_tile(input_tile_r),
                .weight_col(weight_group[gc*M*DATA_W +: M*DATA_W]),
                .valid_in(state == S_FEED),
                .result(col_results[gc*SUM_W +: SUM_W]),
                .result_valid(col_valid_out[gc])
            );
        end
    endgenerate

    // Accumulator (one ACC_W word per output element)
    reg [M*ACC_W-1:0] acc;

    // -------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            sub_cycle    <= '0;
            pipe_cnt     <= '0;
            ready_out    <= 1'b1;
            valid_out    <= 1'b0;
            input_tile_r <= '0;
            weight_tile_r<= '0;
            tile_col_r   <= '0;
            inner_idx_r  <= '0;
            is_last_r    <= 1'b0;
            acc          <= '0;
            result_tile  <= '0;
            result_col   <= '0;
        end else begin
            valid_out <= 1'b0;
            case (state)

                S_IDLE: begin
                    if (valid_in && ready_out) begin
                        input_tile_r  <= input_tile;
                        weight_tile_r <= weight_tile;
                        tile_col_r    <= tile_col;
                        inner_idx_r   <= inner_idx;
                        is_last_r     <= is_last;
                        sub_cycle     <= '0;
                        ready_out     <= 1'b0;
                        state         <= S_FEED;
                    end
                end

                S_FEED: begin
                    // Feed mul_cols; wait adder tree latency = TREE_DEPTH + 1
                    pipe_cnt <= TREE_DEPTH + 1;
                    state    <= S_PIPE;
                end

                S_PIPE: begin
                    if (pipe_cnt == 0) state <= S_ACC;
                    else pipe_cnt <= pipe_cnt - 1;
                end

                S_ACC: begin
                    integer j;
                    for (j = 0; j < NUM_COLS; j = j + 1) begin
                        if (sub_cycle*NUM_COLS + j < M) begin : do_acc
                            reg signed [ACC_W-1:0] cv;
                            cv = $signed({{(ACC_W-SUM_W){col_results[j*SUM_W + SUM_W-1]}},
                                          col_results[j*SUM_W +: SUM_W]});
                            if (inner_idx_r == 0)
                                acc[(sub_cycle*NUM_COLS+j)*ACC_W +: ACC_W] <= cv;
                            else
                                acc[(sub_cycle*NUM_COLS+j)*ACC_W +: ACC_W] <=
                                    $signed(acc[(sub_cycle*NUM_COLS+j)*ACC_W +: ACC_W]) + cv;
                        end
                    end

                    if (sub_cycle >= NUM_SUB - 1) begin
                        if (is_last_r) state <= S_OUT;
                        else begin ready_out <= 1'b1; state <= S_IDLE; end
                    end else begin
                        sub_cycle <= sub_cycle + 1;
                        state     <= S_FEED;
                    end
                end

                S_OUT: begin
                    if (ready_in) begin
                        integer j;
                        reg signed [ACC_W-1:0] av;
                        for (j = 0; j < M; j = j + 1) begin
                            av = $signed(acc[j*ACC_W +: ACC_W]);
                            if (av > MAX_ACC)
                                result_tile[j*DATA_W +: DATA_W] <= MAX_OUT;
                            else if (av < MIN_ACC)
                                result_tile[j*DATA_W +: DATA_W] <= MIN_OUT;
                            else
                                result_tile[j*DATA_W +: DATA_W] <=
                                    av[FRAC_W +: DATA_W];
                        end
                        result_col <= tile_col_r;
                        valid_out  <= 1'b1;
                        ready_out  <= 1'b1;
                        state      <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
