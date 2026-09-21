//============================================================================
// ffn_block_zynq.v — Complete Transformer FFN Block with DyT Normalization
//============================================================================
// Three-phase operation:
//   Phase 0 (PRE_DYT):  y_in = γ₁·tanh(α₁·(x + res₁)) + β₁ → norm_input_bram
//   Phase 1 (FFN):      y_ffn = ReLU(y_in·W_up)·W_down → ffn_output_bram
//   Phase 2 (POST_DYT): y_out = γ₂·tanh(α₂·(y_ffn + y_in)) + β₂ → stream out
//
// AXI read master is shared; phases never overlap.
// FFN fetch reads input from norm_input_bram (local) instead of AXI.
// N sequence elements are processed sequentially; local BRAMs are reused per
// element and out_seq_idx identifies the serialized output row.
//============================================================================

module ffn_block_zynq #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter NUM_COLS   = 8,
    parameter HIDDEN_DIM  = 4*D,
    parameter N          = 1,
    parameter SINGLE_VECTOR = 0,
    parameter DATA_W     = 16,
    parameter FRAC_W     = 8,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [`CLOG2_MIN1(N)-1:0]     seq_start_idx,
    output wire                          done,
    input  wire signed [DATA_W-1:0]     alpha_pre,
    input  wire signed [DATA_W-1:0]     alpha_post,
    // AXI4 Read Master
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
    // Streamed Output
    output wire [AXI_DATA_W-1:0]        out_data,
    output wire [`CLOG2_MIN1(D/M)-1:0]       out_addr,
    output wire [2:0]                   out_offset,
    output wire                          out_valid,
    output wire                          out_last,
    output wire                          out_row_last,
    output wire [`CLOG2_MIN1(N)-1:0]    out_seq_idx
);

    localparam NUM_TILES_D = D / M;
    localparam NUM_TILES_H = HIDDEN_DIM / M;
    localparam MAX_INNER   = (NUM_TILES_H > NUM_TILES_D) ? NUM_TILES_H : NUM_TILES_D;

    localparam PHASE_PRE  = 2'd0;
    localparam PHASE_FFN  = 2'd1;
    localparam PHASE_POST = 2'd2;

    reg [1:0] phase_sel;
    // Declared before module instances because these are driven in the
    // phase-controller always block and consumed as start inputs below.
    reg       pre_start_pulse, ffn_start_pulse, post_start_pulse;
    reg [`CLOG2_MIN1(N)-1:0] seq_idx;
    wire pre_done, ffn_done, post_done;
    wire fetch_done;  // Fetch module done (separate from accumulator done)

    // -------------------------------------------------------------------
    // AXI request wires from each phase — declared BEFORE use
    // -------------------------------------------------------------------
    wire [AXI_ADDR_W-1:0] pre_axi_addr,  ffn_axi_addr,  post_axi_addr;
    wire [7:0]             pre_axi_len,   ffn_axi_len,   post_axi_len;
    wire [2:0]             pre_axi_size,  ffn_axi_size,  post_axi_size;
    wire                   pre_axi_valid, ffn_axi_valid, post_axi_valid;

    // AXI request-ready wires — declared BEFORE assign statements
    wire pre_axi_ready_wire, ffn_axi_ready_wire, post_axi_ready_wire;

    // AXI response-ready wires from each phase
    wire pre_axi_rready, ffn_axi_rready, post_axi_rready;

    // -------------------------------------------------------------------
    // AXI request mux — only active phase drives the master
    // -------------------------------------------------------------------
    wire [AXI_ADDR_W-1:0] mux_axi_addr = (phase_sel == PHASE_PRE) ? pre_axi_addr :
                                          (phase_sel == PHASE_POST) ? post_axi_addr :
                                          ffn_axi_addr;
    wire [7:0]  mux_axi_len = (phase_sel == PHASE_PRE) ? pre_axi_len :
                              (phase_sel == PHASE_POST) ? post_axi_len : ffn_axi_len;
    wire [2:0]  mux_axi_size = (phase_sel == PHASE_PRE) ? pre_axi_size :
                               (phase_sel == PHASE_POST) ? post_axi_size : ffn_axi_size;
    wire        mux_axi_valid = (phase_sel == PHASE_PRE) ? pre_axi_valid :
                                (phase_sel == PHASE_POST) ? post_axi_valid : ffn_axi_valid;

    // AXI response-ready mux: route rready to the active phase
    wire mux_axi_rready = (phase_sel == PHASE_PRE) ? pre_axi_rready :
                          (phase_sel == PHASE_POST) ? post_axi_rready :
                          ffn_axi_rready;

    // AXI request-ready from master: route to active phase only
    wire axi_req_ready;  // From AXI read master
    assign pre_axi_ready_wire  = (phase_sel == PHASE_PRE)  ? axi_req_ready : 1'b0;
    assign ffn_axi_ready_wire  = (phase_sel == PHASE_FFN)  ? axi_req_ready : 1'b0;
    assign post_axi_ready_wire = (phase_sel == PHASE_POST) ? axi_req_ready : 1'b0;

    // -------------------------------------------------------------------
    // AXI Read Master
    //
    // The axi_read_master bridges a simple req/resp interface to AXI4.
    // It drives araddr/arlen/arsize/arburst/arvalid on the AR channel,
    // and rready on the R channel. It reads rdata/rlast/rvalid.
    // The resp_data/resp_last/resp_valid outputs are direct copies of
    // rdata/rlast/rvalid (with valid gated by state), so we don't need
    // separate resp wires — the phases read rdata/rlast/rvalid directly.
    // -------------------------------------------------------------------
    axi_read_master #(
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_axi_read_master (
        .clk(clk), .rst_n(rst_n),
        .req_addr(mux_axi_addr), .req_len(mux_axi_len),
        .req_size(mux_axi_size), .req_valid(mux_axi_valid),
        .req_ready(axi_req_ready),
        // Response outputs not needed — phases read AXI R channel directly
        .resp_data(), .resp_last(), .resp_valid(), .resp_ready(mux_axi_rready),
        // AXI4 AR channel
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready),
        // AXI4 R channel
        .rdata(rdata), .rlast(rlast), .rvalid(rvalid), .rready(rready)
    );

    // -------------------------------------------------------------------
    // Normalized Input BRAM (pre-DyT output → FFN input)
    // -------------------------------------------------------------------
    wire norm_wr_en, norm_rd_en;
    wire [`CLOG2_MIN1(NUM_TILES_D)-1:0] norm_wr_addr, norm_rd_addr;
    wire [M*DATA_W-1:0] norm_wr_data, norm_rd_data;

    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_D))
    u_norm_input_bram (.clk(clk), .wr_en(norm_wr_en), .wr_addr(norm_wr_addr),
        .wr_data(norm_wr_data), .rd_en(norm_rd_en), .rd_addr(norm_rd_addr), .rd_data(norm_rd_data));

    // -------------------------------------------------------------------
    // Residual BRAM (z₁ = x+res₁, stored for post-DyT)
    // -------------------------------------------------------------------
    wire res_wr_en, res_rd_en;
    wire [`CLOG2_MIN1(NUM_TILES_D)-1:0] res_wr_addr, res_rd_addr;
    wire [M*DATA_W-1:0] res_wr_data, res_rd_data;

    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_D))
    u_residual_bram (.clk(clk), .wr_en(res_wr_en), .wr_addr(res_wr_addr),
        .wr_data(res_wr_data), .rd_en(res_rd_en), .rd_addr(res_rd_addr), .rd_data(res_rd_data));

    // -------------------------------------------------------------------
    // Phase 0: Pre-Add-DyT
    // -------------------------------------------------------------------
    add_dyt_stage #(.D(D), .M(M), .N(N), .DATA_W(DATA_W), .FRAC_W(FRAC_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W))
    u_pre_dyt (
        .clk(clk), .rst_n(rst_n),
        .start(pre_start_pulse),
        .seq_idx(seq_idx),
        .done(pre_done),
        .axi_req_addr(pre_axi_addr), .axi_req_len(pre_axi_len),
        .axi_req_size(pre_axi_size), .axi_req_valid(pre_axi_valid),
        .axi_req_ready(pre_axi_ready_wire),
        .axi_resp_data(rdata), .axi_resp_last(rlast),
        .axi_resp_valid(rvalid), .axi_resp_ready(pre_axi_rready),
        .alpha(alpha_pre),
        .out_wr_en(norm_wr_en), .out_wr_addr(norm_wr_addr), .out_wr_data(norm_wr_data),
        .res_wr_en(res_wr_en), .res_wr_addr(res_wr_addr), .res_wr_data(res_wr_data),
        .stream_data(), .stream_addr(), .stream_valid(), .stream_last()
    );

    // -------------------------------------------------------------------
    // Phase 1: FFN Pipeline
    // -------------------------------------------------------------------
    // Weight tiles are M×M matrices — must use M*M*DATA_W width
    wire [M*DATA_W-1:0]           up_input_tile;
    wire [M*M*DATA_W-1:0]         up_weight_tile;     // M×M weight tile for UP projection
    wire [M*DATA_W-1:0]           relu_input_tile;
    wire [M*M*DATA_W-1:0]         down_weight_tile;   // M×M weight tile for DOWN projection

    // Index signals — all use MAX_INNER width for consistency with tm_proj_stage
    // ports.  Valid values always fit; extra bits are zero-padded.
    wire [`CLOG2_MIN1(MAX_INNER)-1:0]  up_tile_col, relu_tile_col;
    wire [`CLOG2_MIN1(MAX_INNER)-1:0]  up_inner_idx;       // fetch_addr_gen_dyt inner range = [0, D/M-1]
    wire [`CLOG2_MIN1(MAX_INNER)-1:0]  down_tile_col;      // DOWN column index range = [0, D/M-1]
    wire [`CLOG2_MIN1(MAX_INNER)-1:0]  down_inner_idx;     // DOWN inner index range = [0, HIDDEN_DIM/M-1]
    wire [`CLOG2_MIN1(MAX_INNER)-1:0]  acc_result_col;     // Accumulator column = [0, D/M-1]
    wire                           up_is_last, up_valid, up_ready;
    wire                           relu_valid_in, relu_ready_out;
    wire                           relu_wr_en, relu_rd_en;
    wire [`CLOG2_MIN1(NUM_TILES_H)-1:0] relu_wr_addr, relu_rd_addr;
    wire [M*DATA_W-1:0]           relu_wr_data, relu_rd_data, down_relu_tile;
    wire [M*DATA_W-1:0]           acc_result_tile;
    wire                           acc_valid_in, acc_ready_out;
    wire                           ffn_out_wr_en, ffn_out_rd_en;
    wire [`CLOG2_MIN1(NUM_TILES_D)-1:0] ffn_out_wr_addr, ffn_out_rd_addr;
    wire [M*DATA_W-1:0]           ffn_out_wr_data, ffn_out_rd_data;
    wire                           down_is_last, down_valid, down_ready;

    fetch_addr_gen_dyt #(.D(D), .M(M), .N(N), .HIDDEN_DIM(HIDDEN_DIM), .DATA_W(DATA_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W), .MAX_INNER(MAX_INNER))
    u_fetch_dyt (
        .clk(clk), .rst_n(rst_n), .start(ffn_start_pulse), .seq_idx(seq_idx),
        .done(fetch_done), .phase_up(), .phase_down(),
        .axi_req_addr(ffn_axi_addr), .axi_req_len(ffn_axi_len),
        .axi_req_size(ffn_axi_size), .axi_req_valid(ffn_axi_valid),
        .axi_req_ready(ffn_axi_ready_wire),
        .axi_resp_data(rdata), .axi_resp_last(rlast),
        .axi_resp_valid(rvalid), .axi_resp_ready(ffn_axi_rready),
        .norm_rd_en(norm_rd_en), .norm_rd_addr(norm_rd_addr), .norm_rd_data(norm_rd_data),
        .relu_rd_en(relu_rd_en), .relu_rd_addr(relu_rd_addr), .relu_rd_data(relu_rd_data),
        .up_input_tile(up_input_tile), .up_weight_tile(up_weight_tile),
        .up_tile_col(up_tile_col), .up_inner_idx(up_inner_idx),
        .up_is_last(up_is_last), .up_valid(up_valid), .up_ready(up_ready),
        .down_relu_tile(down_relu_tile), .down_weight_tile(down_weight_tile),
        .down_tile_col(down_tile_col), .down_inner_idx(down_inner_idx),
        .down_is_last(down_is_last), .down_valid(down_valid), .down_ready(down_ready)
    );

    tm_proj_stage #(.D(D), .M(M), .NUM_COLS(NUM_COLS), .MAX_INNER(MAX_INNER), .DATA_W(DATA_W), .FRAC_W(FRAC_W))
    u_up_proj (.clk(clk), .rst_n(rst_n), .input_tile(up_input_tile),
        .weight_tile(up_weight_tile), .tile_col(up_tile_col), .inner_idx(up_inner_idx),
        .is_last(up_is_last), .valid_in(up_valid), .ready_out(up_ready),
        .result_tile(relu_input_tile), .result_col(relu_tile_col),
        .valid_out(relu_valid_in), .ready_in(relu_ready_out));

    relu_stage #(.D(D), .M(M), .HIDDEN_DIM(HIDDEN_DIM), .DATA_W(DATA_W))
    u_relu (.clk(clk), .rst_n(rst_n), .input_tile(relu_input_tile),
        .tile_col(relu_tile_col), .valid_in(relu_valid_in), .ready_out(relu_ready_out),
        .relu_wr_en(relu_wr_en), .relu_wr_addr(relu_wr_addr), .relu_wr_data(relu_wr_data),
        .result_tile(), .result_col(), .valid_out(), .ready_in(1'b1));

    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_H))
    u_relu_bram (.clk(clk), .wr_en(relu_wr_en), .wr_addr(relu_wr_addr),
        .wr_data(relu_wr_data), .rd_en(relu_rd_en), .rd_addr(relu_rd_addr), .rd_data(relu_rd_data));

    tm_proj_stage #(.D(D), .M(M), .NUM_COLS(NUM_COLS), .MAX_INNER(MAX_INNER), .DATA_W(DATA_W), .FRAC_W(FRAC_W))
    u_down_proj (.clk(clk), .rst_n(rst_n), .input_tile(down_relu_tile),
        .weight_tile(down_weight_tile), .tile_col(down_tile_col), .inner_idx(down_inner_idx),
        .is_last(down_is_last), .valid_in(down_valid), .ready_out(down_ready),
        .result_tile(acc_result_tile), .result_col(acc_result_col),
        .valid_out(acc_valid_in), .ready_in(acc_ready_out));

    accumulator #(.D(D), .M(M), .DATA_W(DATA_W), .DISABLE_READOUT(1), .MAX_INNER(MAX_INNER))
    u_acc (.clk(clk), .rst_n(rst_n), .start(ffn_start_pulse),
        .done(ffn_done), .result_tile(acc_result_tile), .result_col(acc_result_col),
        .valid_in(acc_valid_in), .ready_out(acc_ready_out),
        .out_wr_en(ffn_out_wr_en), .out_wr_addr(ffn_out_wr_addr), .out_wr_data(ffn_out_wr_data),
        .out_rd_en(), .out_rd_addr(), .out_rd_data(ffn_out_rd_data),
        .output_tile_data(), .output_tile_addr(), .output_tile_valid(),
        .output_data(), .output_valid());

    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_D))
    u_ffn_out_bram (.clk(clk), .wr_en(ffn_out_wr_en), .wr_addr(ffn_out_wr_addr),
        .wr_data(ffn_out_wr_data), .rd_en(ffn_out_rd_en), .rd_addr(ffn_out_rd_addr),
        .rd_data(ffn_out_rd_data));

    // -------------------------------------------------------------------
    // Phase 2: Post-Add-DyT
    // -------------------------------------------------------------------
    wire [AXI_DATA_W-1:0] out_data_int;
    wire [`CLOG2_MIN1(NUM_TILES_D)-1:0] out_addr_int;
    wire out_valid_int, out_last_int, out_row_last_int;
    wire [`CLOG2_MIN1(N)-1:0] out_seq_idx_int;

    add_dyt_post #(.D(D), .M(M), .N(N), .SINGLE_VECTOR(SINGLE_VECTOR), .DATA_W(DATA_W), .FRAC_W(FRAC_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W))
    u_post_dyt (
        .clk(clk), .rst_n(rst_n),
        .start(post_start_pulse), .seq_idx(seq_idx), .done(post_done),
        .axi_req_addr(post_axi_addr), .axi_req_len(post_axi_len),
        .axi_req_size(post_axi_size), .axi_req_valid(post_axi_valid),
        .axi_req_ready(post_axi_ready_wire),
        .axi_resp_data(rdata), .axi_resp_last(rlast),
        .axi_resp_valid(rvalid), .axi_resp_ready(post_axi_rready),
        .alpha(alpha_post),
        .ffn_rd_en(ffn_out_rd_en), .ffn_rd_addr(ffn_out_rd_addr), .ffn_rd_data(ffn_out_rd_data),
        .res_rd_en(res_rd_en), .res_rd_addr(res_rd_addr), .res_rd_data(res_rd_data),
        .out_data(out_data_int), .out_addr(out_addr_int),
        .out_valid(out_valid_int), .out_last(out_last_int),
        .out_row_last(out_row_last_int), .out_seq_idx(out_seq_idx_int)
    );

    // -------------------------------------------------------------------
    // Phase controller FSM with one-cycle start pulses
    // -------------------------------------------------------------------
    // CRITICAL: Using phase_sel == PHASE_x as a continuous start signal
    // breaks modules that check start every cycle (accumulator resets
    // perpetually) or that restart on seeing start in S_DONE (add_dyt_post
    // loops infinitely). Instead, generate proper ONE-CYCLE start pulses
    // from phase state transitions.
    // -------------------------------------------------------------------
    reg [1:0] phase_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_sel         <= PHASE_PRE;
            phase_state       <= 2'd0;
            seq_idx           <= '0;
            pre_start_pulse   <= 1'b0;
            ffn_start_pulse   <= 1'b0;
            post_start_pulse  <= 1'b0;
        end else begin
            // Default: clear all pulses (one-cycle width)
            pre_start_pulse  <= 1'b0;
            ffn_start_pulse  <= 1'b0;
            post_start_pulse <= 1'b0;

            case (phase_state)
                2'd0: if (start) begin
                    phase_sel       <= PHASE_PRE;
                    phase_state     <= 2'd1;
                    seq_idx         <= SINGLE_VECTOR ? seq_start_idx : '0;
                    pre_start_pulse <= 1'b1;    // Pulse: start Phase 0
                end
                2'd1: if (pre_done && !pre_start_pulse) begin
                    phase_sel       <= PHASE_FFN;
                    phase_state     <= 2'd2;
                    ffn_start_pulse <= 1'b1;    // Pulse: start Phase 1
                end
                2'd2: if (ffn_done && !ffn_start_pulse) begin
                    phase_sel        <= PHASE_POST;
                    phase_state      <= 2'd3;
                    post_start_pulse <= 1'b1;   // Pulse: start Phase 2
                end
                2'd3: if (post_done && !post_start_pulse) begin
                    if (SINGLE_VECTOR || (seq_idx == N - 1)) begin
                        phase_state <= 2'd0;
                    end else begin
                        seq_idx         <= seq_idx + 1'b1;
                        phase_sel       <= PHASE_PRE;
                        phase_state     <= 2'd1;
                        pre_start_pulse <= 1'b1; // Start next sequence element
                    end
                end
                default: phase_state <= 2'd0;
            endcase
        end
    end

    // Output connections
    assign out_data   = out_data_int;
    assign out_addr   = out_addr_int;
    assign out_offset = 3'd0;   // Simplified — post-DyT streams one beat per tile
    assign out_valid    = out_valid_int;
    assign out_last     = out_last_int;
    assign out_row_last = out_row_last_int;
    assign out_seq_idx  = out_seq_idx_int;
    assign done       = (phase_state == 2'd0) && post_done;

endmodule
