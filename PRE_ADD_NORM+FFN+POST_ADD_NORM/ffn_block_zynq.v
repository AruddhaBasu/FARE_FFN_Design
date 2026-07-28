//============================================================================
// ffn_block_zynq.v — Complete FFN with Pre/Post Dynamic Tanh normalization
//============================================================================
// 3-phase operation:
//   Phase 0 (PRE_DYT):  y_in  = gamma1*tanh(alpha1*(x+res1)) + beta1 → norm_bram
//                       z1    = x+res1                                     → res_bram
//   Phase 1 (FFN):      y_ffn = ReLU(y_in @ W_up) @ W_down                → ffn_bram
//   Phase 2 (POST_DYT): y_out = gamma2*tanh(alpha2*(y_ffn+z1)) + beta2    → stream out
//
// AXI read master is time-multiplexed across phases.
//============================================================================

`include "tanh_lut.v"
`include "add_dyt_stage.v"
`include "add_dyt_post.v"
`include "axi_read_master.v"
`include "bram_dp.v"
`include "bram_dp_wide.v"
`include "adder_tree.v"
`include "mul_col.v"
`include "tm_proj_stage.v"
`include "relu_stage.v"
`include "accumulator.v"
`include "fetch_addr_gen_dyt.v"

module ffn_block_zynq #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter NUM_COLS   = 8,
    parameter DATA_W     = 16,
    parameter FRAC_W     = 8,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    output wire                          done,
    input  wire signed [DATA_W-1:0]     alpha_pre,
    input  wire signed [DATA_W-1:0]     alpha_post,
    // AXI4 read master
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
    // Streamed output
    output wire [AXI_DATA_W-1:0]        out_data,
    output wire [$clog2(D/M)-1:0]       out_addr,
    output wire [2:0]                   out_offset,
    output wire                          out_valid,
    output wire                          out_last
);

    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;
    localparam MAX_INNER    = (NUM_TILES_4D > NUM_TILES_D) ? NUM_TILES_4D : NUM_TILES_D;

    localparam PHASE_PRE=2'd0, PHASE_FFN=2'd1, PHASE_POST=2'd2;
    reg [1:0] phase_sel;

    wire pre_done, ffn_done, post_done;
    wire fetch_done;

    // AXI request/response wires from each phase
    wire [AXI_ADDR_W-1:0] pre_axi_addr,  ffn_axi_addr,  post_axi_addr;
    wire [7:0]            pre_axi_len,   ffn_axi_len,   post_axi_len;
    wire [2:0]            pre_axi_size,  ffn_axi_size,  post_axi_size;
    wire                  pre_axi_valid, ffn_axi_valid, post_axi_valid;
    wire pre_axi_ready_w, ffn_axi_ready_w, post_axi_ready_w;
    wire pre_axi_rready,  ffn_axi_rready, post_axi_rready;

    // AXI request mux
    wire [AXI_ADDR_W-1:0] mux_axi_addr = (phase_sel == PHASE_PRE)  ? pre_axi_addr :
                                         (phase_sel == PHASE_POST) ? post_axi_addr :
                                         ffn_axi_addr;
    wire [7:0] mux_axi_len = (phase_sel == PHASE_PRE)  ? pre_axi_len :
                             (phase_sel == PHASE_POST) ? post_axi_len : ffn_axi_len;
    wire [2:0] mux_axi_size= (phase_sel == PHASE_PRE)  ? pre_axi_size :
                             (phase_sel == PHASE_POST) ? post_axi_size : ffn_axi_size;
    wire       mux_axi_valid=(phase_sel == PHASE_PRE)  ? pre_axi_valid :
                             (phase_sel == PHASE_POST) ? post_axi_valid : ffn_axi_valid;
    wire       mux_axi_rready=(phase_sel==PHASE_PRE)  ? pre_axi_rready :
                              (phase_sel==PHASE_POST) ? post_axi_rready : ffn_axi_rready;

    wire axi_req_ready;
    assign pre_axi_ready_w  = (phase_sel == PHASE_PRE)  ? axi_req_ready : 1'b0;
    assign ffn_axi_ready_w  = (phase_sel == PHASE_FFN)  ? axi_req_ready : 1'b0;
    assign post_axi_ready_w = (phase_sel == PHASE_POST) ? axi_req_ready : 1'b0;

    // AXI read master
    axi_read_master #(.AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W))
    u_axi (
        .clk(clk), .rst_n(rst_n),
        .req_addr(mux_axi_addr), .req_len(mux_axi_len),
        .req_size(mux_axi_size), .req_valid(mux_axi_valid), .req_ready(axi_req_ready),
        .resp_data(), .resp_last(), .resp_valid(), .resp_ready(mux_axi_rready),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rlast(rlast), .rvalid(rvalid), .rready(rready)
    );

    // -------------------------------------------------------------------
    // BRAMs
    // -------------------------------------------------------------------
    wire norm_wr_en, norm_rd_en;
    wire [$clog2(NUM_TILES_D)-1:0] norm_wr_addr, norm_rd_addr;
    wire [M*DATA_W-1:0] norm_wr_data, norm_rd_data;
    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_D))
    u_norm_bram (.clk(clk),
        .wr_en(norm_wr_en), .wr_addr(norm_wr_addr), .wr_data(norm_wr_data),
        .rd_en(norm_rd_en), .rd_addr(norm_rd_addr), .rd_data(norm_rd_data));

    wire res_wr_en, res_rd_en;
    wire [$clog2(NUM_TILES_D)-1:0] res_wr_addr, res_rd_addr;
    wire [M*DATA_W-1:0] res_wr_data, res_rd_data;
    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_D))
    u_res_bram (.clk(clk),
        .wr_en(res_wr_en), .wr_addr(res_wr_addr), .wr_data(res_wr_data),
        .rd_en(res_rd_en), .rd_addr(res_rd_addr), .rd_data(res_rd_data));

    // -------------------------------------------------------------------
    // Phase 0 — Pre-DyT
    // -------------------------------------------------------------------
    wire pre_start_pulse;
    add_dyt_stage #(.D(D), .M(M), .DATA_W(DATA_W), .FRAC_W(FRAC_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W))
    u_pre_dyt (
        .clk(clk), .rst_n(rst_n),
        .start(pre_start_pulse), .done(pre_done),
        .axi_req_addr(pre_axi_addr), .axi_req_len(pre_axi_len),
        .axi_req_size(pre_axi_size), .axi_req_valid(pre_axi_valid),
        .axi_req_ready(pre_axi_ready_w),
        .axi_resp_data(rdata), .axi_resp_last(rlast),
        .axi_resp_valid(rvalid), .axi_resp_ready(pre_axi_rready),
        .alpha(alpha_pre),
        .out_wr_en(norm_wr_en), .out_wr_addr(norm_wr_addr), .out_wr_data(norm_wr_data),
        .res_wr_en(res_wr_en), .res_wr_addr(res_wr_addr), .res_wr_data(res_wr_data),
        .stream_data(), .stream_addr(), .stream_valid(), .stream_last()
    );

    // -------------------------------------------------------------------
    // Phase 1 — FFN pipeline
    // -------------------------------------------------------------------
    wire [M*DATA_W-1:0]    up_input_tile;
    wire [M*M*DATA_W-1:0]  up_weight_tile;
    wire [M*DATA_W-1:0]    relu_input_tile;
    wire [M*M*DATA_W-1:0]  down_weight_tile;
    wire [$clog2(MAX_INNER)-1:0] up_tile_col, relu_tile_col;
    wire [$clog2(MAX_INNER)-1:0] up_inner_idx, down_tile_col, down_inner_idx;
    wire [$clog2(MAX_INNER)-1:0] acc_result_col;
    wire up_is_last, up_valid, up_ready;
    wire relu_valid_in, relu_ready_out;
    wire relu_wr_en, relu_rd_en;
    wire [$clog2(NUM_TILES_4D)-1:0] relu_wr_addr, relu_rd_addr;
    wire [M*DATA_W-1:0] relu_wr_data, relu_rd_data, down_relu_tile;
    wire [M*DATA_W-1:0] acc_result_tile;
    wire acc_valid_in, acc_ready_out;
    wire ffn_out_wr_en, ffn_out_rd_en;
    wire [$clog2(NUM_TILES_D)-1:0] ffn_out_wr_addr, ffn_out_rd_addr;
    wire [M*DATA_W-1:0] ffn_out_wr_data, ffn_out_rd_data;
    wire down_is_last, down_valid, down_ready;
    wire ffn_start_pulse;

    fetch_addr_gen_dyt #(.D(D), .M(M), .DATA_W(DATA_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W))
    u_fetch (
        .clk(clk), .rst_n(rst_n), .start(ffn_start_pulse),
        .done(fetch_done), .phase_up(), .phase_down(),
        .axi_req_addr(ffn_axi_addr), .axi_req_len(ffn_axi_len),
        .axi_req_size(ffn_axi_size), .axi_req_valid(ffn_axi_valid),
        .axi_req_ready(ffn_axi_ready_w),
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

    tm_proj_stage #(.D(D), .M(M), .NUM_COLS(NUM_COLS), .MAX_INNER(MAX_INNER),
        .DATA_W(DATA_W), .FRAC_W(FRAC_W))
    u_up_proj (.clk(clk), .rst_n(rst_n),
        .input_tile(up_input_tile), .weight_tile(up_weight_tile),
        .tile_col(up_tile_col), .inner_idx(up_inner_idx),
        .is_last(up_is_last), .valid_in(up_valid), .ready_out(up_ready),
        .result_tile(relu_input_tile), .result_col(relu_tile_col),
        .valid_out(relu_valid_in), .ready_in(relu_ready_out));

    relu_stage #(.D(D), .M(M), .DATA_W(DATA_W))
    u_relu (.clk(clk), .rst_n(rst_n),
        .input_tile(relu_input_tile), .tile_col(relu_tile_col),
        .valid_in(relu_valid_in), .ready_out(relu_ready_out),
        .relu_wr_en(relu_wr_en), .relu_wr_addr(relu_wr_addr), .relu_wr_data(relu_wr_data),
        .result_tile(), .result_col(), .valid_out(), .ready_in(1'b1));

    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_4D))
    u_relu_bram (.clk(clk),
        .wr_en(relu_wr_en), .wr_addr(relu_wr_addr), .wr_data(relu_wr_data),
        .rd_en(relu_rd_en), .rd_addr(relu_rd_addr), .rd_data(relu_rd_data));

    tm_proj_stage #(.D(D), .M(M), .NUM_COLS(NUM_COLS), .MAX_INNER(MAX_INNER),
        .DATA_W(DATA_W), .FRAC_W(FRAC_W))
    u_down_proj (.clk(clk), .rst_n(rst_n),
        .input_tile(down_relu_tile), .weight_tile(down_weight_tile),
        .tile_col(down_tile_col), .inner_idx(down_inner_idx),
        .is_last(down_is_last), .valid_in(down_valid), .ready_out(down_ready),
        .result_tile(acc_result_tile), .result_col(acc_result_col),
        .valid_out(acc_valid_in), .ready_in(acc_ready_out));

    accumulator #(.D(D), .M(M), .DATA_W(DATA_W), .DISABLE_READOUT(1))
    u_acc (.clk(clk), .rst_n(rst_n), .start(ffn_start_pulse),
        .done(ffn_done), .result_tile(acc_result_tile), .result_col(acc_result_col),
        .valid_in(acc_valid_in), .ready_out(acc_ready_out),
        .out_wr_en(ffn_out_wr_en), .out_wr_addr(ffn_out_wr_addr), .out_wr_data(ffn_out_wr_data),
        .out_rd_en(), .out_rd_addr(), .out_rd_data(ffn_out_rd_data),
        .output_tile_data(), .output_tile_addr(), .output_tile_valid(),
        .output_data(), .output_valid());

    bram_dp_wide #(.DATA_W(DATA_W), .TILE_SIZE(M), .DEPTH(NUM_TILES_D))
    u_ffn_out_bram (.clk(clk),
        .wr_en(ffn_out_wr_en), .wr_addr(ffn_out_wr_addr), .wr_data(ffn_out_wr_data),
        .rd_en(ffn_out_rd_en), .rd_addr(ffn_out_rd_addr), .rd_data(ffn_out_rd_data));

    // -------------------------------------------------------------------
    // Phase 2 — Post-DyT
    // -------------------------------------------------------------------
    wire post_start_pulse;
    wire [AXI_DATA_W-1:0] out_data_int;
    wire [$clog2(NUM_TILES_D)-1:0] out_addr_int;
    wire out_valid_int, out_last_int;

    add_dyt_post #(.D(D), .M(M), .DATA_W(DATA_W), .FRAC_W(FRAC_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W))
    u_post_dyt (
        .clk(clk), .rst_n(rst_n),
        .start(post_start_pulse), .done(post_done),
        .axi_req_addr(post_axi_addr), .axi_req_len(post_axi_len),
        .axi_req_size(post_axi_size), .axi_req_valid(post_axi_valid),
        .axi_req_ready(post_axi_ready_w),
        .axi_resp_data(rdata), .axi_resp_last(rlast),
        .axi_resp_valid(rvalid), .axi_resp_ready(post_axi_rready),
        .alpha(alpha_post),
        .ffn_rd_en(ffn_out_rd_en), .ffn_rd_addr(ffn_out_rd_addr), .ffn_rd_data(ffn_out_rd_data),
        .res_rd_en(res_rd_en), .res_rd_addr(res_rd_addr), .res_rd_data(res_rd_data),
        .out_data(out_data_int), .out_addr(out_addr_int),
        .out_valid(out_valid_int), .out_last(out_last_int)
    );

    // -------------------------------------------------------------------
    // Phase controller (one-cycle start pulses)
    // -------------------------------------------------------------------
    reg [1:0] phase_state;
    reg       pre_start_pulse_r, ffn_start_pulse_r, post_start_pulse_r;
    assign pre_start_pulse  = pre_start_pulse_r;
    assign ffn_start_pulse  = ffn_start_pulse_r;
    assign post_start_pulse = post_start_pulse_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_sel           <= PHASE_PRE;
            phase_state         <= 2'd0;
            pre_start_pulse_r   <= 1'b0;
            ffn_start_pulse_r   <= 1'b0;
            post_start_pulse_r  <= 1'b0;
        end else begin
            pre_start_pulse_r  <= 1'b0;
            ffn_start_pulse_r  <= 1'b0;
            post_start_pulse_r <= 1'b0;
            case (phase_state)
                2'd0: if (start) begin
                    phase_sel         <= PHASE_PRE;
                    phase_state       <= 2'd1;
                    pre_start_pulse_r <= 1'b1;
                end
                2'd1: if (pre_done) begin
                    phase_sel         <= PHASE_FFN;
                    phase_state       <= 2'd2;
                    ffn_start_pulse_r <= 1'b1;
                end
                2'd2: if (ffn_done) begin
                    phase_sel          <= PHASE_POST;
                    phase_state        <= 2'd3;
                    post_start_pulse_r <= 1'b1;
                end
                2'd3: if (post_done) begin
                    phase_state <= 2'd0;
                end
                default: phase_state <= 2'd0;
            endcase
        end
    end

    assign out_data   = out_data_int;
    assign out_addr   = out_addr_int;
    assign out_offset = 3'd0;
    assign out_valid  = out_valid_int;
    assign out_last   = out_last_int;
    assign done       = (phase_state == 2'd0) && post_done;

endmodule
