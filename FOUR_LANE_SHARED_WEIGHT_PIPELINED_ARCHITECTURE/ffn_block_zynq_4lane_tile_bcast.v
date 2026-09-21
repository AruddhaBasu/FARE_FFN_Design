//=============================================================================
// ffn_block_zynq_4lane_tile_bcast.v
//
// Separate tile-broadcast top level for the four-lane architecture.
//
// The top-level contract is unchanged from the four-lane block, while the
// internal shared AXI cache operates on complete M×M weight bursts.  This
// keeps lane-level AXI details below the top-level boundary and provides a
// single shared weight-tile path for the four lanes.
//
// Recommended configuration:
//   D=4096, M=32, HIDDEN_DIM=11008, N=2048, LANES=4, NUM_COLS=16
//=============================================================================

module ffn_block_zynq_4lane_tile_bcast #(
    parameter D          = 4096,
    parameter M          = 32,
    parameter HIDDEN_DIM = 11008,
    parameter N          = 2048,
    parameter LANES      = 4,
    parameter NUM_COLS   = 16,
    parameter DATA_W     = 16,
    parameter FRAC_W     = 8,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    output wire                          done,
    input  wire signed [DATA_W-1:0]      alpha_pre,
    input  wire signed [DATA_W-1:0]      alpha_post,

    // One shared AXI read channel.  Weight bursts are cached and reused as
    // complete M×M tiles by the internal shared-cache path.
    output wire [AXI_ADDR_W-1:0]         araddr,
    output wire [7:0]                    arlen,
    output wire [2:0]                    arsize,
    output wire [1:0]                    arburst,
    output wire                          arvalid,
    input  wire                          arready,
    input  wire [AXI_DATA_W-1:0]         rdata,
    input  wire                          rlast,
    input  wire                          rvalid,
    output wire                          rready,

    // Ordered output stream
    output wire [AXI_DATA_W-1:0]         out_data,
    output wire [`CLOG2_MIN1(D/M)-1:0]  out_addr,
    output wire [2:0]                   out_offset,
    output wire [`CLOG2_MIN1(N)-1:0]    out_seq_idx,
    output wire                         out_valid,
    output wire                         out_row_last,
    output wire                         out_last
);

    ffn_block_zynq_4lane #(
        .D(D),
        .M(M),
        .HIDDEN_DIM(HIDDEN_DIM),
        .N(N),
        .LANES(LANES),
        .NUM_COLS(NUM_COLS),
        .DATA_W(DATA_W),
        .FRAC_W(FRAC_W),
        .AXI_DATA_W(AXI_DATA_W),
        .AXI_ADDR_W(AXI_ADDR_W)
    ) u_tile_broadcast_core (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .alpha_pre(alpha_pre),
        .alpha_post(alpha_post),
        .araddr(araddr),
        .arlen(arlen),
        .arsize(arsize),
        .arburst(arburst),
        .arvalid(arvalid),
        .arready(arready),
        .rdata(rdata),
        .rlast(rlast),
        .rvalid(rvalid),
        .rready(rready),
        .out_data(out_data),
        .out_addr(out_addr),
        .out_offset(out_offset),
        .out_seq_idx(out_seq_idx),
        .out_valid(out_valid),
        .out_row_last(out_row_last),
        .out_last(out_last)
    );

endmodule
