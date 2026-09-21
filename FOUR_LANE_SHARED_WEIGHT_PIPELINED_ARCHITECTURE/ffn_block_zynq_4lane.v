//=============================================================================
// ffn_block_zynq_4lane.v
//
// Four-lane sequence-parallel wrapper for ffn_block_zynq.
//
// Configuration targeted by the design plan:
//   LANES      = 4
//   M          = 32
//   NUM_COLS   = 16 per lane
//   HIDDEN_DIM = 11008
//
// Each lane processes one 1×D vector.  The scheduler dispatches sequence
// indices in groups of four, captures each lane's serialized result into a
// per-lane output buffer, then emits rows in seq_idx order.  The shared AXI
// cache services common weight/DyT-parameter bursts and serializes private
// input/residual bursts.
//=============================================================================

module ffn_block_zynq_4lane #(
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
    output reg                           done,
    input  wire signed [DATA_W-1:0]      alpha_pre,
    input  wire signed [DATA_W-1:0]      alpha_post,

    // Shared external AXI4 read interface
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
    output reg  [AXI_DATA_W-1:0]         out_data,
    output reg  [`CLOG2_MIN1(D/M)-1:0]   out_addr,
    output reg  [2:0]                    out_offset,
    output reg  [`CLOG2_MIN1(N)-1:0]     out_seq_idx,
    output reg                           out_valid,
    output reg                           out_row_last,
    output reg                           out_last
);

    localparam SEQ_W       = `CLOG2_MIN1(N);
    localparam NEXT_SEQ_W  = `CLOG2_MIN1(N+1);
    localparam OUT_BEATS   = (D * DATA_W) / AXI_DATA_W;
    localparam OUT_BEAT_W  = `CLOG2_MIN1(OUT_BEATS);
    localparam TILE_ADDR_W = `CLOG2_MIN1(D/M);

    localparam S_IDLE    = 2'd0;
    localparam S_DISPATCH= 2'd1;
    localparam S_RUN     = 2'd2;
    localparam S_STREAM  = 2'd3;

    reg [1:0] state;
    reg [NEXT_SEQ_W-1:0] next_seq;
    reg [LANES-1:0] active_mask;
    reg [LANES-1:0] lane_start;
    reg [SEQ_W-1:0] lane_seq_idx [0:LANES-1];
    wire [LANES-1:0] lane_done;

    // Per-lane AXI signals
    wire [AXI_ADDR_W-1:0] lane_araddr [0:LANES-1];
    wire [7:0]            lane_arlen  [0:LANES-1];
    wire [2:0]            lane_arsize [0:LANES-1];
    wire [1:0]            lane_arburst[0:LANES-1];
    wire                  lane_arvalid[0:LANES-1];
    wire                  lane_arready[0:LANES-1];
    wire [AXI_DATA_W-1:0] lane_rdata [0:LANES-1];
    wire                  lane_rlast [0:LANES-1];
    wire                  lane_rvalid[0:LANES-1];
    wire                  lane_rready[0:LANES-1];

    // Per-lane output stream and capture buffer
    wire [AXI_DATA_W-1:0] lane_out_data [0:LANES-1];
    wire [TILE_ADDR_W-1:0] lane_out_addr [0:LANES-1];
    wire [2:0]             lane_out_offset [0:LANES-1];
    wire [SEQ_W-1:0]       lane_out_seq_idx [0:LANES-1];
    wire                   lane_out_valid [0:LANES-1];
    wire                   lane_out_row_last [0:LANES-1];
    wire                   lane_out_last [0:LANES-1];

    reg [AXI_DATA_W-1:0] lane_out_mem [0:LANES-1][0:OUT_BEATS-1];
    reg [OUT_BEAT_W:0] lane_capture_count [0:LANES-1];
    reg [OUT_BEAT_W:0] stream_count;
    reg [`CLOG2_MIN1(LANES)-1:0] stream_lane;
    reg [LANES-1:0] stream_complete;
    reg capture_clear;
    integer i;

    // Flatten lane signals for the shared cache/arbiter.
    wire [LANES*AXI_ADDR_W-1:0] lane_araddr_flat;
    wire [LANES*8-1:0]          lane_arlen_flat;
    wire [LANES*3-1:0]          lane_arsize_flat;
    wire [LANES-1:0]            lane_arvalid_flat;
    wire [LANES-1:0]            lane_arready_flat;
    wire [LANES*AXI_DATA_W-1:0] lane_rdata_flat;
    wire [LANES-1:0]            lane_rvalid_flat;
    wire [LANES-1:0]            lane_rlast_flat;
    wire [LANES-1:0]            lane_rready_flat;

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : gen_lane
            assign lane_araddr_flat[g*AXI_ADDR_W +: AXI_ADDR_W] = lane_araddr[g];
            assign lane_arlen_flat[g*8 +: 8] = lane_arlen[g];
            assign lane_arsize_flat[g*3 +: 3] = lane_arsize[g];
            assign lane_arvalid_flat[g] = lane_arvalid[g];
            assign lane_arready[g] = lane_arready_flat[g];
            assign lane_rdata[g] = lane_rdata_flat[g*AXI_DATA_W +: AXI_DATA_W];
            assign lane_rvalid[g] = lane_rvalid_flat[g];
            assign lane_rlast[g] = lane_rlast_flat[g];
            assign lane_rready_flat[g] = lane_rready[g];

            ffn_block_zynq #(
                .D(D), .M(M), .HIDDEN_DIM(HIDDEN_DIM), .N(N),
                .SINGLE_VECTOR(1), .NUM_COLS(NUM_COLS),
                .DATA_W(DATA_W), .FRAC_W(FRAC_W),
                .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W)
            ) u_lane (
                .clk(clk), .rst_n(rst_n), .start(lane_start[g]),
                .seq_start_idx(lane_seq_idx[g]), .done(lane_done[g]),
                .alpha_pre(alpha_pre), .alpha_post(alpha_post),
                .araddr(lane_araddr[g]), .arlen(lane_arlen[g]),
                .arsize(lane_arsize[g]), .arburst(lane_arburst[g]),
                .arvalid(lane_arvalid[g]), .arready(lane_arready[g]),
                .rdata(lane_rdata[g]), .rlast(lane_rlast[g]),
                .rvalid(lane_rvalid[g]), .rready(lane_rready[g]),
                .out_data(lane_out_data[g]), .out_addr(lane_out_addr[g]),
                .out_offset(lane_out_offset[g]), .out_seq_idx(lane_out_seq_idx[g]),
                .out_valid(lane_out_valid[g]),
                .out_row_last(lane_out_row_last[g]), .out_last(lane_out_last[g])
            );
        end
    endgenerate

    axi_shared_weight_cache #(
        .LANES(LANES), .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W)
    ) u_shared_weight_cache (
        .clk(clk), .rst_n(rst_n),
        .lane_araddr(lane_araddr_flat), .lane_arlen(lane_arlen_flat),
        .lane_arsize(lane_arsize_flat), .lane_arvalid(lane_arvalid_flat),
        .lane_arready(lane_arready_flat),
        .lane_rdata(lane_rdata_flat), .lane_rvalid(lane_rvalid_flat),
        .lane_rlast(lane_rlast_flat), .lane_rready(lane_rready_flat),
        .active_mask(active_mask),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready), .rdata(rdata), .rlast(rlast),
        .rvalid(rvalid), .rready(rready)
    );

    // Capture every lane's serialized output into a per-lane reorder buffer.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < LANES; i = i + 1)
                lane_capture_count[i] <= '0;
        end else if (capture_clear) begin
            for (i = 0; i < LANES; i = i + 1)
                lane_capture_count[i] <= '0;
        end else begin
            for (i = 0; i < LANES; i = i + 1) begin
                if (lane_out_valid[i]) begin
                    lane_out_mem[i][lane_capture_count[i]] <= lane_out_data[i];
                    lane_capture_count[i] <= lane_capture_count[i] + 1'b1;
                end
            end
        end
    end

    // Scheduler and ordered output stream.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            done           <= 1'b0;
            next_seq       <= '0;
            active_mask    <= '0;
            lane_start     <= '0;
            stream_count   <= '0;
            stream_lane    <= '0;
            stream_complete<= '0;
            out_data       <= '0;
            out_addr       <= '0;
            out_offset     <= '0;
            out_seq_idx    <= '0;
            out_valid      <= 1'b0;
            out_row_last   <= 1'b0;
            out_last       <= 1'b0;
            for (i = 0; i < LANES; i = i + 1)
                lane_seq_idx[i] <= '0;
        end else begin
            lane_start   <= '0;
            capture_clear <= 1'b0;
            out_valid    <= 1'b0;
            out_row_last <= 1'b0;
            out_last     <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        done <= 1'b0;
                        next_seq        <= '0;
                        stream_complete <= '0;
                        state           <= S_DISPATCH;
                    end
                end

                S_DISPATCH: begin
                    capture_clear <= 1'b1;
                    active_mask <= '0;
                    for (i = 0; i < LANES; i = i + 1) begin
                        if ((next_seq + i) < N) begin
                            lane_seq_idx[i] <= next_seq + i;
                            active_mask[i]  <= 1'b1;
                            lane_start[i]   <= 1'b1;
                        end
                        lane_capture_count[i] <= '0;
                    end
                    if (next_seq + LANES < N)
                        next_seq <= next_seq + LANES;
                    else
                        next_seq <= N;
                    state <= S_RUN;
                end

                S_RUN: begin
                    // Wait until every active lane finishes its vector.
                    if ((lane_done & active_mask) == active_mask) begin
                        stream_lane  <= '0;
                        stream_count <= '0;
                        state        <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    out_data     <= lane_out_mem[stream_lane][stream_count];
                    out_addr     <= stream_count / (M*DATA_W/AXI_DATA_W);
                    out_offset   <= stream_count % (M*DATA_W/AXI_DATA_W);
                    out_seq_idx  <= lane_seq_idx[stream_lane];
                    out_valid    <= 1'b1;
                    out_row_last <= (stream_count == OUT_BEATS-1);
                    out_last     <= (stream_count == OUT_BEATS-1) &&
                                    (stream_lane == LANES-1 ||
                                     !active_mask[stream_lane+1]);

                    if (stream_count == OUT_BEATS-1) begin
                        stream_count <= '0;
                        if (stream_lane == LANES-1 || !active_mask[stream_lane+1]) begin
                            if (next_seq >= N) begin
                                done  <= 1'b1;
                                state <= S_IDLE;
                            end else begin
                                state <= S_DISPATCH;
                            end
                        end else begin
                            stream_lane <= stream_lane + 1'b1;
                        end
                    end else begin
                        stream_count <= stream_count + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
