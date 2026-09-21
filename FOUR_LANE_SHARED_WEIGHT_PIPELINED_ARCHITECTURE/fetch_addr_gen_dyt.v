//============================================================================
// fetch_addr_gen_dyt.v — Modified Fetch for FFN with DyT Normalization
//============================================================================
// Same as fetch_addr_gen.v but reads input tiles from the local
// norm_input_bram instead of AXI memory during the UP phase.
//
// UP phase changes:
//   - Instead of S_UP_REQ_INPUT / S_UP_WAIT_INPUT (AXI fetch),
//     we read from norm_input_bram with S_UP_REQ_NORM / S_UP_RELAY_NORM
//     / S_UP_WAIT_NORM (local BRAM read with 2-cycle latency)
//   - Weight tiles still fetched from AXI (WUP_BASE unchanged)
//
// DOWN phase: unchanged (reads ReLU from local BRAM, weights from AXI)
//============================================================================

module fetch_addr_gen_dyt #(
    parameter D          = 256,
    parameter M          = 16,
    parameter HIDDEN_DIM  = 4*D,
    parameter N          = 1,
    parameter DATA_W     = 16,
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32,
    parameter MAX_INNER  = 256      // Max inner iterations = max(D/M, 4D/M)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    input  wire [`CLOG2_MIN1(N)-1:0]     seq_idx,
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

    // ---- Normalized Input BRAM Read ----
    output reg                           norm_rd_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       norm_rd_addr,
    input  wire [M*DATA_W-1:0]          norm_rd_data,

    // ---- ReLU BRAM Read Interface ----
    output reg                           relu_rd_en,
    output reg  [`CLOG2_MIN1(HIDDEN_DIM/M)-1:0]     relu_rd_addr,
    input  wire [M*DATA_W-1:0]          relu_rd_data,

    // ---- To Up Projection Stage ----
    output reg  [M*DATA_W-1:0]          up_input_tile,
    output reg  [M*M*DATA_W-1:0]        up_weight_tile,
    output reg  [`CLOG2_MIN1(HIDDEN_DIM/M)-1:0]     up_tile_col,
    output reg  [`CLOG2_MIN1(MAX_INNER)-1:0]  up_inner_idx,
    output reg                           up_is_last,
    output reg                           up_valid,
    input  wire                          up_ready,

    // ---- To Down Projection Stage ----
    output reg  [M*DATA_W-1:0]          down_relu_tile,
    output reg  [M*M*DATA_W-1:0]        down_weight_tile,
    output reg  [`CLOG2_MIN1(MAX_INNER)-1:0]  down_tile_col,
    output reg  [`CLOG2_MIN1(HIDDEN_DIM/M)-1:0]     down_inner_idx,
    output reg                           down_is_last,
    output reg                           down_valid,
    input  wire                          down_ready
);

    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_H  = HIDDEN_DIM / M;
    localparam WORDS_PER_BEAT = AXI_DATA_W / DATA_W;

    localparam WUP_BASE     = 32'h1000_0000;
    localparam WDOWN_BASE   = 32'h2000_0000;
    localparam WTILE_BYTES  = M * M * (DATA_W / 8);

    localparam WEIGHT_BEATS  = (M * M + WORDS_PER_BEAT - 1) / WORDS_PER_BEAT;
    localparam AXI_BEAT_BYTES = AXI_DATA_W / 8;

    // AXI4 LEN is eight bits and encodes beats-1, so one burst can contain
    // at most 256 beats.  Large M×M weight tiles are split into legal AXI
    // bursts while beat_cnt remains an absolute index within the tile.
    function [7:0] weight_burst_len;
        input integer first_beat;
        integer remaining;
        begin
            remaining = WEIGHT_BEATS - first_beat;
            if (remaining > 256)
                weight_burst_len = 8'hFF;
            else if (remaining <= 0)
                weight_burst_len = 8'd0;
            else
                weight_burst_len = remaining - 1;
        end
    endfunction

    function [2:0] axi_size;
        input integer bytes;
        integer s;
        begin s = 0; while ((1 << s) < bytes) s = s + 1; axi_size = s[2:0]; end
    endfunction
    localparam AXI_SZ = axi_size(AXI_DATA_W / 8);

    // -------------------------------------------------------------------
    // State machine — UP phase uses BRAM reads instead of AXI for input
    // -------------------------------------------------------------------
    localparam S_IDLE           = 4'd0;
    localparam S_UP_REQ_NORM    = 4'd1;   // Read norm_input_bram (issue)
    localparam S_UP_RELAY_NORM  = 4'd2;   // BRAM read latency (cycle 1)
    localparam S_UP_WAIT_NORM   = 4'd3;   // BRAM data available
    localparam S_UP_REQ_WEIGHT  = 4'd4;   // AXI request for W_up tile
    localparam S_UP_WAIT_WEIGHT = 4'd5;   // AXI wait for W_up
    localparam S_UP_SEND        = 4'd6;
    localparam S_DOWN_REQ_RELU  = 4'd7;
    localparam S_DOWN_RELAY     = 4'd8;
    localparam S_DOWN_WAIT_RELU = 4'd9;
    localparam S_DOWN_REQ_WGT   = 4'd10;
    localparam S_DOWN_WAIT_WGT  = 4'd11;
    localparam S_DOWN_SEND      = 4'd12;
    localparam S_DONE           = 4'd13;

    reg [3:0] state;

    reg [`CLOG2_MIN1(NUM_TILES_H)-1:0] tile_col;
    reg [`CLOG2_MIN1(NUM_TILES_D)-1:0]  inner_idx;
    reg [`CLOG2_MIN1(NUM_TILES_H)-1:0] inner_idx_wide;

    reg [M*DATA_W-1:0]     input_tile_buf;
    reg [M*M*DATA_W-1:0]   weight_tile_buf;
    reg [M*DATA_W-1:0]     relu_tile_buf;
    reg [`CLOG2_MIN1(M*M)-1:0]  beat_cnt;

    integer k;

    // AXI resp ready: only during AXI wait states (weight fetches)
    assign axi_resp_ready = (state == S_UP_WAIT_WEIGHT) ||
                            (state == S_DOWN_WAIT_WGT);

    // -------------------------------------------------------------------
    // Main state machine
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
            norm_rd_en      <= 1'b0;
            norm_rd_addr    <= '0;
            relu_rd_en      <= 1'b0;
            relu_rd_addr    <= '0;
        end else begin
            up_valid     <= 1'b0;
            down_valid   <= 1'b0;
            norm_rd_en   <= 1'b0;
            relu_rd_en   <= 1'b0;
            axi_req_valid <= 1'b0;

            case (state)
                // =====================================================
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        phase_up   <= 1'b1;
                        phase_down <= 1'b0;
                        tile_col   <= '0;
                        inner_idx  <= '0;
                        state      <= S_UP_REQ_NORM;
                    end
                end

                // =====================================================
                // UP PHASE: Read normalized input from local BRAM
                // =====================================================
                S_UP_REQ_NORM: begin
                    norm_rd_en   <= 1'b1;
                    norm_rd_addr <= inner_idx;
                    state        <= S_UP_RELAY_NORM;
                end

                S_UP_RELAY_NORM: begin
                    // BRAM read in progress (1-cycle latency for registered read)
                    // rd_en was asserted in previous state, takes effect at end of that cycle.
                    // BRAM sees rd_en at next posedge (this cycle), latches data.
                    // Data available next cycle.
                    state <= S_UP_WAIT_NORM;
                end

                S_UP_WAIT_NORM: begin
                    input_tile_buf <= norm_rd_data;
                    // Pre-set AXI request for weight tile
                    axi_req_addr  <= WUP_BASE +
                                     (inner_idx * NUM_TILES_H + tile_col) * WTILE_BYTES;
                    axi_req_len   <= weight_burst_len(0);
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state         <= S_UP_REQ_WEIGHT;
                end

                // =====================================================
                // UP PHASE: Fetch weight tile from AXI
                // =====================================================
                S_UP_REQ_WEIGHT: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        // beat_cnt is an absolute index within the tile;
                        // do not clear it when issuing a continuation burst.
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
                            if (beat_cnt == WEIGHT_BEATS - 1) begin
                                state <= S_UP_SEND;
                            end else begin
                                // Continue the same tile with the next legal
                                // AXI burst.  beat_cnt is the absolute tile
                                // beat index, so the buffer is filled in order.
                                axi_req_addr <= WUP_BASE +
                                    (inner_idx * NUM_TILES_H + tile_col) * WTILE_BYTES +
                                    (beat_cnt + 1) * AXI_BEAT_BYTES;
                                axi_req_len  <= weight_burst_len(beat_cnt + 1);
                                // Keep the request asserted across the
                                // response-to-request state transition; the
                                // AXI master becomes idle after this final
                                // response beat.
                                axi_req_valid <= 1'b1;
                                state <= S_UP_REQ_WEIGHT;
                            end
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
                        up_tile_col     <= tile_col[`CLOG2_MIN1(NUM_TILES_H)-1:0];
                        up_inner_idx    <= inner_idx;
                        up_is_last      <= (inner_idx == NUM_TILES_D - 1);
                        up_valid        <= 1'b1;

                        if (inner_idx == NUM_TILES_D - 1) begin
                            if (tile_col == NUM_TILES_H - 1) begin
                                phase_up   <= 1'b0;
                                phase_down <= 1'b1;
                                tile_col   <= '0;
                                inner_idx_wide <= '0;
                                state <= S_DOWN_REQ_RELU;
                            end else begin
                                tile_col   <= tile_col + 1;
                                inner_idx  <= '0;
                                state      <= S_UP_REQ_NORM;
                            end
                        end else begin
                            inner_idx <= inner_idx + 1;
                            state     <= S_UP_REQ_NORM;
                        end
                    end
                end

                // =====================================================
                // DOWN PHASE: Unchanged from original fetch_addr_gen
                // =====================================================
                S_DOWN_REQ_RELU: begin
                    relu_rd_addr <= inner_idx_wide[`CLOG2_MIN1(NUM_TILES_H)-1:0];
                    relu_rd_en   <= 1'b1;
                    state        <= S_DOWN_RELAY;
                end

                S_DOWN_RELAY: begin
                    state <= S_DOWN_WAIT_RELU;
                end

                S_DOWN_WAIT_RELU: begin
                    relu_tile_buf <= relu_rd_data;
                    axi_req_addr  <= WDOWN_BASE +
                                    (inner_idx_wide * NUM_TILES_D + tile_col) * WTILE_BYTES;
                    axi_req_len   <= weight_burst_len(0);
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state         <= S_DOWN_REQ_WGT;
                end

                S_DOWN_REQ_WGT: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        // Preserve the absolute tile beat index across
                        // continuation bursts.
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
                            if (beat_cnt == WEIGHT_BEATS - 1) begin
                                state <= S_DOWN_SEND;
                            end else begin
                                // Continue the current down-projection
                                // weight tile with another legal AXI burst.
                                axi_req_addr <= WDOWN_BASE +
                                    (inner_idx_wide * NUM_TILES_D + tile_col) * WTILE_BYTES +
                                    (beat_cnt + 1) * AXI_BEAT_BYTES;
                                axi_req_len  <= weight_burst_len(beat_cnt + 1);
                                // Keep the continuation request asserted
                                // until the AXI master returns to IDLE.
                                axi_req_valid <= 1'b1;
                                state <= S_DOWN_REQ_WGT;
                            end
                        end
                    end
                end

                S_DOWN_SEND: begin
                    if (down_ready) begin
                        down_relu_tile   <= relu_tile_buf;
                        down_weight_tile <= weight_tile_buf;
                        down_tile_col    <= tile_col[`CLOG2_MIN1(NUM_TILES_D)-1:0];
                        down_inner_idx   <= inner_idx_wide[`CLOG2_MIN1(NUM_TILES_H)-1:0];
                        down_is_last     <= (inner_idx_wide == NUM_TILES_H - 1);
                        down_valid       <= 1'b1;

                        if (inner_idx_wide == NUM_TILES_H - 1) begin
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

                S_DONE: begin
                    done <= 1'b1;
                    if (start) begin
                        done           <= 1'b0;
                        phase_up       <= 1'b1;
                        phase_down     <= 1'b0;
                        tile_col       <= '0;
                        inner_idx      <= '0;
                        inner_idx_wide <= '0;
                        state          <= S_UP_REQ_NORM;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
