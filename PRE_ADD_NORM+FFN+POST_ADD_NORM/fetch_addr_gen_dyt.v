//============================================================================
// fetch_addr_gen_dyt.v — Address generator for FFN with DyT normalization
//============================================================================
// UP phase: reads input tiles from local norm_input_bram (not AXI); weight
// tiles fetched via AXI. DOWN phase: reads ReLU tiles from local BRAM and
// weights from AXI.
//============================================================================

module fetch_addr_gen_dyt #(
    parameter D          = 256,
    parameter M          = 16,
    parameter DATA_W     = 16,
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    output reg                           done,
    output reg                           phase_up,
    output reg                           phase_down,
    // AXI R master (for weights)
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,
    // Norm input BRAM read
    output reg                           norm_rd_en,
    output reg  [$clog2(D/M)-1:0]       norm_rd_addr,
    input  wire [M*DATA_W-1:0]          norm_rd_data,
    // ReLU BRAM read
    output reg                           relu_rd_en,
    output reg  [$clog2(4*D/M)-1:0]     relu_rd_addr,
    input  wire [M*DATA_W-1:0]          relu_rd_data,
    // To up projection
    output reg  [M*DATA_W-1:0]          up_input_tile,
    output reg  [M*M*DATA_W-1:0]        up_weight_tile,
    output reg  [$clog2(4*D/M)-1:0]     up_tile_col,
    output reg  [$clog2(D/M)-1:0]       up_inner_idx,
    output reg                           up_is_last,
    output reg                           up_valid,
    input  wire                          up_ready,
    // To down projection
    output reg  [M*DATA_W-1:0]          down_relu_tile,
    output reg  [M*M*DATA_W-1:0]        down_weight_tile,
    output reg  [$clog2(D/M)-1:0]       down_tile_col,
    output reg  [$clog2(4*D/M)-1:0]     down_inner_idx,
    output reg                           down_is_last,
    output reg                           down_valid,
    input  wire                          down_ready
);

    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;
    localparam WORDS_PER_BEAT = AXI_DATA_W / DATA_W;
    localparam WUP_BASE     = 32'h1000_0000;
    localparam WDOWN_BASE   = 32'h2000_0000;
    localparam WTILE_BYTES  = M*M*(DATA_W/8);
    localparam WEIGHT_BEATS = (M*M + WORDS_PER_BEAT - 1) / WORDS_PER_BEAT;

    function [2:0] axi_size;
        input integer bytes;
        integer s;
        begin s=0; while ((1<<s)<bytes) s=s+1; axi_size=s[2:0]; end
    endfunction
    localparam AXI_SZ = axi_size(AXI_DATA_W/8);

    localparam S_IDLE=4'd0, S_UP_REQ_NORM=4'd1, S_UP_RELAY_NORM=4'd2,
               S_UP_WAIT_NORM=4'd3, S_UP_REQ_WEIGHT=4'd4,
               S_UP_WAIT_WEIGHT=4'd5, S_UP_SEND=4'd6,
               S_DOWN_REQ_RELU=4'd7, S_DOWN_RELAY=4'd8,
               S_DOWN_WAIT_RELU=4'd9, S_DOWN_REQ_WGT=4'd10,
               S_DOWN_WAIT_WGT=4'd11, S_DOWN_SEND=4'd12, S_DONE=4'd13;

    reg [3:0] state;
    reg [$clog2(NUM_TILES_4D)-1:0] tile_col;
    reg [$clog2(NUM_TILES_D)-1:0]  inner_idx;
    reg [$clog2(NUM_TILES_4D)-1:0] inner_idx_wide;

    reg [M*DATA_W-1:0]    input_tile_buf;
    reg [M*M*DATA_W-1:0]  weight_tile_buf;
    reg [M*DATA_W-1:0]    relu_tile_buf;
    reg [$clog2(M*M)-1:0] beat_cnt;

    integer k;

    assign axi_resp_ready = (state == S_UP_WAIT_WEIGHT) ||
                            (state == S_DOWN_WAIT_WGT);

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
            axi_req_valid<= 1'b0;

            case (state)

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

                // ---- UP phase: local BRAM read for norm input ----
                S_UP_REQ_NORM: begin
                    norm_rd_en   <= 1'b1;
                    norm_rd_addr <= inner_idx;
                    state        <= S_UP_RELAY_NORM;
                end
                S_UP_RELAY_NORM: state <= S_UP_WAIT_NORM;
                S_UP_WAIT_NORM: begin
                    input_tile_buf <= norm_rd_data;
                    axi_req_addr   <= WUP_BASE +
                        (inner_idx * NUM_TILES_4D + tile_col) * WTILE_BYTES;
                    axi_req_len    <= WEIGHT_BEATS - 1;
                    axi_req_size   <= AXI_SZ;
                    axi_req_valid  <= 1'b1;
                    beat_cnt       <= '0;
                    state          <= S_UP_REQ_WEIGHT;
                end

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
                            if (beat_cnt*WORDS_PER_BEAT + k < M*M)
                                weight_tile_buf[(beat_cnt*WORDS_PER_BEAT+k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) state <= S_UP_SEND;
                    end
                end

                S_UP_SEND: begin
                    if (up_ready) begin
                        up_input_tile <= input_tile_buf;
                        up_weight_tile<= weight_tile_buf;
                        up_tile_col   <= tile_col;
                        up_inner_idx  <= inner_idx;
                        up_is_last    <= (inner_idx == NUM_TILES_D - 1);
                        up_valid      <= 1'b1;
                        if (inner_idx == NUM_TILES_D - 1) begin
                            if (tile_col == NUM_TILES_4D - 1) begin
                                phase_up       <= 1'b0;
                                phase_down     <= 1'b1;
                                tile_col       <= '0;
                                inner_idx_wide <= '0;
                                state          <= S_DOWN_REQ_RELU;
                            end else begin
                                tile_col  <= tile_col + 1;
                                inner_idx <= '0;
                                state     <= S_UP_REQ_NORM;
                            end
                        end else begin
                            inner_idx <= inner_idx + 1;
                            state     <= S_UP_REQ_NORM;
                        end
                    end
                end

                // ---- DOWN phase ----
                S_DOWN_REQ_RELU: begin
                    relu_rd_addr <= inner_idx_wide;
                    relu_rd_en   <= 1'b1;
                    state        <= S_DOWN_RELAY;
                end
                S_DOWN_RELAY: state <= S_DOWN_WAIT_RELU;
                S_DOWN_WAIT_RELU: begin
                    relu_tile_buf <= relu_rd_data;
                    axi_req_addr  <= WDOWN_BASE +
                        (inner_idx_wide * NUM_TILES_D + tile_col) * WTILE_BYTES;
                    axi_req_len   <= WEIGHT_BEATS - 1;
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state         <= S_DOWN_REQ_WGT;
                end

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
                            if (beat_cnt*WORDS_PER_BEAT + k < M*M)
                                weight_tile_buf[(beat_cnt*WORDS_PER_BEAT+k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) state <= S_DOWN_SEND;
                    end
                end

                S_DOWN_SEND: begin
                    if (down_ready) begin
                        down_relu_tile  <= relu_tile_buf;
                        down_weight_tile<= weight_tile_buf;
                        down_tile_col   <= tile_col[$clog2(NUM_TILES_D)-1:0];
                        down_inner_idx  <= inner_idx_wide;
                        down_is_last    <= (inner_idx_wide == NUM_TILES_4D - 1);
                        down_valid      <= 1'b1;
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

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
