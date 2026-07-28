//============================================================================
// add_dyt_post.v — Post-Add-DyT Stage (reads FFN output from internal BRAMs)
//============================================================================
// Computes: y_out[k] = gamma2[k] * tanh(alpha2 * (ffn_out[k] + residual[k])) + beta2[k]
//
// Inputs:
//   - ffn_out   : from ffn_output_bram (local)
//   - residual  : from residual_bram written by pre-DyT (holds z1 = x+res1)
//   - gamma2, beta2 : from AXI
//
// Output: streamed in AXI_DATA_W-wide beats (final transformer output).
//
// The MAC pipeline uses the same width-safe signed arithmetic as
// add_dyt_stage (see comments there for the multiplication/shift pitfall).
//============================================================================

module add_dyt_post #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter DATA_W     = 16,
    parameter FRAC_W     = 8,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    output reg                           done,
    // AXI4 Read Master (gamma2, beta2)
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,
    // alpha2  (Q8.8)
    input  wire signed [DATA_W-1:0]     alpha,
    // FFN Output BRAM
    output reg                           ffn_rd_en,
    output reg  [$clog2(D/M)-1:0]       ffn_rd_addr,
    input  wire [M*DATA_W-1:0]          ffn_rd_data,
    // Residual BRAM (z1 from pre-DyT)
    output reg                           res_rd_en,
    output reg  [$clog2(D/M)-1:0]       res_rd_addr,
    input  wire [M*DATA_W-1:0]          res_rd_data,
    // Streamed output
    output reg  [AXI_DATA_W-1:0]        out_data,
    output reg  [$clog2(D/M)-1:0]       out_addr,
    output reg                           out_valid,
    output reg                           out_last
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_TILES    = D / M;
    localparam ELEMS_PER_BEAT = AXI_DATA_W / DATA_W;
    localparam BEATS_PER_TILE = M / ELEMS_PER_BEAT;
    localparam INPUT_BEATS  = (M + ELEMS_PER_BEAT - 1) / ELEMS_PER_BEAT;
    localparam TILE_BYTES   = M * (DATA_W/8);

    function [2:0] axi_size;
        input integer bytes;
        integer s;
        begin s=0; while ((1<<s)<bytes) s=s+1; axi_size=s[2:0]; end
    endfunction
    localparam AXI_SZ = axi_size(AXI_DATA_W/8);

    localparam GAMMA2_BASE = 32'h3000_0000;
    localparam BETA2_BASE  = 32'h3100_0000;

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE       = 5'd0;
    localparam S_REQ_FFN    = 5'd1;
    localparam S_RELAY_FFN  = 5'd2;
    localparam S_WAIT_FFN   = 5'd3;
    localparam S_REQ_RES    = 5'd4;
    localparam S_RELAY_RES  = 5'd5;
    localparam S_WAIT_RES   = 5'd6;
    localparam S_REQ_GAM    = 5'd7;
    localparam S_WAIT_GAM   = 5'd8;
    localparam S_REQ_BET    = 5'd9;
    localparam S_WAIT_BET   = 5'd10;
    localparam S_ADD        = 5'd11;
    localparam S_DSP_SCALE  = 5'd12;
    localparam S_DSP_TANH   = 5'd13;
    localparam S_DSP_AFFINE = 5'd14;
    localparam S_ELEM_DONE  = 5'd15;
    localparam S_STREAM     = 5'd16;
    localparam S_NEXT       = 5'd17;
    localparam S_DONE       = 5'd18;

    reg [4:0] state;
    reg [$clog2(NUM_TILES)-1:0] tile_idx;
    reg [$clog2(M)-1:0]         elem_idx;
    reg [$clog2(BEATS_PER_TILE)-1:0] beat_cnt;
    reg [$clog2(BEATS_PER_TILE)-1:0] stream_beat_cnt;

    reg [M*DATA_W-1:0] ffn_tile_buf, res_tile_buf, gamma_tile_buf,
                       beta_tile_buf, z_tile_buf, y_tile_buf;

    reg signed [DATA_W-1:0] tanh_val_reg, gamma_elem_reg, beta_elem_reg;

    assign axi_resp_ready = (state == S_WAIT_GAM) || (state == S_WAIT_BET);

    // -------------------------------------------------------------------
    // Parallel addition
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0] z_comb;
    integer j;
    always @(*) begin
        for (j = 0; j < M; j = j + 1) begin
            z_comb[j*DATA_W +: DATA_W] =
                ffn_tile_buf[j*DATA_W +: DATA_W] + res_tile_buf[j*DATA_W +: DATA_W];
        end
    end

    // -------------------------------------------------------------------
    // DSP1: s[k] = alpha * z[k] — width-safe multiply
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]   z_elem_wire = z_tile_buf[elem_idx*DATA_W +: DATA_W];
    wire signed [2*DATA_W-1:0] alpha_w    = {{DATA_W{alpha[DATA_W-1]}},     alpha};
    wire signed [2*DATA_W-1:0] z_elem_w   = {{DATA_W{z_elem_wire[DATA_W-1]}}, z_elem_wire};
    wire signed [2*DATA_W-1:0] s_product  = alpha_w * z_elem_w;

    // -------------------------------------------------------------------
    // Tanh LUT
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0] tanh_lut_out;
    wire                     tanh_lut_valid;

    tanh_lut #(
        .DATA_W        (DATA_W),
        .FRAC_W        (FRAC_W),
        .S_WIDTH       (2*DATA_W),
        .S_FRAC        (2*FRAC_W),
        .SAT_LIMIT_INT (4)
    ) u_tanh_lut (
        .clk        (clk),
        .rst_n      (rst_n),
        .rd_en      (state == S_DSP_SCALE),
        .s_val      (s_product),
        .tanh_out   (tanh_lut_out),
        .tanh_valid (tanh_lut_valid)
    );

    // -------------------------------------------------------------------
    // DSP2: y = gamma2 * tanh + beta2 (width-safe MAC, Q8.8 saturated)
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]   gamma_w_curr = gamma_tile_buf[elem_idx*DATA_W +: DATA_W];
    wire signed [DATA_W-1:0]   beta_w_curr  = beta_tile_buf [elem_idx*DATA_W +: DATA_W];

    wire signed [2*DATA_W-1:0] gamma_ext = {{DATA_W{gamma_elem_reg[DATA_W-1]}}, gamma_elem_reg};
    wire signed [2*DATA_W-1:0] tanh_ext  = {{DATA_W{tanh_val_reg[DATA_W-1]}},   tanh_val_reg};
    wire signed [2*DATA_W-1:0] y_prod    = gamma_ext * tanh_ext;

    // sign-extend THEN shift
    wire signed [2*DATA_W-1:0] beta_ext     = {{DATA_W{beta_elem_reg[DATA_W-1]}}, beta_elem_reg};
    wire signed [2*DATA_W-1:0] beta_shifted = beta_ext <<< FRAC_W;

    wire signed [2*DATA_W:0]   y_sum_ext = $signed({y_prod[2*DATA_W-1], y_prod}) +
                                           $signed({beta_shifted[2*DATA_W-1], beta_shifted});

    localparam signed [2*DATA_W-1:0] MAX_THRESH = ((1 << (DATA_W-1)) - 1) << FRAC_W;
    localparam signed [2*DATA_W-1:0] MIN_THRESH = -(1 << (DATA_W-1)) << FRAC_W;
    localparam signed [DATA_W-1:0]   MAX_OUT    = (1 << (DATA_W-1)) - 1;
    localparam signed [DATA_W-1:0]   MIN_OUT    = -(1 << (DATA_W-1));

    wire signed [DATA_W-1:0] y_elem_sat;
    assign y_elem_sat = (y_sum_ext >  MAX_THRESH) ? MAX_OUT :
                        (y_sum_ext <  MIN_THRESH) ? MIN_OUT :
                        $signed(y_sum_ext[FRAC_W +: DATA_W]);

    // -------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            tile_idx        <= '0;
            elem_idx        <= '0;
            beat_cnt        <= '0;
            stream_beat_cnt <= '0;
            ffn_tile_buf    <= '0;
            res_tile_buf    <= '0;
            gamma_tile_buf  <= '0;
            beta_tile_buf   <= '0;
            z_tile_buf      <= '0;
            y_tile_buf      <= '0;
            tanh_val_reg    <= '0;
            gamma_elem_reg  <= '0;
            beta_elem_reg   <= '0;
            axi_req_addr    <= '0;
            axi_req_len     <= '0;
            axi_req_size    <= '0;
            axi_req_valid   <= 1'b0;
            ffn_rd_en       <= 1'b0;
            ffn_rd_addr     <= '0;
            res_rd_en       <= 1'b0;
            res_rd_addr     <= '0;
            out_data        <= '0;
            out_addr        <= '0;
            out_valid       <= 1'b0;
            out_last        <= 1'b0;
        end else begin
            axi_req_valid <= 1'b0;
            ffn_rd_en     <= 1'b0;
            res_rd_en     <= 1'b0;
            out_valid     <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        tile_idx    <= '0;
                        elem_idx    <= '0;
                        ffn_rd_en   <= 1'b1;
                        ffn_rd_addr <= '0;
                        state       <= S_REQ_FFN;
                    end
                end

                // --- Local BRAM reads: FFN output (1-cycle BRAM latency,
                //     but issue/relay/wait FSM keeps 2-cycle total delay,
                //     which is harmless latency padding) ---
                S_REQ_FFN:  state <= S_RELAY_FFN;
                S_RELAY_FFN: state <= S_WAIT_FFN;
                S_WAIT_FFN: begin
                    ffn_tile_buf <= ffn_rd_data;
                    res_rd_en    <= 1'b1;
                    res_rd_addr  <= tile_idx;
                    state        <= S_REQ_RES;
                end

                S_REQ_RES:  state <= S_RELAY_RES;
                S_RELAY_RES: state <= S_WAIT_RES;
                S_WAIT_RES: begin
                    res_tile_buf  <= res_rd_data;
                    axi_req_addr  <= GAMMA2_BASE + tile_idx * TILE_BYTES;
                    axi_req_len   <= INPUT_BEATS - 1;
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state         <= S_REQ_GAM;
                end

                S_REQ_GAM: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_WAIT_GAM;
                    end
                end
                S_WAIT_GAM: begin
                    if (axi_resp_valid) begin
                        for (j = 0; j < ELEMS_PER_BEAT; j = j + 1) begin
                            if (beat_cnt*ELEMS_PER_BEAT + j < M)
                                gamma_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            axi_req_addr  <= BETA2_BASE + tile_idx * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_REQ_BET;
                        end
                    end
                end

                S_REQ_BET: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_WAIT_BET;
                    end
                end
                S_WAIT_BET: begin
                    if (axi_resp_valid) begin
                        for (j = 0; j < ELEMS_PER_BEAT; j = j + 1) begin
                            if (beat_cnt*ELEMS_PER_BEAT + j < M)
                                beta_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last)
                            state <= S_ADD;
                    end
                end

                S_ADD: begin
                    z_tile_buf <= z_comb;
                    elem_idx   <= '0;
                    y_tile_buf <= '0;
                    state      <= S_DSP_SCALE;
                end

                S_DSP_SCALE: begin
                    gamma_elem_reg <= gamma_w_curr;
                    beta_elem_reg  <= beta_w_curr;
                    state          <= S_DSP_TANH;
                end

                S_DSP_TANH: begin
                    tanh_val_reg <= tanh_lut_out;
                    state        <= S_DSP_AFFINE;
                end

                S_DSP_AFFINE: begin
                    y_tile_buf[elem_idx*DATA_W +: DATA_W] <= y_elem_sat;
                    state <= S_ELEM_DONE;
                end

                S_ELEM_DONE: begin
                    if (elem_idx == M - 1) begin
                        stream_beat_cnt <= '0;
                        state <= S_STREAM;
                    end else begin
                        elem_idx <= elem_idx + 1;
                        state    <= S_DSP_SCALE;
                    end
                end

                S_STREAM: begin
                    out_data  <= y_tile_buf[stream_beat_cnt * AXI_DATA_W +: AXI_DATA_W];
                    out_addr  <= tile_idx;
                    out_valid <= 1'b1;
                    out_last  <= (stream_beat_cnt == BEATS_PER_TILE - 1) &&
                                 (tile_idx == NUM_TILES - 1);
                    if (stream_beat_cnt == BEATS_PER_TILE - 1)
                        state <= S_NEXT;
                    else
                        stream_beat_cnt <= stream_beat_cnt + 1;
                end

                S_NEXT: begin
                    if (tile_idx == NUM_TILES - 1) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        tile_idx    <= tile_idx + 1;
                        elem_idx    <= '0;
                        ffn_rd_en   <= 1'b1;
                        ffn_rd_addr <= tile_idx + 1;
                        state       <= S_REQ_FFN;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (start) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
