//============================================================================
// add_dyt_post.v — Post-Add-DyT Stage (reads from internal BRAMs)
//============================================================================
// Computes: y_out[k] = γ₂[k] * tanh(α₂ · (ffn_out[k] + residual[k])) + β₂[k]
//
// Different from add_dyt_stage (pre-DyT):
//   - ffn_out comes from the FFN output BRAM (internal, not AXI)
//   - residual comes from the norm_input_bram or residual_bram (internal)
//   - γ₂ and β₂ are fetched from AXI
//   - Final output is streamed via a serializer (not written to BRAM)
//
// Architecture:
//   For each tile t = 0..NUM_TILES_D-1:
//     1. Read ffn_output_tile from ffn_output_bram (local BRAM, 2-cycle latency)
//     2. Read residual_tile from residual_bram or norm_input_bram (local BRAM)
//     3. Fetch γ₂_tile and β₂_tile from AXI
//     4. Compute z = ffn_output + residual (M parallel adders)
//     5. For each k: s = α₂*z[k], tanh_val = LUT(s), y[k] = γ₂[k]*tanh_val+β₂[k]
//     6. Stream y_tile as AXI_DATA_W-bit beats
//============================================================================

module add_dyt_post #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter N          = 1,
    parameter SINGLE_VECTOR = 0,
    parameter DATA_W     = 16,
    parameter FRAC_W     = 8,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    input  wire [`CLOG2_MIN1(N)-1:0]     seq_idx,
    output reg                           done,

    // ---- AXI4 Read Master Interface (for γ₂, β₂) ----
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,

    // ---- α₂ (learnable scalar) — Q8.8 format ----
    input  wire signed [DATA_W-1:0]     alpha,

    // ---- FFN Output BRAM Read ----
    output reg                           ffn_rd_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       ffn_rd_addr,
    input  wire [M*DATA_W-1:0]          ffn_rd_data,

    // ---- Residual BRAM Read (y_in from pre-DyT) ----
    output reg                           res_rd_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       res_rd_addr,
    input  wire [M*DATA_W-1:0]          res_rd_data,

    // ---- Streamed Output (final) ----
    output reg  [AXI_DATA_W-1:0]        out_data,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       out_addr,
    output reg                           out_valid,
    output reg                           out_last,
    output reg                           out_row_last,
    output reg [`CLOG2_MIN1(N)-1:0]      out_seq_idx
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_TILES   = D / M;
    localparam ELEMS_PER_BEAT = AXI_DATA_W / DATA_W;
    localparam BEATS_PER_TILE = M / ELEMS_PER_BEAT;

    function [2:0] axi_size;
        input integer bytes;
        integer s;
        begin s = 0; while ((1 << s) < bytes) s = s + 1; axi_size = s[2:0]; end
    endfunction
    localparam AXI_SZ = axi_size(AXI_DATA_W / 8);

    // Memory base addresses for post-DyT parameters
    localparam GAMMA2_BASE  = 32'h3000_0000;
    localparam BETA2_BASE   = 32'h3100_0000;
    localparam TILE_BYTES   = M * (DATA_W / 8);
    localparam INPUT_BEATS  = (M + ELEMS_PER_BEAT - 1) / ELEMS_PER_BEAT;

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE         = 5'd0;
    localparam S_REQ_FFN      = 5'd1;    // Read FFN output from local BRAM
    localparam S_RELAY_FFN    = 5'd2;    // BRAM latency cycle 1
    localparam S_WAIT_FFN     = 5'd3;    // BRAM data available
    localparam S_REQ_RES      = 5'd4;    // Read residual from local BRAM
    localparam S_RELAY_RES    = 5'd5;
    localparam S_WAIT_RES     = 5'd6;
    localparam S_REQ_GAM      = 5'd7;    // AXI: γ₂ tile
    localparam S_WAIT_GAM     = 5'd8;
    localparam S_REQ_BET      = 5'd9;    // AXI: β₂ tile
    localparam S_WAIT_BET     = 5'd10;
    localparam S_ADD          = 5'd11;    // z = ffn_out + residual
    localparam S_DSP_SCALE    = 5'd12;   // s[k] = α₂ * z[k]
    localparam S_DSP_TANH     = 5'd13;   // tanh LUT output
    localparam S_DSP_AFFINE   = 5'd14;   // y[k] = γ₂[k]*tanh+β₂[k]
    localparam S_ELEM_DONE    = 5'd15;   // Store y[k]
    localparam S_STREAM       = 5'd16;   // Stream y_tile as beats
    localparam S_NEXT         = 5'd17;
    localparam S_DONE         = 5'd18;

    reg [4:0] state;
    reg [`CLOG2_MIN1(NUM_TILES)-1:0]  tile_idx;
    reg [`CLOG2_MIN1(M)-1:0]          elem_idx;
    reg [`CLOG2_MIN1(BEATS_PER_TILE)-1:0] beat_cnt;
    reg [`CLOG2_MIN1(BEATS_PER_TILE)-1:0] stream_beat_cnt;

    // Data buffers
    reg [M*DATA_W-1:0]   ffn_tile_buf;
    reg [M*DATA_W-1:0]   res_tile_buf;
    reg [M*DATA_W-1:0]   gamma_tile_buf;
    reg [M*DATA_W-1:0]   beta_tile_buf;
    reg [M*DATA_W-1:0]   z_tile_buf;
    reg [M*DATA_W-1:0]   y_tile_buf;

    // Pipeline registers
    reg signed [2*DATA_W-1:0] s_elem_reg;
    reg signed [DATA_W-1:0]   tanh_val_reg;
    reg signed [DATA_W-1:0]   gamma_elem_reg;
    reg signed [DATA_W-1:0]   beta_elem_reg;

    // AXI resp ready
    assign axi_resp_ready = (state == S_WAIT_GAM) || (state == S_WAIT_BET);

    // -------------------------------------------------------------------
    // Parallel addition: z[k] = ffn_out[k] + residual[k]
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
    // DSP1: s[k] = α₂ * z[k]  (Q8.8 × Q8.8 → Q16.16)
    // Must be declared BEFORE tanh_lut instance and s_product port
    // connection, otherwise the port connection creates an implicit
    // 1-bit wire for s_product, causing Vivado IPDW warning.
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]   z_elem_wire = z_tile_buf[elem_idx*DATA_W +: DATA_W];
    wire signed [2*DATA_W-1:0] s_product   = alpha * z_elem_wire;

    // -------------------------------------------------------------------
    // Tanh LUT instance
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]  tanh_lut_out;
    wire                      tanh_lut_valid;

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
        .s_val      (s_product),        // Use combinational product (not registered s_elem_reg)
        .tanh_out   (tanh_lut_out),
        .tanh_valid (tanh_lut_valid)
    );

    // DSP2: y[k] = γ₂[k] * tanh_val + β₂[k]
    wire signed [2*DATA_W-1:0] y_prod = gamma_elem_reg * tanh_val_reg;

    // Use proper signed arithmetic — concatenation is unsigned in Verilog!
    wire signed [2*DATA_W-1:0] beta_shifted;
    assign beta_shifted = beta_elem_reg << FRAC_W;  // β in Q16.16 (sign-extended by <<)

    wire signed [2*DATA_W:0]   y_sum_ext;
    assign y_sum_ext = y_prod + beta_shifted;

    // Saturation thresholds in wide format to avoid overflow
    // MAX_VAL<<FRAC_W = 32767<<8 = 8388352, which doesn't fit in 16 bits
    // Using wide [2*DATA_W-1:0] thresholds prevents overflow
    localparam signed [2*DATA_W-1:0] MAX_THRESH = (1 << (DATA_W-1+FRAC_W)) - 1;  // = 8388352
    localparam signed [2*DATA_W-1:0] MIN_THRESH = -(1 << (DATA_W-1+FRAC_W));       // = -8388608
    localparam signed [DATA_W-1:0]   MAX_OUT    = (1 << (DATA_W-1)) - 1;            // = 32767
    localparam signed [DATA_W-1:0]   MIN_OUT    = -(1 << (DATA_W-1));                // = -32768

    wire signed [DATA_W-1:0] y_elem_sat;
    assign y_elem_sat = (y_sum_ext > MAX_THRESH) ? MAX_OUT :
                        (y_sum_ext < MIN_THRESH) ? MIN_OUT :
                        y_sum_ext[FRAC_W +: DATA_W];

    // -------------------------------------------------------------------
    // Main state machine
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
            s_elem_reg      <= '0;
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
            out_row_last    <= 1'b0;
            out_seq_idx     <= '0;
        end else begin
            axi_req_valid <= 1'b0;
            ffn_rd_en     <= 1'b0;
            res_rd_en     <= 1'b0;
            out_valid     <= 1'b0;
            out_row_last  <= 1'b0;

            case (state)
                // =====================================================
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        tile_idx  <= '0;
                        elem_idx  <= '0;
                        // Start by reading FFN output from local BRAM
                        ffn_rd_en   <= 1'b1;
                        ffn_rd_addr <= '0;
                        state <= S_REQ_FFN;
                    end
                end

                // =====================================================
                // Read ffn_output from local BRAM (2-cycle latency)
                // =====================================================
                S_REQ_FFN: begin
                    // ffn_rd_en was set in S_IDLE or S_NEXT
                    state <= S_RELAY_FFN;
                end

                S_RELAY_FFN: begin
                    // BRAM sees rd_en, latches data
                    state <= S_WAIT_FFN;
                end

                S_WAIT_FFN: begin
                    ffn_tile_buf <= ffn_rd_data;
                    // Read residual from norm_input_bram
                    res_rd_en   <= 1'b1;
                    res_rd_addr <= tile_idx;
                    state <= S_REQ_RES;
                end

                // =====================================================
                // Read residual from local BRAM (2-cycle latency)
                // =====================================================
                S_REQ_RES: begin
                    // res_rd_en was set in S_WAIT_FFN
                    state <= S_RELAY_RES;
                end

                S_RELAY_RES: begin
                    state <= S_WAIT_RES;
                end

                S_WAIT_RES: begin
                    res_tile_buf <= res_rd_data;
                    // Fetch γ₂ from AXI
                    axi_req_addr  <= GAMMA2_BASE + tile_idx * TILE_BYTES;
                    axi_req_len   <= INPUT_BEATS - 1;
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state <= S_REQ_GAM;
                end

                // =====================================================
                // AXI: γ₂ tile
                // =====================================================
                S_REQ_GAM: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt <= '0;
                        state <= S_WAIT_GAM;
                    end
                end

                S_WAIT_GAM: begin
                    if (axi_resp_valid) begin
                        for (j = 0; j < ELEMS_PER_BEAT; j = j + 1) begin
                            if (beat_cnt * ELEMS_PER_BEAT + j < M)
                                gamma_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            axi_req_addr  <= BETA2_BASE + tile_idx * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state <= S_REQ_BET;
                        end
                    end
                end

                // =====================================================
                // AXI: β₂ tile
                // =====================================================
                S_REQ_BET: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt <= '0;
                        state <= S_WAIT_BET;
                    end
                end

                S_WAIT_BET: begin
                    if (axi_resp_valid) begin
                        for (j = 0; j < ELEMS_PER_BEAT; j = j + 1) begin
                            if (beat_cnt * ELEMS_PER_BEAT + j < M)
                                beta_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            state <= S_ADD;
                        end
                    end
                end

                // =====================================================
                // z = ffn_out + residual (M parallel adders)
                // =====================================================
                S_ADD: begin
                    z_tile_buf <= z_comb;
                    elem_idx   <= '0;
                    y_tile_buf <= '0;
                    state <= S_DSP_SCALE;
                end

                // =====================================================
                // Sequential element processing: α₂*z → tanh → γ₂*t+β₂
                // =====================================================
                S_DSP_SCALE: begin
                    s_elem_reg     <= s_product;
                    gamma_elem_reg <= gamma_tile_buf[elem_idx*DATA_W +: DATA_W];
                    beta_elem_reg  <= beta_tile_buf[elem_idx*DATA_W +: DATA_W];
                    state <= S_DSP_TANH;
                end

                S_DSP_TANH: begin
                    tanh_val_reg <= tanh_lut_out;
                    state <= S_DSP_AFFINE;
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
                        state <= S_DSP_SCALE;
                    end
                end

                // =====================================================
                // Stream y_tile as AXI_DATA_W-bit beats
                // =====================================================
                S_STREAM: begin
                    out_data  <= y_tile_buf[stream_beat_cnt * AXI_DATA_W +: AXI_DATA_W];
                    out_addr    <= tile_idx;
                    out_seq_idx <= seq_idx;
                    out_valid   <= 1'b1;
                    out_row_last <= (stream_beat_cnt == BEATS_PER_TILE - 1) &&
                                    (tile_idx == NUM_TILES - 1);
                    out_last  <= (stream_beat_cnt == BEATS_PER_TILE - 1) &&
                                 (tile_idx == NUM_TILES - 1) &&
                                 (SINGLE_VECTOR || (seq_idx == N - 1));

                    if (stream_beat_cnt == BEATS_PER_TILE - 1) begin
                        state <= S_NEXT;
                    end else begin
                        stream_beat_cnt <= stream_beat_cnt + 1;
                    end
                end

                // =====================================================
                // Next tile or finish
                // =====================================================
                S_NEXT: begin
                    if (tile_idx == NUM_TILES - 1) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        tile_idx <= tile_idx + 1;
                        elem_idx <= '0;
                        ffn_rd_en   <= 1'b1;
                        ffn_rd_addr <= tile_idx + 1;
                        state <= S_REQ_FFN;
                    end
                end

                // =====================================================
                S_DONE: begin
                    done <= 1'b1;
                    if (start) begin
                        // Restart post-DyT for the next sequence element.
                        done        <= 1'b0;
                        tile_idx    <= '0;
                        elem_idx    <= '0;
                        ffn_rd_en   <= 1'b1;
                        ffn_rd_addr <= '0;
                        state       <= S_REQ_FFN;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
