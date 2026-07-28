//============================================================================
// add_dyt_stage.v — Add + Dynamic Tanh (DyT) Normalization Stage
//============================================================================
// Computes: y[k] = gamma[k] * tanh(alpha * (x[k] + residual[k])) + beta[k]
//
// Drop-in substitute for LayerNorm per "Transformers without Normalization"
// (Zhu et al., 2025). No mean/variance, no division, no sqrt.
//
// Pipeline (per-element, 2+1 cycles):
//   Cycle N  (S_DSP_SCALE) : compute s = alpha * z[k], launch BRAM read
//   Cycle N+1(S_DSP_TANH)  : BRAM returns tanh(s[k])
//   Cycle N+2(S_DSP_AFFINE): compute y = gamma * tanh + beta, store
//
// Fixed-point:
//   - alpha, z, gamma, beta, tanh_val, y are Q8.8 signed (DATA_W=16, FRAC_W=8)
//   - Products are Q16.16 (2*DATA_W bits)
//   - beta is explicitly sign-extended to 2*DATA_W BEFORE shifting left, to
//     avoid Verilog's width truncation pitfall (beta_elem_reg << FRAC_W would
//     otherwise operate on 16 bits, drop the upper byte, and extend zero).
//============================================================================

module add_dyt_stage #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter DATA_W     = 16,
    parameter FRAC_W     = 8,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    // Control
    input  wire                          start,
    output reg                           done,
    // AXI4 Read Master Interface
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,
    // alpha (learnable scalar) — Q8.8
    input  wire signed [DATA_W-1:0]     alpha,
    // Output BRAM write (y = DyT(x+residual))
    output reg                           out_wr_en,
    output reg  [$clog2(D/M)-1:0]       out_wr_addr,
    output reg  [M*DATA_W-1:0]          out_wr_data,
    // Residual BRAM write (z = x + residual)
    output reg                           res_wr_en,
    output reg  [$clog2(D/M)-1:0]       res_wr_addr,
    output reg  [M*DATA_W-1:0]          res_wr_data,
    // Streaming output
    output reg  [M*DATA_W-1:0]          stream_data,
    output reg  [$clog2(D/M)-1:0]       stream_addr,
    output reg                           stream_valid,
    output reg                           stream_last
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

    // Memory base addresses
    localparam INPUT_BASE    = 32'h0000_0000;
    localparam RESIDUAL_BASE = 32'h0800_0000;
    localparam GAMMA_BASE    = 32'h0C00_0000;
    localparam BETA_BASE     = 32'h0D00_0000;

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE       = 5'd0;
    localparam S_REQ_X      = 5'd1;
    localparam S_WAIT_X     = 5'd2;
    localparam S_REQ_RES    = 5'd3;
    localparam S_WAIT_RES   = 5'd4;
    localparam S_REQ_GAM    = 5'd5;
    localparam S_WAIT_GAM   = 5'd6;
    localparam S_REQ_BET    = 5'd7;
    localparam S_WAIT_BET   = 5'd8;
    localparam S_ADD        = 5'd9;
    localparam S_DSP_SCALE  = 5'd10;
    localparam S_DSP_TANH   = 5'd11;
    localparam S_DSP_AFFINE = 5'd12;
    localparam S_ELEM_DONE  = 5'd13;
    localparam S_WRITE      = 5'd14;
    localparam S_NEXT       = 5'd15;
    localparam S_DONE       = 5'd16;

    reg [4:0] state;
    reg [$clog2(NUM_TILES)-1:0] tile_idx;
    reg [$clog2(M)-1:0]         elem_idx;
    reg [$clog2(BEATS_PER_TILE)-1:0] beat_cnt;

    reg [M*DATA_W-1:0] x_tile_buf, res_tile_buf, gamma_tile_buf,
                       beta_tile_buf, z_tile_buf, y_tile_buf;

    // Pipeline registers
    reg signed [DATA_W-1:0] tanh_val_reg;
    reg signed [DATA_W-1:0] gamma_elem_reg;
    reg signed [DATA_W-1:0] beta_elem_reg;

    // AXI resp ready
    assign axi_resp_ready = (state == S_WAIT_X)  ||
                            (state == S_WAIT_RES)||
                            (state == S_WAIT_GAM)||
                            (state == S_WAIT_BET);

    // -------------------------------------------------------------------
    // Parallel addition  z = x + residual
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0] z_comb;
    integer j;
    always @(*) begin
        for (j = 0; j < M; j = j + 1) begin
            z_comb[j*DATA_W +: DATA_W] =
                x_tile_buf[j*DATA_W +: DATA_W] + res_tile_buf[j*DATA_W +: DATA_W];
        end
    end

    // -------------------------------------------------------------------
    // DSP1: s[k] = alpha * z[k]   (Q8.8 * Q8.8 -> Q16.16)
    //
    // Cast to 2*DATA_W WIDTH BEFORE multiplying to avoid Verilog's
    // "result width = max operand width" pitfall: {N,N}*N -> N bits.
    // We explicitly widen alpha/z to 2*DATA_W so the product is full-width.
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]    z_elem_wire = z_tile_buf[elem_idx*DATA_W +: DATA_W];
    wire signed [2*DATA_W-1:0]  alpha_w = {{DATA_W{alpha[DATA_W-1]}}, alpha};
    wire signed [2*DATA_W-1:0]  z_elem_w = {{DATA_W{z_elem_wire[DATA_W-1]}}, z_elem_wire};
    wire signed [2*DATA_W-1:0]  s_product = alpha_w * z_elem_w;

    // -------------------------------------------------------------------
    // Tanh LUT instance
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
    // DSP2: y[k] = gamma[k] * tanh_val + beta[k]
    //
    //   gamma * tanh_val  : Q8.8 * Q8.8 -> Q16.16
    //   beta (Q8.8) is sign-extended to Q16.16 (i.e. << FRAC_W after extend)
    //   Sum Q16.16 + Q16.16 -> Q17.16  (one extra sign/carry bit)
    //   Extract DATA_W bits starting at FRAC_W, with saturation.
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]   gamma_w_curr = gamma_tile_buf[elem_idx*DATA_W +: DATA_W];
    wire signed [DATA_W-1:0]   beta_w_curr  = beta_tile_buf [elem_idx*DATA_W +: DATA_W];

    // Widen operands to 2*DATA_W BEFORE multiplying to keep full product
    wire signed [2*DATA_W-1:0] gamma_ext = {{DATA_W{gamma_elem_reg[DATA_W-1]}}, gamma_elem_reg};
    wire signed [2*DATA_W-1:0] tanh_ext  = {{DATA_W{tanh_val_reg[DATA_W-1]}},  tanh_val_reg};
    wire signed [2*DATA_W-1:0] y_prod    = gamma_ext * tanh_ext;

    // CRITICAL BUG FIX (width extension before shift):
    // The original code did `beta_elem_reg << FRAC_W` with beta_elem_reg at
    // DATA_W bits, so the << operated on 16 bits, truncated the high byte,
    // and then zero-extended to 32 bits on assignment. We must first sign-
    // extend beta to 2*DATA_W, THEN shift.
    wire signed [2*DATA_W-1:0] beta_ext   = {{DATA_W{beta_elem_reg[DATA_W-1]}}, beta_elem_reg};
    wire signed [2*DATA_W-1:0] beta_shifted = beta_ext <<< FRAC_W;

    // Sum with carry bit
    wire signed [2*DATA_W:0] y_sum_ext = $signed({y_prod[2*DATA_W-1], y_prod}) +
                                         $signed({beta_shifted[2*DATA_W-1], beta_shifted});

    // Saturation thresholds (Q16.16).  127.996 = 0x7FFF << 8 = 0x007FFF00
    localparam signed [2*DATA_W-1:0] MAX_THRESH = ((1 << (DATA_W-1)) - 1) << FRAC_W;
    localparam signed [2*DATA_W-1:0] MIN_THRESH = -(1 << (DATA_W-1)) << FRAC_W;
    localparam signed [DATA_W-1:0]   MAX_OUT    = (1 << (DATA_W-1)) - 1;
    localparam signed [DATA_W-1:0]   MIN_OUT    = -(1 << (DATA_W-1));

    wire signed [DATA_W-1:0] y_elem_sat;
    assign y_elem_sat = (y_sum_ext >  MAX_THRESH) ? MAX_OUT :
                        (y_sum_ext <  MIN_THRESH) ? MIN_OUT :
                        $signed(y_sum_ext[FRAC_W +: DATA_W]);

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
            x_tile_buf      <= '0;
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
            out_wr_en       <= 1'b0;
            out_wr_addr     <= '0;
            out_wr_data     <= '0;
            res_wr_en       <= 1'b0;
            res_wr_addr     <= '0;
            res_wr_data     <= '0;
            stream_data     <= '0;
            stream_addr     <= '0;
            stream_valid    <= 1'b0;
            stream_last     <= 1'b0;
        end else begin
            axi_req_valid <= 1'b0;
            out_wr_en     <= 1'b0;
            res_wr_en     <= 1'b0;
            stream_valid  <= 1'b0;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        tile_idx      <= '0;
                        elem_idx      <= '0;
                        axi_req_addr  <= INPUT_BASE;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;
                        state <= S_REQ_X;
                    end
                end

                S_REQ_X: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_WAIT_X;
                    end
                end
                S_WAIT_X: begin
                    if (axi_resp_valid) begin
                        for (j = 0; j < ELEMS_PER_BEAT; j = j + 1) begin
                            if (beat_cnt * ELEMS_PER_BEAT + j < M)
                                x_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            axi_req_addr  <= RESIDUAL_BASE + tile_idx * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_REQ_RES;
                        end
                    end
                end

                S_REQ_RES: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_WAIT_RES;
                    end
                end
                S_WAIT_RES: begin
                    if (axi_resp_valid) begin
                        for (j = 0; j < ELEMS_PER_BEAT; j = j + 1) begin
                            if (beat_cnt * ELEMS_PER_BEAT + j < M)
                                res_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            axi_req_addr  <= GAMMA_BASE + tile_idx * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_REQ_GAM;
                        end
                    end
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
                            if (beat_cnt * ELEMS_PER_BEAT + j < M)
                                gamma_tile_buf[(beat_cnt*ELEMS_PER_BEAT+j)*DATA_W +: DATA_W]
                                    <= axi_resp_data[j*DATA_W +: DATA_W];
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            axi_req_addr  <= BETA_BASE + tile_idx * TILE_BYTES;
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

                // -----------------------------------------------------
                // COMPUTE
                // -----------------------------------------------------
                S_ADD: begin
                    z_tile_buf <= z_comb;
                    elem_idx   <= '0;
                    y_tile_buf <= '0;
                    state      <= S_DSP_SCALE;
                end

                S_DSP_SCALE: begin
                    // Capture gamma/beta for THIS element so they line up
                    // with tanh_lut_out one cycle later.
                    gamma_elem_reg <= gamma_w_curr;
                    beta_elem_reg  <= beta_w_curr;
                    state <= S_DSP_TANH;
                end

                S_DSP_TANH: begin
                    // BRAM LUT output is now valid (1-cycle latency)
                    tanh_val_reg <= tanh_lut_out;
                    state <= S_DSP_AFFINE;
                end

                S_DSP_AFFINE: begin
                    y_tile_buf[elem_idx*DATA_W +: DATA_W] <= y_elem_sat;
                    state <= S_ELEM_DONE;
                end

                S_ELEM_DONE: begin
                    if (elem_idx == M - 1) begin
                        state <= S_WRITE;
                    end else begin
                        elem_idx <= elem_idx + 1;
                        state    <= S_DSP_SCALE;
                    end
                end

                S_WRITE: begin
                    out_wr_en    <= 1'b1;
                    out_wr_addr  <= tile_idx;
                    out_wr_data  <= y_tile_buf;
                    res_wr_en    <= 1'b1;
                    res_wr_addr  <= tile_idx;
                    res_wr_data  <= z_tile_buf;
                    stream_data  <= y_tile_buf;
                    stream_addr  <= tile_idx;
                    stream_valid <= 1'b1;
                    stream_last  <= (tile_idx == NUM_TILES - 1);
                    state        <= S_NEXT;
                end

                S_NEXT: begin
                    if (tile_idx == NUM_TILES - 1) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        tile_idx      <= tile_idx + 1;
                        elem_idx      <= '0;
                        axi_req_addr  <= INPUT_BASE + (tile_idx + 1) * TILE_BYTES;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;
                        state         <= S_REQ_X;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
