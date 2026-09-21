//============================================================================
// add_dyt_stage.v — Add + Dynamic Tanh (DyT) Normalization Stage
//============================================================================
// Computes: y[k] = γ[k] * tanh(α * (x[k] + residual[k])) + β[k]
//
// This is a drop-in replacement for LayerNorm using Dynamic Tanh (DyT)
// from "Transformers without Normalization" (Zhu et al., 2025).
//
// Key advantages over LayerNorm:
//   1. NO statistics: No mean/variance computation
//   2. NO two-pass: Element-wise, tile-independent
//   3. NO division/sqrt: Just multiply + tanh LUT + multiply-add
//   4. MUCH simpler hardware: ~2 DSPs vs 64+ DSPs for LayerNorm
//
// Architecture:
//   Phase 1 (FETCH): AXI reads for x, residual, γ, β tiles
//   Phase 2 (COMPUTE): Sequential element processing through pipeline:
//     Step A: z[k] = x[k] + residual[k]  (M parallel adders, stored in z_tile_buf)
//     Step B: s = α * z[k]                (1 DSP, sequential per element)
//     Step C: tanh_val = tanh(s)           (BRAM LUT, 1-cycle latency)
//     Step D: y[k] = γ[k] * tanh_val + β[k]  (1 DSP MAC, sequential per element)
//   Phase 3 (WRITE): y_tile → output BRAM, z_tile → residual BRAM
//
// Pipeline for steps B-C-D:
//   Since tanh LUT has 1-cycle read latency, we use a simple 2-cycle
//   loop per element:
//     Cycle N  : DSP1 computes s[k] = α * z[k], sets LUT address
//     Cycle N+1: LUT output tanh_val available
//     Cycle N+2: DSP2 computes y[k] = γ[k] * tanh_val + β[k], stores result
//
//   Per element: 3 cycles. Per tile: M*3 + 1 (add) + 1 (write) = 98 cycles
//   For 64 tiles: ~6.4K cycles — negligible vs FFN's ~950K cycles.
//
// Resource usage:
//   DSP48E1  : 2 (1 for α*z, 1 for γ*tanh+β)
//   BRAM36   : 1 (tanh LUT ROM)
//   LUT      : ~500 (FSM, adders, muxes)
//   FF       : ~300 (pipeline regs, FSM)
//
// Parameters:
//   D, M, N, DATA_W, FRAC_W : Standard dimension/format; N rows are
//                              processed sequentially and x/residual are
//                              row-major in external memory.
//   AXI_DATA_W, AXI_ADDR_W : AXI bus widths
//============================================================================

module add_dyt_stage #(
    parameter D          = 2048,
    parameter M          = 32,
    parameter N          = 1,
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

    // ---- AXI4 Read Master Interface ----
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,

    // ---- α (learnable scalar) — Q8.8 format ----
    input  wire signed [DATA_W-1:0]     alpha,

    // ---- Output BRAM Write (normalized y = DyT(x+residual)) ----
    output reg                           out_wr_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       out_wr_addr,
    output reg  [M*DATA_W-1:0]          out_wr_data,

    // ---- Residual BRAM Write (z = x + residual, for post-DyT) ----
    output reg                           res_wr_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       res_wr_addr,
    output reg  [M*DATA_W-1:0]          res_wr_data,

    // ---- Streaming output (final result, for post-DyT) ----
    output reg  [M*DATA_W-1:0]          stream_data,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       stream_addr,
    output reg                           stream_valid,
    output reg                           stream_last
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

    // Memory base addresses
    localparam INPUT_BASE    = 32'h0000_0000;
    localparam RESIDUAL_BASE = 32'h0800_0000;
    localparam GAMMA_BASE    = 32'h0C00_0000;
    localparam BETA_BASE     = 32'h0D00_0000;
    localparam TILE_BYTES    = M * (DATA_W / 8);
    localparam VECTOR_BYTES  = D * (DATA_W / 8);
    localparam INPUT_BEATS   = (M + ELEMS_PER_BEAT - 1) / ELEMS_PER_BEAT;

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE        = 5'd0;
    localparam S_REQ_X       = 5'd1;
    localparam S_WAIT_X      = 5'd2;
    localparam S_REQ_RES     = 5'd3;
    localparam S_WAIT_RES    = 5'd4;
    localparam S_REQ_GAM     = 5'd5;
    localparam S_WAIT_GAM    = 5'd6;
    localparam S_REQ_BET     = 5'd7;
    localparam S_WAIT_BET    = 5'd8;
    localparam S_ADD         = 5'd9;    // z = x + residual (parallel)
    localparam S_DSP_SCALE   = 5'd10;   // s[k] = α * z[k] (DSP1 + LUT address)
    localparam S_DSP_TANH    = 5'd11;   // Wait for tanh LUT output (1-cycle latency)
    localparam S_DSP_AFFINE  = 5'd12;   // y[k] = γ[k] * tanh + β[k] (DSP2)
    localparam S_ELEM_DONE   = 5'd13;   // Store y[k], advance elem_idx
    localparam S_WRITE       = 5'd14;   // Write y_tile to BRAMs
    localparam S_NEXT        = 5'd15;
    localparam S_DONE        = 5'd16;

    reg [4:0] state;
    reg [`CLOG2_MIN1(NUM_TILES)-1:0]  tile_idx;
    reg [`CLOG2_MIN1(M)-1:0]          elem_idx;
    reg [`CLOG2_MIN1(BEATS_PER_TILE)-1:0] beat_cnt;

    // Tile data buffers
    reg [M*DATA_W-1:0]   x_tile_buf;
    reg [M*DATA_W-1:0]   res_tile_buf;
    reg [M*DATA_W-1:0]   gamma_tile_buf;
    reg [M*DATA_W-1:0]   beta_tile_buf;
    reg [M*DATA_W-1:0]   z_tile_buf;
    reg [M*DATA_W-1:0]   y_tile_buf;

    // Pipeline registers
    reg signed [2*DATA_W-1:0] s_elem_reg;      // s[k] from DSP1 (Q16.16)
    reg signed [DATA_W-1:0]   tanh_val_reg;    // tanh(s[k]) from LUT
    reg signed [DATA_W-1:0]   gamma_elem_reg;  // γ[k] for pipeline alignment
    reg signed [DATA_W-1:0]   beta_elem_reg;   // β[k] for pipeline alignment

    // -------------------------------------------------------------------
    // AXI response ready
    // -------------------------------------------------------------------
    assign axi_resp_ready = (state == S_WAIT_X) ||
                            (state == S_WAIT_RES) ||
                            (state == S_WAIT_GAM) ||
                            (state == S_WAIT_BET);

    // -------------------------------------------------------------------
    // Parallel addition: z[k] = x[k] + residual[k]
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
    // DSP1: s[k] = α * z[k]  (Q8.8 × Q8.8 → Q16.16)
    // Must be declared BEFORE tanh_lut instance and s_for_lut assign,
    // otherwise the assign creates an implicit 1-bit wire for s_product,
    // causing Vivado "Identifier previously declared" warning.
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]   z_elem_wire = z_tile_buf[elem_idx*DATA_W +: DATA_W];
    wire signed [2*DATA_W-1:0] s_product   = alpha * z_elem_wire;

    // -------------------------------------------------------------------
    // Tanh LUT instance
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0]  tanh_lut_out;
    wire                      tanh_lut_valid;

    // CRITICAL: Use combinational s_product, not registered s_elem_reg.
    // s_elem_reg is set with non-blocking assignment in S_DSP_SCALE,
    // so the tanh LUT would read the OLD value if we used s_elem_reg.
    // Using s_product gives the current element's value immediately.
    wire signed [2*DATA_W-1:0] s_for_lut;
    assign s_for_lut = s_product;

    tanh_lut #(
        .DATA_W        (DATA_W),
        .FRAC_W        (FRAC_W),
        .S_WIDTH       (2*DATA_W),
        .S_FRAC        (2*FRAC_W),
        .SAT_LIMIT_INT (4)
    ) u_tanh_lut (
        .clk        (clk),
        .rst_n      (rst_n),
        .rd_en      (state == S_DSP_SCALE),   // Issue read when computing s[k]
        .s_val      (s_for_lut),
        .tanh_out   (tanh_lut_out),
        .tanh_valid (tanh_lut_valid)
    );

    // -------------------------------------------------------------------
    // DSP2: y[k] = γ[k] * tanh_val[k] + β[k]
    // Compute as: y_raw = γ * tanh_val (Q8.8 × Q8.8 → Q16.16)
    // Then: y = y_raw >> 8 + β (saturate to Q8.8)
    // -------------------------------------------------------------------
    wire signed [2*DATA_W-1:0] y_prod = gamma_elem_reg * tanh_val_reg;
    // Shift product from Q16.16 to Q8.8 and add β
    // y = γ*tanh_val + β, where γ*tanh_val is Q16.16 and β needs <<FRAC_W
    // Use proper signed arithmetic — concatenation is unsigned in Verilog!
    wire signed [2*DATA_W-1:0] beta_shifted;
    assign beta_shifted = beta_elem_reg << FRAC_W;  // β in Q16.16 (sign-extended by <<)

    wire signed [2*DATA_W:0]   y_sum_ext;
    assign y_sum_ext = y_prod + beta_shifted;

    // Saturating extraction to DATA_W bits (Q8.8)
    // Saturation thresholds in wide format to avoid overflow
    // MAX_VAL<<FRAC_W = 32767<<8 = 8388352, which doesn't fit in 16 bits
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
            x_tile_buf      <= '0;
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
            // Defaults
            axi_req_valid <= 1'b0;
            out_wr_en     <= 1'b0;
            res_wr_en     <= 1'b0;
            stream_valid  <= 1'b0;

            case (state)
                // =====================================================
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        tile_idx  <= '0;
                        elem_idx  <= '0;
                        axi_req_addr  <= INPUT_BASE + seq_idx * VECTOR_BYTES;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;
                        state <= S_REQ_X;
                    end
                end

                // =====================================================
                S_REQ_X: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt <= '0;
                        state <= S_WAIT_X;
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
                            axi_req_addr  <= RESIDUAL_BASE + seq_idx * VECTOR_BYTES +
                                              tile_idx * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state <= S_REQ_RES;
                        end
                    end
                end

                // =====================================================
                S_REQ_RES: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt <= '0;
                        state <= S_WAIT_RES;
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
                            state <= S_REQ_GAM;
                        end
                    end
                end

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
                            axi_req_addr  <= BETA_BASE + tile_idx * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state <= S_REQ_BET;
                        end
                    end
                end

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
                // COMPUTE: z = x + residual (all M elements, parallel)
                // =====================================================
                S_ADD: begin
                    z_tile_buf <= z_comb;
                    elem_idx   <= '0;
                    y_tile_buf <= '0;
                    state      <= S_DSP_SCALE;
                end

                // =====================================================
                // COMPUTE: s[k] = α * z[k]  (DSP1 multiply)
                // Also: tanh LUT read issued for s[k] (rd_en=1)
                // =====================================================
                S_DSP_SCALE: begin
                    s_elem_reg    <= s_product;
                    gamma_elem_reg <= gamma_tile_buf[elem_idx*DATA_W +: DATA_W];
                    beta_elem_reg  <= beta_tile_buf[elem_idx*DATA_W +: DATA_W];
                    state <= S_DSP_TANH;
                end

                // =====================================================
                // Wait for tanh LUT output (1-cycle BRAM latency)
                // =====================================================
                S_DSP_TANH: begin
                    tanh_val_reg <= tanh_lut_out;
                    state <= S_DSP_AFFINE;
                end

                // =====================================================
                // COMPUTE: y[k] = γ[k] * tanh_val[k] + β[k]  (DSP2 MAC)
                // =====================================================
                S_DSP_AFFINE: begin
                    y_tile_buf[elem_idx*DATA_W +: DATA_W] <= y_elem_sat;
                    state <= S_ELEM_DONE;
                end

                // =====================================================
                // Element done — advance or finish tile
                // =====================================================
                S_ELEM_DONE: begin
                    if (elem_idx == M - 1) begin
                        // All elements in this tile are done
                        state <= S_WRITE;
                    end else begin
                        elem_idx <= elem_idx + 1;
                        state <= S_DSP_SCALE;
                    end
                end

                // =====================================================
                // WRITE: y_tile → output BRAM, z_tile → residual BRAM
                // =====================================================
                S_WRITE: begin
                    out_wr_en   <= 1'b1;
                    out_wr_addr <= tile_idx;
                    out_wr_data <= y_tile_buf;

                    res_wr_en   <= 1'b1;
                    res_wr_addr <= tile_idx;
                    res_wr_data <= z_tile_buf;

                    // Streaming output for post-DyT mode
                    stream_data  <= y_tile_buf;
                    stream_addr  <= tile_idx;
                    stream_valid <= 1'b1;
                    stream_last  <= (tile_idx == NUM_TILES - 1);

                    state <= S_NEXT;
                end

                // =====================================================
                // Next tile or done
                // =====================================================
                S_NEXT: begin
                    if (tile_idx == NUM_TILES - 1) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        tile_idx <= tile_idx + 1;
                        elem_idx <= '0;
                        axi_req_addr  <= INPUT_BASE + seq_idx * VECTOR_BYTES +
                                          (tile_idx + 1) * TILE_BYTES;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;
                        state <= S_REQ_X;
                    end
                end

                // =====================================================
                S_DONE: begin
                    done <= 1'b1;
                    if (start) begin
                        // Restart the next sequence element without requiring
                        // an external reset.  x/residual are row-major in
                        // external memory; gamma/beta remain shared.
                        done          <= 1'b0;
                        tile_idx      <= '0;
                        elem_idx      <= '0;
                        axi_req_addr  <= INPUT_BASE + seq_idx * VECTOR_BYTES;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;
                        state         <= S_REQ_X;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
