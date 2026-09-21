//============================================================================
// tb_ffn_block_zynq.v — Self-Checking Testbench for Complete 3-Phase Pipeline
//============================================================================
// Tests ffn_block_zynq (Pre-DyT → FFN → Post-DyT) with small parameters.
//
// Parameters: D=16, M=4, NUM_COLS=4, AXI_DATA_W=64
//   NUM_TILES_D  = 4,  NUM_TILES_4D = 16
//   Weight tile: 4×4 = 16 elements, 4 AXI beats per weight tile
//   Input tile: 4 elements, 1 AXI beat per input tile
//
// Reference model (all Q8.8 integer arithmetic matching HW):
//   1. z₁[k] = x[k] + res₁[k]               (Q8.8 add, stored in residual_bram)
//   2. y_in[k] = γ₁·tanh(α₁·z₁[k]) + β₁[k]  (DyT, LUT-matching, Q8.8 output)
//   3. inter[j] = Σ_k y_in_Q8[k] · W_up[k][j]   (Q8.8 × DATA_W signed → 32-bit products)
//   4. relu[j] = max(0, inter[j])             (ReLU on wide accumulator)
//   5. ffn_out[k] = Σ_j relu[j] · W_down[j][k] (wide accum, saturate to Q8.8)
//   6. z₂[k] = ffn_out_Q8[k] + z₁[k]       (Q8.8 add for post-DyT)
//   7. y_out[k] = γ₂·tanh(α₂·z₂[k]) + β₂[k] (DyT, LUT-matching)
//
// AXI address regions:
//   0x00000000: x          0x08000000: residual
//   0x0C000000: γ₁         0x0D000000: β₁
//   0x10000000: W_up       0x20000000: W_down
//   0x30000000: γ₂         0x31000000: β₂
//============================================================================

`timescale 1ns / 1ps

module tb_ffn_block_zynq;

    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    localparam D          = 16;
    localparam M          = 4;
    localparam HIDDEN_DIM  = 4 * D;
    localparam N          = 1;
    localparam NUM_COLS   = 4;
    localparam DATA_W     = 16;
    localparam FRAC_W     = 8;
    localparam AXI_DATA_W = 64;
    localparam AXI_ADDR_W = 32;

    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = HIDDEN_DIM / M;
    localparam ELEMS_PER_BEAT = AXI_DATA_W / DATA_W;
    localparam BEATS_PER_TILE = M / ELEMS_PER_BEAT;
    localparam WEIGHT_BEATS   = (M * M) / ELEMS_PER_BEAT;
    localparam TILE_BYTES     = M * (DATA_W / 8);
    localparam WTILE_BYTES    = M * M * (DATA_W / 8);
    // 500 MHz clock: 2 ns period (timescale is 1 ns / 1 ps).
    localparam CLK_PERIOD     = 2;

    // -------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------
    reg                          clk, rst_n, start;
    wire                         done;
    wire signed [DATA_W-1:0]     alpha_pre  = 128;   // α₁ = 0.5 in Q8.8
    wire signed [DATA_W-1:0]     alpha_post = 128;   // α₂ = 0.5 in Q8.8

    // AXI4 Read Master interface
    wire  [AXI_ADDR_W-1:0]      araddr;
    wire  [7:0]                 arlen;
    wire  [2:0]                 arsize;
    wire  [1:0]                 arburst;
    wire                         arvalid;
    reg                          arready;
    reg   [AXI_DATA_W-1:0]      rdata;
    reg                          rlast, rvalid;
    wire                         rready;

    // Streamed output
    wire  [AXI_DATA_W-1:0]      out_data;
    wire  [$clog2(D/M)-1:0]     out_addr;
    wire  [2:0]                  out_offset;
    wire                         out_valid;
    wire                         out_last;
    wire [`CLOG2_MIN1(N)-1:0]   out_seq_idx;

    // -------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------
    ffn_block_zynq #(
        .D(D), .M(M), .N(N), .HIDDEN_DIM(HIDDEN_DIM), .NUM_COLS(NUM_COLS), .DATA_W(DATA_W), .FRAC_W(FRAC_W),
        .AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W)
    ) u_dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .alpha_pre(alpha_pre), .alpha_post(alpha_post),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rlast(rlast), .rvalid(rvalid), .rready(rready),
        .out_data(out_data), .out_addr(out_addr), .out_offset(out_offset),
        .out_valid(out_valid), .out_last(out_last), .out_seq_idx(out_seq_idx)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------------------------------------------------------
    // Transaction latency measurement
    // -------------------------------------------------------------------
    // The block has no explicit input/output ID ports and allows one block
    // transaction at a time, so the TB assigns a monotonically increasing ID.
    // Completion is measured on out_valid && out_last, i.e. the final
    // serialized output beat rather than the first output beat.
    //
    // A fixed realtime table is used instead of realtime start_times[int]
    // because Icarus Verilog does not support realtime associative arrays.
    // -------------------------------------------------------------------
    localparam integer MAX_TXN_IDS = 16;
    realtime start_times [0:MAX_TXN_IDS-1];
    reg      start_valid [0:MAX_TXN_IDS-1];
    integer  next_txn_id, active_txn_id;
    integer  in_id, out_id;
    realtime measured_latency_ns;
    integer  measured_latency_cycles;

    wire in_valid = start;
    wire in_ready = (u_dut.phase_state == 2'd0);

    initial begin
        next_txn_id = 0;
        active_txn_id = -1;
        measured_latency_ns = 0.0;
        measured_latency_cycles = 0;
        for (integer ti = 0; ti < MAX_TXN_IDS; ti = ti + 1)
            start_valid[ti] = 1'b0;
    end

    always @* begin
        in_id  = next_txn_id;
        out_id = active_txn_id;
    end

    // Capture start time with a unique transaction ID.
    always @(posedge clk) begin
        if (in_valid && in_ready) begin
            if (in_id < MAX_TXN_IDS) begin
                start_times[in_id] = $realtime;
                start_valid[in_id] = 1'b1;
                active_txn_id = in_id;
                next_txn_id = next_txn_id + 1;
                $display("[LAT] START ID=%0d | time=%0.3f ns", in_id, $realtime);
            end
        end
    end

    // Calculate latency when the complete output transaction finishes.
    always @(posedge clk) begin
        if (out_valid && out_last && out_id >= 0 &&
            out_id < MAX_TXN_IDS && start_valid[out_id]) begin
            measured_latency_ns = $realtime - start_times[out_id];
            measured_latency_cycles =
                $rtoi((measured_latency_ns / CLK_PERIOD) + 0.5);
            $display("[LAT] ID: %0d | Latency: %0.3f ns | %0d cycles @ 500 MHz",
                     out_id, measured_latency_ns, measured_latency_cycles);
            start_valid[out_id] = 1'b0;
        end
    end

    // -------------------------------------------------------------------
    // Test data arrays — all in Q8.8 format (signed 16-bit)
    // -------------------------------------------------------------------
    reg signed [DATA_W-1:0] x_mem      [0:D-1];
    reg signed [DATA_W-1:0] res_mem    [0:D-1];
    reg signed [DATA_W-1:0] gamma1_mem [0:D-1];
    reg signed [DATA_W-1:0] beta1_mem  [0:D-1];
    reg signed [DATA_W-1:0] W_up      [0:D-1][0:HIDDEN_DIM-1];
    reg signed [DATA_W-1:0] W_down    [0:HIDDEN_DIM-1][0:D-1];
    reg signed [DATA_W-1:0] gamma2_mem [0:D-1];
    reg signed [DATA_W-1:0] beta2_mem  [0:D-1];

    // -------------------------------------------------------------------
    // AXI4 Slave Memory Model (proper handshake protocol)
    // -------------------------------------------------------------------
    // The slave follows proper AXI4 protocol:
    //   - After AR handshake, prepares beat data and asserts rvalid
    //   - Holds rvalid until master accepts (rready && rvalid)
    //   - After acceptance, advances to next beat or completes burst
    // This ensures no beats are lost regardless of master timing.
    // -------------------------------------------------------------------
    reg                          axi_busy;
    reg  [AXI_ADDR_W-1:0]       burst_start_addr;
    reg  [7:0]                  burst_len, beat_cnt;
    reg  [AXI_ADDR_W-1:0]       current_addr;
    reg                          beat_pending;  // Data prepared, waiting for master acceptance

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            arready       <= 1'b1;
            rvalid        <= 1'b0;
            rdata         <= '0;
            rlast         <= 1'b0;
            axi_busy      <= 1'b0;
            burst_start_addr <= '0;
            burst_len     <= '0;
            beat_cnt      <= '0;
            current_addr  <= '0;
            beat_pending  <= 1'b0;
        end else begin
            if (!axi_busy) begin
                // IDLE: accept AR requests
                arready <= 1'b1;
                if (arvalid && arready) begin
                    arready         <= 1'b0;
                    burst_start_addr <= araddr;
                    burst_len       <= arlen + 8'd1;
                    beat_cnt        <= 8'd0;
                    current_addr    <= araddr;
                    axi_busy        <= 1'b1;
                    beat_pending    <= 1'b1;  // Prepare first beat immediately
                    rdata           <= build_axi_beat(araddr, 8'd0);
                    rvalid          <= 1'b1;
                    rlast           <= (arlen == 8'd0);  // Single-beat burst
                end
            end else begin
                // BURST: delivering R beats
                if (beat_pending && rvalid && rready) begin
                    // Master accepted current beat — advance to next
                    beat_pending <= 1'b0;
                    rvalid       <= 1'b0;
                    rlast        <= 1'b0;
                    beat_cnt     <= beat_cnt + 8'd1;
                    current_addr <= current_addr + (AXI_DATA_W / 8);

                    if (beat_cnt < burst_len - 8'd1) begin
                        // More beats to deliver — prepare next beat
                        beat_pending <= 1'b1;
                        rdata        <= build_axi_beat(current_addr + (AXI_DATA_W / 8),
                                                      beat_cnt + 8'd1);
                        rvalid       <= 1'b1;
                        rlast        <= (beat_cnt + 8'd1 == burst_len - 8'd1);
                    end else begin
                        // Burst complete — return to idle
                        axi_busy <= 1'b0;
                        arready  <= 1'b1;
                    end
                end
                // If rvalid && !rready: slave holds beat (rvalid stays asserted)
                // If !rvalid: slave is between beats (brief gap)
            end
        end
    end

    // -------------------------------------------------------------------
    // AXI beat builder — maps addresses to test data
    // -------------------------------------------------------------------
    function [AXI_DATA_W-1:0] build_axi_beat;
        input [AXI_ADDR_W-1:0] addr;
        input [7:0] beat;
        integer i, region, offset_bytes, elem_idx;
        integer tile_idx_w, inner_idx_w, col_idx_w, elem_in_tile_w, k_in_tile, j_in_tile;
        reg signed [DATA_W-1:0] val;
        reg [AXI_ADDR_W-1:0] region_base;
        begin
            build_axi_beat = '0;

            // Determine address region
            if (addr < 32'h0800_0000) begin region = 0; region_base = 32'h0000_0000; end
            else if (addr < 32'h0C00_0000) begin region = 1; region_base = 32'h0800_0000; end
            else if (addr < 32'h0D00_0000) begin region = 2; region_base = 32'h0C00_0000; end
            else if (addr < 32'h1000_0000) begin region = 3; region_base = 32'h0D00_0000; end
            else if (addr < 32'h2000_0000) begin region = 4; region_base = 32'h1000_0000; end
            else if (addr < 32'h3000_0000) begin region = 5; region_base = 32'h2000_0000; end
            else if (addr < 32'h3100_0000) begin region = 6; region_base = 32'h3000_0000; end
            else begin region = 7; region_base = 32'h3100_0000; end

            offset_bytes = addr - region_base;

            for (i = 0; i < ELEMS_PER_BEAT; i = i + 1) begin
                case (region)
                    0, 1, 2, 3, 6, 7: begin
                        // Vector data: x, res, gamma, beta
                        // offset_bytes already accounts for beat position
                        // (current_addr is incremented per beat), so we
                        // only need to add the intra-beat element offset i.
                        elem_idx = (offset_bytes / (DATA_W / 8)) + i;
                        case (region)
                            0: val = x_mem[elem_idx];
                            1: val = res_mem[elem_idx];
                            2: val = gamma1_mem[elem_idx];
                            3: val = beta1_mem[elem_idx];
                            6: val = gamma2_mem[elem_idx];
                            7: val = beta2_mem[elem_idx];
                            default: val = 0;
                        endcase
                    end
                    4: begin
                        // W_up weights: tile-based layout
                        // offset_bytes already accounts for beat position
                        // (current_addr is incremented per beat), so we
                        // only need the intra-beat element offset i.
                        tile_idx_w = offset_bytes / WTILE_BYTES;
                        inner_idx_w = tile_idx_w / NUM_TILES_4D;
                        col_idx_w = tile_idx_w % NUM_TILES_4D;
                        elem_in_tile_w = (offset_bytes % WTILE_BYTES) / (DATA_W / 8) + i;
                        k_in_tile = elem_in_tile_w / M;
                        j_in_tile = elem_in_tile_w % M;
                        val = W_up[inner_idx_w * M + k_in_tile][col_idx_w * M + j_in_tile];
                    end
                    5: begin
                        // W_down weights: tile-based layout
                        // offset_bytes already accounts for beat position
                        tile_idx_w = offset_bytes / WTILE_BYTES;
                        inner_idx_w = tile_idx_w / NUM_TILES_D;
                        col_idx_w = tile_idx_w % NUM_TILES_D;
                        elem_in_tile_w = (offset_bytes % WTILE_BYTES) / (DATA_W / 8) + i;
                        k_in_tile = elem_in_tile_w / M;
                        j_in_tile = elem_in_tile_w % M;
                        val = W_down[inner_idx_w * M + k_in_tile][col_idx_w * M + j_in_tile];
                    end
                    default: val = 0;
                endcase
                build_axi_beat[i*DATA_W +: DATA_W] = val;
            end
        end
    endfunction

    // -------------------------------------------------------------------
    // Helper functions: Q8.8 ↔ float, LUT-matching tanh
    // -------------------------------------------------------------------
    function [DATA_W-1:0] float_to_q8;
        input real f;
        integer q;
        begin
            q = $rtoi(f * 256.0 + 0.5);
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            float_to_q8 = q[DATA_W-1:0];
        end
    endfunction

    function real q8_to_float;
        input [DATA_W-1:0] q;
        integer iq;
        begin
            iq = q;
            if (iq >= 32768) iq = iq - 65536;
            q8_to_float = iq / 256.0;
        end
    endfunction

    // LUT-matching tanh reference (mirrors HW tanh_lut.v)
    function real lut_tanh_ref;
        input real x;
        real ax, r, e2x;
        integer lut_idx;
        begin
            ax = x;
            if (ax < 0.0) ax = -ax;
            if (ax >= 4.0) begin
                r = 1.0;
            end else begin
                lut_idx = $rtoi(ax * 256.0);
                if (lut_idx >= 1024) lut_idx = 1023;
                e2x = $exp(2.0 * (lut_idx / 256.0));
                r = (e2x - 1.0) / (e2x + 1.0);
            end
            if (x < 0.0) r = -r;
            lut_tanh_ref = r;
        end
    endfunction

    // -------------------------------------------------------------------
    // Output capture
    // -------------------------------------------------------------------
    reg signed [DATA_W-1:0] y_hw [0:D-1];
    integer capture_cnt;

    always @(posedge clk) begin
        if (out_valid) begin
            for (integer ci = 0; ci < ELEMS_PER_BEAT; ci = ci + 1) begin
                y_hw[out_addr * M + ci] <= out_data[ci*DATA_W +: DATA_W];
            end
            capture_cnt = capture_cnt + 1;
        end
    end

    // -------------------------------------------------------------------
    // Reference model — Q8.8 integer arithmetic matching HW
    // -------------------------------------------------------------------
    reg signed [DATA_W-1:0] z1_q8     [0:D-1];
    reg signed [DATA_W-1:0] y_in_q8   [0:D-1];
    reg signed [DATA_W-1:0] ffn_q8    [0:D-1];
    reg signed [DATA_W-1:0] z2_q8     [0:D-1];
    reg signed [DATA_W-1:0] y_out_q8  [0:D-1];

    // Wide accumulator for FFN matrix multiply
    reg signed [63:0] inter_wide [0:HIDDEN_DIM-1];
    reg signed [63:0] relu_wide  [0:HIDDEN_DIM-1];
    reg signed [63:0] ffn_wide   [0:D-1];
    // Q8.8 relu values (matching HW relu_bram storage)
    reg signed [DATA_W-1:0] relu_q8 [0:HIDDEN_DIM-1];

    real alpha1_real = 0.5;
    real alpha2_real = 0.5;

    task compute_reference;
        integer i, j, k;
        real z1_f, s1_f, tanh1_f, y_in_f;
        real ffn_f, z2_f, s2_f, tanh2_f, y_out_f;
        reg signed [63:0] val;
        reg signed [DATA_W-1:0] sat_val;
        begin
            $display("====== Computing Reference (Q8.8 integer arithmetic) ======");

            // ---- Phase 0: Pre-Add-DyT ----
            for (i = 0; i < D; i = i + 1) begin
                // z₁ = x + residual (Q8.8 integer addition)
                z1_q8[i] = x_mem[i] + res_mem[i];

                // Compute DyT in floating-point (LUT-matching)
                z1_f  = q8_to_float(z1_q8[i]);
                s1_f  = alpha1_real * z1_f;
                tanh1_f = lut_tanh_ref(s1_f);
                y_in_f  = q8_to_float(gamma1_mem[i]) * tanh1_f
                         + q8_to_float(beta1_mem[i]);
                y_in_q8[i] = float_to_q8(y_in_f);
            end

            // ---- Phase 1: FFN ----
            // Matrix multiply: inter[j] = Σ_k y_in[k] * W_up[k][j]
            // Both y_in and W_up are signed DATA_W (Q8.8 × Q8.8 = Q16.16 products)
            // Accumulate in 64-bit to avoid overflow
            for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                inter_wide[j] = 0;
                for (i = 0; i < D; i = i + 1) begin
                    inter_wide[j] = inter_wide[j] + y_in_q8[i] * W_up[i][j];
                end
            end

            // Convert inter_wide (Q16.16) to Q8.8 with saturation, then apply ReLU
            // This matches the HW: tm_proj_stage extracts Q8.8 from Q16.16 accumulator,
            // then relu_stage applies ReLU on the Q8.8 values stored in relu_bram.
            // CRITICAL: Must use $rtoi($itor()) for correct signed right shift,
            // because Verilog >> is logical (unsigned) and >>> may not work
            // correctly in all simulators for 64-bit signed values.
            for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                // Extract Q8.8 from Q16.16 using real conversion (avoids shift issues)
                val = $rtoi($itor(inter_wide[j]) / 256.0);
                if (val > 32767)
                    sat_val = 32767;
                else if (val < -32768)
                    sat_val = -32768;
                else
                    sat_val = val[DATA_W-1:0];
                // ReLU on Q8.8 values (matching HW relu_stage: max(0, x))
                relu_q8[j] = (sat_val > 0) ? sat_val : 16'sd0;
                // Also store in Q16.16 format for debug display
                relu_wide[j] = relu_q8[j] * 256;  // Q8.8 → Q16.16
            end

            // Down projection: ffn[k] = Σ_j relu_q8[j] * W_down[j][k]
            // CRITICAL: Use Q8.8 relu values to match HW.
            // HW reads Q8.8 from relu_bram, multiplies Q8.8 × Q8.8 = Q16.16 products.
            // The HW accumulator is in Q16.16 format, extraction >>8 gives Q8.8.
            for (k = 0; k < D; k = k + 1) begin
                ffn_wide[k] = 0;
                for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                    // Q8.8 relu × Q8.8 weight = Q16.16 product (same as HW mul_col)
                    ffn_wide[k] = ffn_wide[k] + relu_q8[j] * W_down[j][k];
                end
                // Saturate to Q8.8: the HW extracts bits [FRAC_W+DATA_W-1:FRAC_W]
                // from the ACC_W-wide accumulator and saturates to ±MAX_OUT
                // For 64-bit ref, use $rtoi/$itor for correct signed extraction
                val = $rtoi($itor(ffn_wide[k]) / 256.0);
                if (val > 32767)
                    sat_val = 32767;
                else if (val < -32768)
                    sat_val = -32768;
                else
                    sat_val = val[DATA_W-1:0];
                ffn_q8[k] = sat_val;
            end

            // ---- Phase 2: Post-Add-DyT ----
            for (i = 0; i < D; i = i + 1) begin
                // z₂ = ffn_out + z₁ (Q8.8 integer addition)
                z2_q8[i] = ffn_q8[i] + z1_q8[i];

                // Compute DyT in floating-point (LUT-matching)
                z2_f  = q8_to_float(z2_q8[i]);
                s2_f  = alpha2_real * z2_f;
                tanh2_f = lut_tanh_ref(s2_f);
                y_out_f  = q8_to_float(gamma2_mem[i]) * tanh2_f
                           + q8_to_float(beta2_mem[i]);
                y_out_q8[i] = float_to_q8(y_out_f);
            end

            $display("  Reference computed (D=%0d, 4D=%0d)", D, HIDDEN_DIM);
            $display("  Ref y_in_q8[0..3] = {%0d, %0d, %0d, %0d}",
                     y_in_q8[0], y_in_q8[1], y_in_q8[2], y_in_q8[3]);
            $display("  Ref ffn_q8[0..3] = {%0d, %0d, %0d, %0d}",
                     ffn_q8[0], ffn_q8[1], ffn_q8[2], ffn_q8[3]);
            $display("  Ref z1_q8[0..3] = {%0d, %0d, %0d, %0d}",
                     z1_q8[0], z1_q8[1], z1_q8[2], z1_q8[3]);
        end
    endtask

    // -------------------------------------------------------------------
    // Debug monitors — targeted data path probes
    // -------------------------------------------------------------------
    integer wr_cnt, out_cnt, norm_wr_cnt;
    integer probe_cycle;
    reg [1:0] prev_phase_state;

    initial begin
        wr_cnt = 0; out_cnt = 0; norm_wr_cnt = 0; probe_cycle = 0;
        prev_phase_state = 0;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            probe_cycle = probe_cycle + 1;
            
            // Phase state changes (reduced verbosity)
            if (u_dut.phase_state !== prev_phase_state) begin
                $display("[DBG] phase=%0d sel=%0d time=%0t",
                         u_dut.phase_state, u_dut.phase_sel, $time);
                prev_phase_state <= u_dut.phase_state;
            end
            
            // Pre-DyT BRAM writes — summary only
            if (u_dut.norm_wr_en) begin
                norm_wr_cnt = norm_wr_cnt + 1;
            end
            
            // FFN output BRAM writes — summary only
            if (u_dut.ffn_out_wr_en) begin
                wr_cnt = wr_cnt + 1;
            end
            
            // Post-DyT output capture (already captured in y_hw array)
            if (out_valid) begin
                out_cnt = out_cnt + 1;
            end
        end
    end

    // -------------------------------------------------------------------
    // AXI monitor
    // -------------------------------------------------------------------
    always @(posedge clk)
        if (arvalid && arready)
            $display("[AXI] AR: addr=0x%08h len=%0d", araddr, arlen+1);

    // -------------------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------------------
    integer i, j, errors;
    real y_hw_f, y_exp_f, diff;
    real tol;

    initial begin
        $dumpfile("tb_ffn_block_zynq.vcd");
        $dumpvars(0, tb_ffn_block_zynq);

        // Initialize output capture
        for (i = 0; i < D; i = i + 1) y_hw[i] = 0;
        capture_cnt = 0;

        // ---- Initialize test data in Q8.8 ----
        for (i = 0; i < D; i = i + 1) begin
            x_mem[i]      = float_to_q8(i * 0.25 - 2.0);
            res_mem[i]    = float_to_q8(i * 0.125 - 1.0);
            gamma1_mem[i] = float_to_q8(1.0);
            beta1_mem[i]  = float_to_q8(0.0);
            gamma2_mem[i] = float_to_q8(1.0);
            beta2_mem[i]  = float_to_q8(0.0);
        end

        // Weight matrices: Q8.8 values — small real values to avoid overflow
        // Use 0.5 and -0.5 range for manageable accumulation
        for (i = 0; i < D; i = i + 1)
            for (j = 0; j < HIDDEN_DIM; j = j + 1)
                W_up[i][j] = float_to_q8((((i*3 + j*7 + 1) % 11) - 5) * 0.1);

        for (i = 0; i < HIDDEN_DIM; i = i + 1)
            for (j = 0; j < D; j = j + 1)
                W_down[i][j] = float_to_q8((((i*5 + j*11 + 3) % 9) - 4) * 0.1);

        // ---- Compute software reference ----
        compute_reference;

        // ---- Reset & start ----
        $display("");
        $display("====== HW Test (D=%0d, M=%0d, AXI=%0d) ======", D, M, AXI_DATA_W);

        rst_n <= 1'b0;
        start <= 1'b0;
        #(CLK_PERIOD * 10);
        rst_n <= 1'b1;
        #(CLK_PERIOD * 5);

        // Assert start for 2 cycles (non-blocking)
        start <= 1'b1;
        @(posedge clk);
        @(posedge clk);
        start <= 1'b0;

        $display("[TB] Start at time %0t", $time);

        // ---- Wait for completion ----
        i = 0;
        while (!done && i < 200000) begin
            @(posedge clk);
            i = i + 1;
        end

        if (!done) begin
            $display("ERROR: DUT did not complete after 200000 cycles!");
            $finish;
        end

        $display("[TB] Done at time %0t (cycles: %0d)", $time, i);

        // Wait for output streaming
        #(CLK_PERIOD * 100);

        // ---- Compare output ----
        $display("");
        $display("========================================");
        $display("  Post-DyT Output Comparison (Q8.8)");
        $display("========================================");

        errors = 0;
        // Tolerance: LUT quantization (±1/256) + DSP rounding (±2/256) + FFN accumulation
        tol = 6.0 / 256.0;

        for (i = 0; i < D; i = i + 1) begin
            y_hw_f  = q8_to_float(y_hw[i]);
            y_exp_f = q8_to_float(y_out_q8[i]);
            diff = y_hw_f - y_exp_f;
            if (diff < 0) diff = -diff;

            if (diff <= tol) begin
                $display("  PASS y_out[%0d]: HW=%0.4f EXP=%0.4f diff=%0.4f",
                         i, y_hw_f, y_exp_f, diff);
            end else begin
                $display("  FAIL y_out[%0d]: HW=%0d EXP=%0d diff=%0.4f (z2=%0.4f)",
                         i, $signed(y_hw[i]), $signed(y_out_q8[i]), diff, q8_to_float(z2_q8[i]));
                errors = errors + 1;
            end
        end

        $display("");
        if (errors == 0)
            $display("*** ALL %0d OUTPUT TESTS PASS ***", D);
        else
            $display("*** %0d / %0d FAIL ***", errors, D);

        $display("====== TEST COMPLETE ======");
        $finish;
    end

    // Timeout
    initial begin #(CLK_PERIOD * 500000); $display("TIMEOUT!"); $finish; end

endmodule
