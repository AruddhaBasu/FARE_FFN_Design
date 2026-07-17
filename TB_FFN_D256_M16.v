//============================================================================
// tb_ffn_complete.v — Comprehensive Self-Checking Testbench for Pipelined FFN
//============================================================================
// This testbench:
//   1. Defines complete input vector (1×D), up-projection matrix (D×4D),
//      and down-projection matrix (4D×D) with deterministic signed values
//   2. Populates a compact, address-mapped memory model in the exact tiled
//      layout the fetch stage expects
//   3. Computes a software reference: y = saturate(ReLU(x · W_up) · W_down)
//   4. Runs the hardware pipeline end-to-end
//   5. Captures ReLU intermediate values and final output tiles
//   6. Verifies output_tile_data / output_tile_addr / output_tile_valid
//      port signals are correct and properly latency-aligned
//   7. Compares every element against the software reference
//   8. Reports PASS / FAIL with per-element diagnostics
//============================================================================

`timescale 1ns / 1ps

module tb_ffn_complete;

    // -------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------
    parameter D          = 8;
    parameter M          = 4;
    parameter DATA_W     = 16;
    parameter AXI_DATA_W = 64;
    parameter AXI_ADDR_W = 32;

    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;
    localparam WORDS_PER_BEAT = AXI_DATA_W / DATA_W;
    localparam ELEM_BYTES    = DATA_W / 8;
    localparam CLK_PERIOD    = 10;

    localparam INPUT_BASE  = 32'h0000_0000;
    localparam WUP_BASE    = 32'h1000_0000;
    localparam WDOWN_BASE  = 32'h2000_0000;

    localparam INPUT_WORDS  = D;
    localparam WUP_WORDS    = D * 4 * D;
    localparam WDOWN_WORDS  = 4 * D * D;
    localparam TOTAL_WORDS  = INPUT_WORDS + WUP_WORDS + WDOWN_WORDS;

    // -------------------------------------------------------------------
    // Signals
    // -------------------------------------------------------------------
    reg                          clk, rst_n, start;
    wire                         done;
    wire  [AXI_ADDR_W-1:0]      araddr;
    wire  [7:0]                 arlen;
    wire  [2:0]                 arsize;
    wire  [1:0]                 arburst;
    wire                         arvalid;
    reg                          arready;
    reg   [AXI_DATA_W-1:0]      rdata;
    reg                          rlast, rvalid;
    wire                         rready;
    wire  [M*DATA_W-1:0]        output_tile_data;
    wire  [$clog2(D/M)-1:0]     output_tile_addr;
    wire                         output_tile_valid;

    // -------------------------------------------------------------------
    // Complete Test Matrices
    // -------------------------------------------------------------------
    reg signed [DATA_W-1:0] x       [0:D-1];
    reg signed [DATA_W-1:0] W_up    [0:D-1][0:4*D-1];
    reg signed [DATA_W-1:0] W_down  [0:4*D-1][0:D-1];

    // -------------------------------------------------------------------
    // Software reference
    // -------------------------------------------------------------------
    reg signed [31:0]       inter_ref  [0:4*D-1];
    reg signed [31:0]       activ_ref  [0:4*D-1];
    reg signed [31:0]       y_ref_wide [0:D-1];
    reg signed [DATA_W-1:0] y_ref      [0:D-1];

    // -------------------------------------------------------------------
    // Hardware output capture
    // -------------------------------------------------------------------
    // Method 1: Capture from BRAM write path (internal snoop)
    reg signed [DATA_W-1:0] y_hw    [0:D-1];
    reg signed [DATA_W-1:0] hw_relu [0:4*D-1];
    integer hw_tile_cnt, hw_relu_cnt;

    // Method 2: Capture from top-level output_tile_data port
    reg signed [DATA_W-1:0] y_hw_port [0:D-1];
    integer port_tile_cnt;

    // -------------------------------------------------------------------
    // Compact memory model with address mapping
    // -------------------------------------------------------------------
    reg [DATA_W-1:0] flat_mem [0:TOTAL_WORDS-1];

    function integer addr_to_idx;
        input [AXI_ADDR_W-1:0] byte_addr;
        reg [AXI_ADDR_W-1:0] word_addr;
        begin
            word_addr = byte_addr / ELEM_BYTES;
            if (byte_addr < WUP_BASE)
                addr_to_idx = word_addr - INPUT_BASE / ELEM_BYTES;
            else if (byte_addr < WDOWN_BASE)
                addr_to_idx = INPUT_WORDS + word_addr - WUP_BASE / ELEM_BYTES;
            else
                addr_to_idx = INPUT_WORDS + WUP_WORDS + word_addr - WDOWN_BASE / ELEM_BYTES;
        end
    endfunction

    // ===================================================================
    // DUT
    // ===================================================================
    ffn_top #(
        .D          (D),
        .M          (M),
        .DATA_W     (DATA_W),
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .done           (done),
        .araddr         (araddr),
        .arlen          (arlen),
        .arsize         (arsize),
        .arburst        (arburst),
        .arvalid        (arvalid),
        .arready        (arready),
        .rdata          (rdata),
        .rlast          (rlast),
        .rvalid         (rvalid),
        .rready         (rready),
        .output_tile_data  (output_tile_data),
        .output_tile_addr  (output_tile_addr),
        .output_tile_valid (output_tile_valid)
    );

    // ===================================================================
    // Clock
    // ===================================================================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ===================================================================
    // AXI4 Slave Memory Model
    // ===================================================================
    reg                          axi_busy;
    reg  [AXI_ADDR_W-1:0]       burst_start_addr;
    reg  [7:0]                  burst_len, beat_cnt;
    reg  [AXI_ADDR_W-1:0]       current_addr;
    integer                      mem_idx, flat_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            arready  <= 1'b1;  rvalid <= 1'b0;  rdata <= '0;
            rlast    <= 1'b0;  axi_busy <= 1'b0;
            burst_start_addr <= '0; burst_len <= '0;
            beat_cnt <= '0; current_addr <= '0;
        end else begin
            rvalid <= 1'b0;  rlast <= 1'b0;
            if (!axi_busy) begin
                arready <= 1'b1;
                if (arvalid && arready) begin
                    arready         <= 1'b0;
                    burst_start_addr <= araddr;
                    burst_len       <= arlen + 8'd1;
                    beat_cnt        <= 8'd0;
                    current_addr    <= araddr;
                    axi_busy        <= 1'b1;
                end
            end else begin
                if (rready || !rvalid) begin
                    rdata <= '0;
                    for (mem_idx = 0; mem_idx < WORDS_PER_BEAT; mem_idx = mem_idx + 1) begin
                        flat_idx = addr_to_idx(current_addr + mem_idx * ELEM_BYTES);
                        if (flat_idx >= 0 && flat_idx < TOTAL_WORDS)
                            rdata[mem_idx*DATA_W +: DATA_W] <= flat_mem[flat_idx];
                    end
                    rvalid   <= 1'b1;
                    beat_cnt <= beat_cnt + 8'd1;
                    current_addr <= current_addr + (AXI_DATA_W / 8);
                    if (beat_cnt == burst_len - 1) begin
                        rlast    <= 1'b1;
                        axi_busy <= 1'b0;
                    end
                end
            end
        end
    end

    // ===================================================================
    // Task: init_matrices
    // ===================================================================
    task init_matrices;
        integer i, j;
        begin
            $display("====== Initializing matrices (D=%0d, 4D=%0d) ======", D, 4*D);
            for (i = 0; i < D; i = i + 1)
                x[i] = (i % 5) + 1;
            for (i = 0; i < D; i = i + 1)
                for (j = 0; j < 4*D; j = j + 1)
                    W_up[i][j] = ((i*3 + j*7 + 1) % 11) - 5;
            for (i = 0; i < 4*D; i = i + 1)
                for (j = 0; j < D; j = j + 1)
                    W_down[i][j] = ((i*5 + j*11 + 3) % 9) - 4;
            $display("  x[0:%0d] = %0d %0d %0d %0d %0d %0d %0d %0d",
                     D-1, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        end
    endtask

    // ===================================================================
    // Task: populate_memory
    // ===================================================================
    task populate_memory;
        integer i, j, r, c, k, fi;
        begin
            $display("====== Populating memory (%0d words) ======", TOTAL_WORDS);
            for (i = 0; i < D; i = i + 1) begin
                fi = addr_to_idx(INPUT_BASE + i * ELEM_BYTES);
                flat_mem[fi] = x[i];
            end
            for (r = 0; r < NUM_TILES_D; r = r + 1)
                for (c = 0; c < NUM_TILES_4D; c = c + 1)
                    for (k = 0; k < M; k = k + 1)
                        for (j = 0; j < M; j = j + 1) begin
                            fi = addr_to_idx(WUP_BASE +
                                (r * NUM_TILES_4D + c) * M * M * ELEM_BYTES +
                                (k * M + j) * ELEM_BYTES);
                            flat_mem[fi] = W_up[r*M+k][c*M+j];
                        end
            for (r = 0; r < NUM_TILES_4D; r = r + 1)
                for (c = 0; c < NUM_TILES_D; c = c + 1)
                    for (k = 0; k < M; k = k + 1)
                        for (j = 0; j < M; j = j + 1) begin
                            fi = addr_to_idx(WDOWN_BASE +
                                (r * NUM_TILES_D + c) * M * M * ELEM_BYTES +
                                (k * M + j) * ELEM_BYTES);
                            flat_mem[fi] = W_down[r*M+k][c*M+j];
                        end
            $display("  Verify: flat_mem[0]=%0d (x[0]=%0d)", $signed(flat_mem[0]), x[0]);
            fi = addr_to_idx(WUP_BASE);
            $display("  Verify: W_up[0][0]=%0d (expect %0d)", $signed(flat_mem[fi]), W_up[0][0]);
            fi = addr_to_idx(WDOWN_BASE);
            $display("  Verify: W_down[0][0]=%0d (expect %0d)", $signed(flat_mem[fi]), W_down[0][0]);
        end
    endtask

    // ===================================================================
    // Task: compute_software_reference
    // y = saturate_to_DATA_W( ReLU( x · W_up ) · W_down )
    // ===================================================================
    task compute_software_reference;
        integer i, j;
        reg signed [31:0] val;
        reg signed [DATA_W-1:0] sat;
        begin
            $display("====== Software reference ======");
            for (j = 0; j < 4*D; j = j + 1) begin
                inter_ref[j] = 0;
                for (i = 0; i < D; i = i + 1)
                    inter_ref[j] = inter_ref[j] + x[i] * W_up[i][j];
            end
            for (j = 0; j < 4*D; j = j + 1)
                activ_ref[j] = (inter_ref[j] > 0) ? inter_ref[j] : 32'sd0;
            for (j = 0; j < D; j = j + 1) begin
                y_ref_wide[j] = 0;
                for (i = 0; i < 4*D; i = i + 1)
                    y_ref_wide[j] = y_ref_wide[j] + activ_ref[i] * W_down[i][j];
            end
            for (j = 0; j < D; j = j + 1) begin
                val = y_ref_wide[j];
                if (val > (1 << (DATA_W-1)) - 1)       sat = (1 << (DATA_W-1)) - 1;
                else if (val < -(1 << (DATA_W-1)))      sat = -(1 << (DATA_W-1));
                else                                     sat = val[DATA_W-1:0];
                y_ref[j] = sat;
            end
            $display("  y_ref = [%0d %0d %0d %0d %0d %0d %0d %0d]",
                     $signed(y_ref[0]), $signed(y_ref[1]), $signed(y_ref[2]), $signed(y_ref[3]),
                     $signed(y_ref[4]), $signed(y_ref[5]), $signed(y_ref[6]), $signed(y_ref[7]));
        end
    endtask

    // ===================================================================
    // Capture Method 1: Internal BRAM write snoop
    // ===================================================================
    integer cj;
    always @(negedge clk) begin
        if (u_dut.out_wr_en) begin
            for (cj = 0; cj < M; cj = cj + 1)
                y_hw[u_dut.out_wr_addr * M + cj] = u_dut.out_wr_data[cj*DATA_W +: DATA_W];
            hw_tile_cnt = hw_tile_cnt + 1;
        end
        if (u_dut.relu_wr_en) begin
            for (cj = 0; cj < M; cj = cj + 1)
                hw_relu[u_dut.relu_wr_addr * M + cj] = u_dut.relu_wr_data[cj*DATA_W +: DATA_W];
            hw_relu_cnt = hw_relu_cnt + 1;
        end
    end

    // ===================================================================
    // Capture Method 2: Top-level output_tile_data port
    // ===================================================================
    always @(posedge clk) begin
        if (output_tile_valid) begin
            for (cj = 0; cj < M; cj = cj + 1)
                y_hw_port[output_tile_addr * M + cj] = output_tile_data[cj*DATA_W +: DATA_W];
            port_tile_cnt = port_tile_cnt + 1;
            $display("[PORT] Output tile #%0d at addr=%0d: [%0d %0d %0d %0d]",
                     port_tile_cnt, output_tile_addr,
                     $signed(output_tile_data[0*DATA_W +: DATA_W]),
                     $signed(output_tile_data[1*DATA_W +: DATA_W]),
                     $signed(output_tile_data[2*DATA_W +: DATA_W]),
                     $signed(output_tile_data[3*DATA_W +: DATA_W]));
        end
    end

    // ===================================================================
    // Task: compare_relu
    // ===================================================================
    task compare_relu;
        integer i, errors;
        integer ref_val;
        begin
            errors = 0;
            $display("");
            $display("========================================");
            $display("    ReLU INTERMEDIATE COMPARISON (1×%0d)", 4*D);
            $display("========================================");
            for (i = 0; i < 4*D; i = i + 1) begin
                if (activ_ref[i] > (1 << (DATA_W-1)) - 1)       ref_val = (1 << (DATA_W-1)) - 1;
                else if (activ_ref[i] < -(1 << (DATA_W-1)))      ref_val = -(1 << (DATA_W-1));
                else                                               ref_val = activ_ref[i];
                if (hw_relu[i] !== ref_val[DATA_W-1:0]) begin
                    $display("  FAIL  relu[%0d]: HW=%0d  REF=%0d", i, $signed(hw_relu[i]), $signed(ref_val[DATA_W-1:0]));
                    errors = errors + 1;
                end
            end
            if (errors == 0)  $display("  *** ALL %0d ReLU VALUES MATCH! ***", 4*D);
            else              $display("  *** %0d / %0d ReLU MISMATCH ***", errors, 4*D);
            $display("========================================");
        end
    endtask

    // ===================================================================
    // Task: compare_results — verify both capture methods vs reference
    // ===================================================================
    task compare_results;
        integer i, errors;
        begin
            // --- Method 1: BRAM write snoop ---
            errors = 0;
            $display("");
            $display("========================================");
            $display("  FINAL OUTPUT (BRAM write snoop, 1×%0d)", D);
            $display("========================================");
            for (i = 0; i < D; i = i + 1) begin
                if (y_hw[i] !== y_ref[i]) begin
                    $display("  FAIL  y[%0d]: HW=%0d  REF=%0d", i, $signed(y_hw[i]), $signed(y_ref[i]));
                    errors = errors + 1;
                end else
                    $display("  PASS  y[%0d]: HW=%0d  REF=%0d", i, $signed(y_hw[i]), $signed(y_ref[i]));
            end
            $display("----------------------------------------");
            if (errors == 0)  $display("  *** ALL %0d ELEMENTS MATCH (BRAM snoop) ***", D);
            else              $display("  *** %0d / %0d MISMATCH (BRAM snoop) ***", errors, D);
            $display("========================================");

            // --- Method 2: output_tile_data port ---
            errors = 0;
            $display("");
            $display("========================================");
            $display("  FINAL OUTPUT (output_tile_data port, 1×%0d)", D);
            $display("========================================");
            for (i = 0; i < D; i = i + 1) begin
                if (y_hw_port[i] !== y_ref[i]) begin
                    $display("  FAIL  y[%0d]: PORT=%0d  REF=%0d", i, $signed(y_hw_port[i]), $signed(y_ref[i]));
                    errors = errors + 1;
                end else
                    $display("  PASS  y[%0d]: PORT=%0d  REF=%0d", i, $signed(y_hw_port[i]), $signed(y_ref[i]));
            end
            $display("----------------------------------------");
            if (errors == 0)  $display("  *** ALL %0d ELEMENTS MATCH (output_tile_data port) ***", D);
            else              $display("  *** %0d / %0d MISMATCH (output_tile_data port) ***", errors, D);
            $display("========================================");
        end
    endtask

    // ===================================================================
    // Main Test Sequence
    // ===================================================================
    integer i;
    initial begin
        $dumpfile("ffn_complete.vcd");
        $dumpvars(0, tb_ffn_complete);

        for (i = 0; i < D; i = i + 1)    y_hw[i] = 0;
        for (i = 0; i < D; i = i + 1)    y_hw_port[i] = 0;
        for (i = 0; i < 4*D; i = i + 1)  hw_relu[i] = 0;
        hw_tile_cnt = 0;  hw_relu_cnt = 0;  port_tile_cnt = 0;

        init_matrices;
        populate_memory;
        compute_software_reference;

        $display("");
        $display("====== Resetting DUT ======");
        rst_n <= 1'b0; start <= 1'b0;
        #(CLK_PERIOD * 10);
        rst_n <= 1'b1;
        #(CLK_PERIOD * 5);

        $display("====== Starting hardware FFN (D=%0d M=%0d) ======", D, M);
        start <= 1'b1;
        #(CLK_PERIOD);
        start <= 1'b0;

        wait(done);
        $display("[HW] Computation complete at time %0t", $time);
        #(CLK_PERIOD * 20);

        compare_relu;
        compare_results;

        $display("");
        $display("Simulation finished at time %0t", $time);
        $finish;
    end

    // Timeout watchdog
    initial begin #(CLK_PERIOD * 500000); $display("TIMEOUT!"); $finish; end

    // AXI monitor
    always @(posedge clk)
        if (arvalid && arready)
            $display("[AXI] Read: addr=0x%08h  len=%0d  size=%0d", araddr, arlen+1, arsize);

    // Pipeline stage monitor
    always @(posedge clk) begin
        if (u_dut.u_up_projection.valid_out)
            $display("[UP]  Output: col=%0d  [%0d %0d %0d %0d]",
                     u_dut.u_up_projection.result_col,
                     $signed(u_dut.u_up_projection.result_tile[0*DATA_W +: DATA_W]),
                     $signed(u_dut.u_up_projection.result_tile[1*DATA_W +: DATA_W]),
                     $signed(u_dut.u_up_projection.result_tile[2*DATA_W +: DATA_W]),
                     $signed(u_dut.u_up_projection.result_tile[3*DATA_W +: DATA_W]));
        if (u_dut.u_down_projection.valid_out)
            $display("[DN]  Output: col=%0d  [%0d %0d %0d %0d]",
                     u_dut.u_down_projection.result_col,
                     $signed(u_dut.u_down_projection.result_tile[0*DATA_W +: DATA_W]),
                     $signed(u_dut.u_down_projection.result_tile[1*DATA_W +: DATA_W]),
                     $signed(u_dut.u_down_projection.result_tile[2*DATA_W +: DATA_W]),
                     $signed(u_dut.u_down_projection.result_tile[3*DATA_W +: DATA_W]));
    end

endmodule
