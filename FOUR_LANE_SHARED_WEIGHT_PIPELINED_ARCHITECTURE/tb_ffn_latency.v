//=============================================================================
// tb_ffn_latency.v -- End-to-end latency benchmark for ffn_block_zynq
//
// The transaction is accepted when start/in_valid is asserted while the
// top-level block is idle/in_ready.  Completion is defined as the final
// serialized output beat: out_valid && out_last.  This is the correct
// transaction boundary for this DUT because one input transaction produces
// multiple output beats when M*DATA_W > AXI_DATA_W.
//
// Clock: 500 MHz -> 2.0 ns period
// Default sweep point: d=16, RTL tile M=d=16, hidden dimension=4*d=64
// Override at compile time, for example:
//   iverilog ... -Ptb_ffn_latency.D=32 -Ptb_ffn_latency.M=32
//=============================================================================

`timescale 1ns/1ps
module tb_ffn_latency #(
    parameter integer D          = 16,
    parameter integer M          = D,
    parameter integer N          = 1,
    parameter integer HIDDEN_DIM  = 4*D,
    parameter integer NUM_COLS   = 4,
    parameter integer DATA_W     = 16,
    parameter integer FRAC_W     = 8,
    parameter integer AXI_DATA_W = 64,
    parameter integer AXI_ADDR_W = 32,
    parameter integer TIMEOUT_CYCLES = 100000000
);

    localparam integer CLK_PERIOD_NS   = 2;       // 500 MHz
    localparam integer ELEMS_PER_BEAT  = AXI_DATA_W / DATA_W;
    localparam integer NUM_TILES_D     = D / M;
    localparam integer NUM_TILES_H     = HIDDEN_DIM / M;

    reg clk;
    reg rst_n;
    reg start;

    wire done;
    wire signed [DATA_W-1:0] alpha_pre  = 16'sd128;
    wire signed [DATA_W-1:0] alpha_post = 16'sd128;

    // AXI4 read channel
    wire [AXI_ADDR_W-1:0] araddr;
    wire [7:0]            arlen;
    wire [2:0]            arsize;
    wire [1:0]            arburst;
    wire                  arvalid;
    reg                   arready;
    reg  [AXI_DATA_W-1:0] rdata;
    reg                   rlast;
    reg                   rvalid;
    wire                  rready;

    // Streamed output
    wire [AXI_DATA_W-1:0] out_data;
    wire [`CLOG2_MIN1(D/M)-1:0] out_addr;
    wire [`CLOG2_MIN1(N)-1:0] out_seq_idx;
    wire [2:0]             out_offset;
    wire                   out_valid;
    wire                   out_last;
    wire                   out_row_last;

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
        .out_valid(out_valid), .out_last(out_last), .out_row_last(out_row_last),
        .out_seq_idx(out_seq_idx)
    );

    initial clk = 1'b0;
    always #(CLK_PERIOD_NS/2) clk = ~clk;

    //-------------------------------------------------------------------------
    // Minimal AXI4 read slave.  Data values are zero because this benchmark
    // measures control/datapath latency, not numerical correctness.  The
    // slave nevertheless obeys RVALID/RREADY and supports 256-beat bursts.
    //-------------------------------------------------------------------------
    reg       axi_busy;
    reg [8:0] beats_left;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            arready    <= 1'b1;
            rvalid     <= 1'b0;
            rlast      <= 1'b0;
            rdata      <= '0;
            axi_busy   <= 1'b0;
            beats_left <= 9'd0;
        end else if (!axi_busy) begin
            arready <= 1'b1;
            if (arvalid && arready) begin
                axi_busy   <= 1'b1;
                arready    <= 1'b0;
                beats_left <= {1'b0, arlen} + 9'd1;
                rvalid     <= 1'b1;
                rlast      <= (arlen == 8'd0);
                rdata      <= '0;
            end
        end else if (rvalid && rready) begin
            if (beats_left == 9'd1) begin
                axi_busy   <= 1'b0;
                arready    <= 1'b1;
                beats_left <= 9'd0;
                rvalid     <= 1'b0;
                rlast      <= 1'b0;
            end else begin
                beats_left <= beats_left - 9'd1;
                rvalid     <= 1'b1;
                rlast      <= (beats_left == 9'd2);
                rdata      <= '0;
            end
        end
    end

    //-------------------------------------------------------------------------
    // Block-wise phase timing.  The top-level phase FSM transitions are:
    //   0->1 PRE-DyT, 1->2 FFN, 2->3 POST-DyT, 3->0/1 completion/next row.
    // Totals are accumulated over all N sequence elements.
    //-------------------------------------------------------------------------
    reg [1:0] phase_prev;
    realtime phase_start_time;
    realtime pre_phase_total_ns;
    realtime ffn_phase_total_ns;
    realtime post_phase_total_ns;
    integer pre_phase_cycles, ffn_phase_cycles, post_phase_cycles;
    integer post_row_complete;
    integer latency_measured;
    realtime latency_ns;
    integer latency_cycles;

    initial begin
        phase_prev          = 2'd0;
        phase_start_time    = 0.0;
        pre_phase_total_ns  = 0.0;
        ffn_phase_total_ns  = 0.0;
        post_phase_total_ns = 0.0;
        pre_phase_cycles    = 0;
        ffn_phase_cycles    = 0;
        post_phase_cycles   = 0;
        post_row_complete   = 0;
    end

    always @(u_dut.phase_state) begin
        case ({phase_prev, u_dut.phase_state})
            4'b0001: begin
                phase_start_time = $realtime;
            end
            4'b0110: begin
                pre_phase_total_ns = pre_phase_total_ns +
                                     ($realtime - phase_start_time);
                pre_phase_cycles = $rtoi((pre_phase_total_ns / CLK_PERIOD_NS) + 0.5);
                phase_start_time = $realtime;
            end
            4'b1011: begin
                ffn_phase_total_ns = ffn_phase_total_ns +
                                     ($realtime - phase_start_time);
                ffn_phase_cycles = $rtoi((ffn_phase_total_ns / CLK_PERIOD_NS) + 0.5);
                phase_start_time = $realtime;
            end
            4'b1100: begin
                // The post-DyT interval is closed on out_row_last, so no
                // extra phase-controller completion cycle is included here.
                if (latency_measured) begin
                    $display("[BREAKDOWN] PRE-DyT  : %0.3f ns | %0d cycles",
                             pre_phase_total_ns, pre_phase_cycles);
                    $display("[BREAKDOWN] FFN      : %0.3f ns | %0d cycles",
                             ffn_phase_total_ns, ffn_phase_cycles);
                    $display("[BREAKDOWN] POST-DyT : %0.3f ns | %0d cycles",
                             post_phase_total_ns, post_phase_cycles);
                    $display("[BREAKDOWN] TOTAL    : %0.3f ns | %0d cycles",
                             latency_ns, latency_cycles);
                end
            end
            4'b1101: begin
                // Start timing the next sequence element's pre-DyT phase.
                phase_start_time = $realtime;
                post_row_complete = 0;
            end
            default: begin end
        endcase
        phase_prev = u_dut.phase_state;
    end

    //-------------------------------------------------------------------------
    // In-flight transaction latency measurement
    //-------------------------------------------------------------------------
    // The DUT has no explicit input/output ID ports because it permits only
    // one block transaction at a time.  The testbench therefore assigns a
    // monotonically increasing ID at the start handshake and carries the
    // active ID to the final output beat.
    // Icarus Verilog does not support an associative array whose element
    // type is realtime.  Use a fixed realtime table plus valid bits; this is
    // functionally equivalent to realtime start_times[int] for this
    // single-transaction-at-a-time DUT and is accepted by both Icarus and
    // Vivado/XSim.
    localparam integer MAX_TRANSACTIONS = 1024;
    realtime start_times [0:MAX_TRANSACTIONS-1];
    reg      start_valid [0:MAX_TRANSACTIONS-1];
    integer  next_transaction_id;
    integer  active_transaction_id;
    integer  in_id;
    integer  out_id;
    // latency_ns, latency_cycles, and latency_measured are declared above
    // so the phase monitor can print the final block breakdown.

    // Adapt the requested generic names to this block's interface.
    wire in_valid = start;
    wire in_ready = (u_dut.phase_state == 2'd0);

    always @* begin
        in_id  = next_transaction_id;
        out_id = active_transaction_id;
    end

    // Capture start time with a unique transaction ID.
    always @(posedge clk) begin
        if (in_valid && in_ready) begin
            if (in_id < MAX_TRANSACTIONS) begin
                start_times[in_id] = $realtime;
                start_valid[in_id] = 1'b1;
                active_transaction_id = in_id;
                next_transaction_id = next_transaction_id + 1;
                $display("[LAT] START ID=%0d | time=%0.3f ns", in_id, $realtime);
            end else begin
                $display("[LAT] ERROR: transaction ID table exhausted");
            end
        end
    end

    // Calculate end-to-end latency when the final output beat completes.
    // Do not measure every out_valid beat: the post-DyT block serializes one
    // tile into multiple beats.  out_last identifies transaction completion.
    always @(posedge clk) begin
        if (out_valid && out_row_last) begin
            post_phase_total_ns = post_phase_total_ns +
                                  ($realtime - phase_start_time);
            post_phase_cycles = $rtoi((post_phase_total_ns / CLK_PERIOD_NS) + 0.5);
            post_row_complete = 1;
        end

        if (out_valid && out_last) begin
            if ((out_id >= 0) && (out_id < MAX_TRANSACTIONS) && start_valid[out_id]) begin
                latency_ns = $realtime - start_times[out_id];
                latency_cycles = $rtoi((latency_ns / CLK_PERIOD_NS) + 0.5);
                $display("[LAT] ID: %0d | Latency: %0.3f ns | %0d cycles @ 500 MHz",
                         out_id, latency_ns, latency_cycles);
                start_valid[out_id] = 1'b0;
                latency_measured = 1;
            end else begin
                $display("[LAT] ERROR: completion ID %0d has no start timestamp", out_id);
            end
        end
    end

    //-------------------------------------------------------------------------
    // Test sequence
    //-------------------------------------------------------------------------
    integer wait_cycles;

    initial begin
        next_transaction_id   = 0;
        active_transaction_id = -1;
        latency_measured      = 0;
        rst_n                 = 1'b0;
        start                 = 1'b0;

        $display("============================================================");
        $display("Latency benchmark: d=%0d, RTL tile M=%0d, hidden dimension=%0d, N=%0d",
                 D, M, HIDDEN_DIM, N);
        $display("Clock frequency: 500 MHz, period: %0d ns, AXI width: %0d bits",
                 CLK_PERIOD_NS, AXI_DATA_W);
        $display("============================================================");

        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        // Assert one transaction.  The handshake is sampled on the next
        // rising edge while the top-level FSM is idle.
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait_cycles = 0;
        while (!latency_measured && wait_cycles < TIMEOUT_CYCLES) begin
            @(posedge clk);
            wait_cycles = wait_cycles + 1;
        end

        if (!latency_measured)
            $display("[LAT] ERROR: timeout after %0d cycles", TIMEOUT_CYCLES);

        // Allow the top-level FSM to consume post_done and transition back
        // to IDLE so the final POST-DyT timing interval is recorded.
        while (u_dut.phase_state != 2'd0)
            @(posedge clk);

        $display("[LAT] TEST COMPLETE for d=%0d, RTL tile M=%0d, hidden=%0d, N=%0d",
                 D, M, HIDDEN_DIM, N);
        $finish;
    end

endmodule
