`timescale 1ns / 1ps

module tb_ffn_bram_pipeline;

    // Parameters matching UUT
    parameter N = 8;
    parameter BIT_WIDTH = 16;
    parameter FRAC_BIT = 8;
    parameter TILE_WIDTH = 1024; // 8x8 elements * 16-bits

    // Testbench Signals
    reg clk;
    reg rst;
    reg start;
    reg signed [(N*BIT_WIDTH)-1:0] row_input;
    
    wire [1:0] up_bram_addr;
    reg signed [TILE_WIDTH-1:0] up_bram_rdata;
    
    wire [1:0] down_bram_addr;
    reg signed [TILE_WIDTH-1:0] down_bram_rdata;
    
    wire signed [(N*BIT_WIDTH)-1:0] accumulator_bank;
    wire done;

    // Emulated External Synchronous BRAM Memories
    reg [TILE_WIDTH-1:0] up_bram_memory [0:3];
    reg [TILE_WIDTH-1:0] down_bram_memory [0:3];

    // Instantiate Unit Under Test (UUT)
    ffn_bram_pipeline #(
        .N(N),
        .BIT_WIDTH(BIT_WIDTH),
        .FRAC_BIT(FRAC_BIT),
        .TILE_WIDTH(TILE_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .row_input(row_input),
        .up_bram_addr(up_bram_addr),
        .up_bram_rdata(up_bram_rdata),
        .down_bram_addr(down_bram_addr),
        .down_bram_rdata(down_bram_rdata),
        .accumulator_bank(accumulator_bank),
        .done(done)
    );
  initial begin
    $dumpfile ("OUTPUT_WAVEFORMS.vcd");
    $dumpvars (0, tb_ffn_bram_pipeline);
  end

    // 1-Cycle Latency Clocked Read Behavior for BRAMs
    always @(posedge clk) begin
        up_bram_rdata   <= up_bram_memory[up_bram_addr];
        down_bram_rdata <= down_bram_memory[down_bram_addr];
    end

    // Clock Generator (50MHz -> 20ns period)
    always #10 clk = ~clk;

    // Variables for population loops
    integer t, idx;
    reg signed [BIT_WIDTH-1:0] sample_input;
    reg signed [BIT_WIDTH-1:0] sample_up_w;
    reg signed [BIT_WIDTH-1:0] sample_down_w;

    initial begin
        // Initialize Signals
        clk = 0;
        rst = 1;
        start = 0;
        row_input = 0;

        // --- PRE-POPULATE MOCK STORAGE DATA ---
        sample_input  = 16'h0100; // Value: 1.0  (Q8.8)
        sample_up_w   = 16'h0080; // Value: 0.5  (Q8.8)
        sample_down_w = 16'h0040; // Value: 0.25 (Q8.8)

        // Structure the 1x8 flat input row vector
        for (idx = 0; idx < N; idx = idx + 1) begin
            row_input[idx*BIT_WIDTH +: BIT_WIDTH] = sample_input;
        end

        // Populate the 4 virtual memory slots inside BRAM with flattened 8x8 arrays
        for (t = 0; t < 4; t = t + 1) begin
            for (idx = 0; idx < 64; idx = idx + 1) begin
                up_bram_memory[t][idx*BIT_WIDTH +: BIT_WIDTH]   = sample_up_w;
                down_bram_memory[t][idx*BIT_WIDTH +: BIT_WIDTH] = sample_down_w;
            end
        end

        // --- SYSTEM RESET TIMING ---
        #40;
        rst = 0;
        #20;

        // --- START ACCELERATION EXECUTION ---
        $display("[TB TIMELINE] Pulsing Accelerator START Asserted.");
        start = 1;
        #20;
        start = 0; // De-assert start; system operates autonomously now

        // --- WAIT FOR PIPELINE FLUSH & COMPLETE FLAG ---
        @(posedge done);
        #1; // Brief settle step
        $display("[TB TIMELINE] Hardware Pipeline Finish Execution Flag Caught.");
        
        // --- DISPLAY MATRIX CALCULATION VALUES ---
        $display("\n=======================================================");
        $display("          FINAL FORWARD PASS ACCUMULATOR RESULTS       ");
        $display("=======================================================");
        for (idx = 0; idx < N; idx = idx + 1) begin
            $display("Accumulator Node [%0d] | Hex: 16'h%h | Real Appx Decimal: %f", 
                     idx, 
                     accumulator_bank[idx*BIT_WIDTH +: BIT_WIDTH],
                     $itor(accumulator_bank[idx*BIT_WIDTH +: BIT_WIDTH]) / 256.0);
        end
        $display("=======================================================\n");

        // Finish Simulation Safely
        $finish;
    end

endmodule
