//============================================================================
// accumulator.v — Stage 5: Output Accumulation (REVISED v2)
//============================================================================
// Receives completed 1×M tiles from the Down Projection stage.
// Decodes the tile_col index to determine where in the 1×D output
// vector each tile should be written.
//
// After all tiles are written, a readout FSM streams them out through
// output_tile_data / output_tile_addr / output_tile_valid with proper
// BRAM latency alignment.
//
// Timing analysis (critical path):
//
//   Because out_rd_en and out_rd_addr are registered outputs (non-blocking
//   in this always block), and the BRAM also has a registered read:
//
//   Cycle N   : FSM in RD_ISSUE  → sets out_rd_en<=1, out_rd_addr<=idx
//   Cycle N+1 : out_rd_en=1 reaches BRAM → BRAM latches mem[idx] into rd_reg
//   Cycle N+2 : out_rd_data = mem[idx]  → FSM in RD_CAPTURE reads it
//
//   Total: 2 cycles from RD_ISSUE to valid data.
//   FSM path: RD_ISSUE → RD_RELAY → RD_CAPTURE → (next RD_ISSUE or RD_DONE)
//
//============================================================================

module accumulator #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16,
    parameter DISABLE_READOUT = 0,  // 1 = disable internal readout FSM;
                                     //     external module controls BRAM reads
    parameter MAX_INNER = 256       // Max inner iterations = max(D/M, 4D/M)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output reg                           done,

    // ---- Input from Down Projection Stage ----
    input  wire [M*DATA_W-1:0]           result_tile,
    input  wire [`CLOG2_MIN1(MAX_INNER)-1:0]  result_col,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- Output BRAM Write Interface ----
    output reg                           out_wr_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       out_wr_addr,
    output reg  [M*DATA_W-1:0]          out_wr_data,

    // ---- Output BRAM Read Interface ----
    output reg                           out_rd_en,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       out_rd_addr,
    input  wire [M*DATA_W-1:0]          out_rd_data,

    // ---- Streaming output port (latency-aligned) ----
    output reg  [M*DATA_W-1:0]          output_tile_data,
    output reg  [`CLOG2_MIN1(D/M)-1:0]       output_tile_addr,
    output reg                           output_tile_valid,

    // ---- Legacy single-element output (kept for compatibility) ----
    output reg  [DATA_W-1:0]            output_data,
    output reg                           output_valid
);

    localparam NUM_TILES = D / M;

    // -------------------------------------------------------------------
    // Write-side state
    // -------------------------------------------------------------------
    reg [`CLOG2_MIN1(NUM_TILES)-1:0] tile_cnt;

    // -------------------------------------------------------------------
    // Readout FSM states
    // -------------------------------------------------------------------
    localparam RD_IDLE    = 3'd0;
    localparam RD_ISSUE   = 3'd1;  // Assert rd_en for current index (registered)
    localparam RD_RELAY   = 3'd2;  // Wait for registered rd_en to reach BRAM
    localparam RD_CAPTURE = 3'd3;  // BRAM data is now valid — capture & output
    localparam RD_DONE    = 3'd4;  // All tiles streamed out

    reg [2:0] rd_state;
    reg [`CLOG2_MIN1(NUM_TILES)-1:0] rd_idx;
    reg [`CLOG2_MIN1(NUM_TILES)-1:0] rd_idx_pipe;  // Pipelined address (aligns with data)
    reg        rd_done_flag;

    // -------------------------------------------------------------------
    // Main sequential logic
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out       <= 1'b1;
            done            <= 1'b0;
            tile_cnt        <= 0;
            out_wr_en       <= 1'b0;
            out_wr_addr     <= 0;
            out_wr_data     <= 0;
            out_rd_en       <= 1'b0;
            out_rd_addr     <= 0;
            output_tile_data  <= 0;
            output_tile_addr  <= 0;
            output_tile_valid <= 1'b0;
            output_data     <= 0;
            output_valid    <= 1'b0;
            rd_state        <= RD_IDLE;
            rd_idx          <= 0;
            rd_idx_pipe     <= 0;
            rd_done_flag    <= 1'b0;
        end else begin
            // Defaults: one-cycle pulses
            out_wr_en        <= 1'b0;
            out_rd_en        <= 1'b0;
            output_tile_valid <= 1'b0;
            output_valid     <= 1'b0;

            // -----------------------------------------------------------
            // Reset on new start
            // -----------------------------------------------------------
            if (start) begin
                done         <= 1'b0;
                tile_cnt     <= 0;
                rd_state     <= RD_IDLE;
                rd_done_flag <= 1'b0;
            end

            // -----------------------------------------------------------
            // Write path: accept result tile from down projection
            // -----------------------------------------------------------
            if (valid_in && ready_out) begin
                out_wr_en   <= 1'b1;
                out_wr_addr <= result_col;
                out_wr_data <= result_tile;

                tile_cnt <= tile_cnt + 1;

                if (tile_cnt == NUM_TILES - 1) begin
                    done <= 1'b1;
                end
            end

            // -----------------------------------------------------------
            // Readout FSM: stream output tiles after computation done
            // -----------------------------------------------------------
            case (rd_state)
                RD_IDLE: begin
                    if (!DISABLE_READOUT && done && !rd_done_flag) begin
                        rd_idx    <= 0;
                        rd_state  <= RD_ISSUE;
                    end
                end

                RD_ISSUE: begin
                    // Issue BRAM read for rd_idx.
                    // out_rd_en and out_rd_addr are non-blocking, so they
                    // take effect at the END of this cycle.
                    out_rd_en   <= 1'b1;
                    out_rd_addr <= rd_idx;
                    // Pipeline the address to align with data arrival
                    rd_idx_pipe <= rd_idx;
                    rd_state    <= RD_RELAY;
                end

                RD_RELAY: begin
                    // Cycle after RD_ISSUE: out_rd_en is now 1 at the BRAM
                    // input. The BRAM latches mem[rd_idx] into rd_reg at the
                    // end of THIS cycle. Data will be on out_rd_data next cycle.
                    // Deassert rd_en (already default 0).
                    rd_state <= RD_CAPTURE;
                end

                RD_CAPTURE: begin
                    // out_rd_data now holds mem[rd_idx_pipe].
                    // Capture and present on the output port.
                    output_tile_data  <= out_rd_data;
                    output_tile_addr  <= rd_idx_pipe;
                    output_tile_valid <= 1'b1;
                    // Legacy single-element output
                    output_data  <= out_rd_data[0*DATA_W +: DATA_W];
                    output_valid <= 1'b1;

                    if (rd_idx < NUM_TILES - 1) begin
                        rd_idx   <= rd_idx + 1;
                        rd_state <= RD_ISSUE;
                    end else begin
                        rd_done_flag <= 1'b1;
                        rd_state     <= RD_DONE;
                    end
                end

                RD_DONE: begin
                    // Stay here until next start resets us
                    rd_done_flag <= 1'b1;
                end

                default: rd_state <= RD_IDLE;
            endcase

            ready_out <= 1'b1;
        end
    end

endmodule
