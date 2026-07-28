//============================================================================
// accumulator.v — Output Accumulator
//============================================================================
// Receives completed 1×M tiles from the down-projection stage and writes
// them into an output tile BRAM indexed by result_col. After all tiles are
// written, a readout FSM streams them back out with proper 2-cycle BRAM
// latency alignment.
//
// When DISABLE_READOUT=1 the internal readout FSM is disabled; an external
// module (post-DyT in this design) drives the output BRAM read port.
//============================================================================

module accumulator #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16,
    parameter DISABLE_READOUT = 0
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,
    output reg                        done,
    input  wire [M*DATA_W-1:0]        result_tile,
    input  wire [$clog2(D/M)-1:0]     result_col,
    input  wire                       valid_in,
    output reg                        ready_out,
    // Output BRAM write port
    output reg                        out_wr_en,
    output reg  [$clog2(D/M)-1:0]    out_wr_addr,
    output reg  [M*DATA_W-1:0]       out_wr_data,
    // Output BRAM read port (used by external module or internal readout)
    output reg                        out_rd_en,
    output reg  [$clog2(D/M)-1:0]    out_rd_addr,
    input  wire [M*DATA_W-1:0]       out_rd_data,
    // Streaming output (tile level)
    output reg  [M*DATA_W-1:0]       output_tile_data,
    output reg  [$clog2(D/M)-1:0]    output_tile_addr,
    output reg                        output_tile_valid,
    // Legacy single-element output
    output reg  [DATA_W-1:0]         output_data,
    output reg                        output_valid
);

    localparam NUM_TILES = D / M;

    reg [$clog2(NUM_TILES)-1:0] tile_cnt;

    localparam RD_IDLE=3'd0, RD_ISSUE=3'd1, RD_RELAY=3'd2, RD_CAPTURE=3'd3, RD_DONE=3'd4;
    reg [2:0] rd_state;
    reg [$clog2(NUM_TILES)-1:0] rd_idx, rd_idx_pipe;
    reg rd_done_flag;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out        <= 1'b1;
            done             <= 1'b0;
            tile_cnt         <= '0;
            out_wr_en        <= 1'b0;
            out_wr_addr      <= '0;
            out_wr_data      <= '0;
            out_rd_en        <= 1'b0;
            out_rd_addr      <= '0;
            output_tile_data <= '0;
            output_tile_addr <= '0;
            output_tile_valid<= 1'b0;
            output_data      <= '0;
            output_valid     <= 1'b0;
            rd_state         <= RD_IDLE;
            rd_idx           <= '0;
            rd_idx_pipe      <= '0;
            rd_done_flag     <= 1'b0;
        end else begin
            out_wr_en        <= 1'b0;
            out_rd_en        <= 1'b0;
            output_tile_valid<= 1'b0;
            output_valid     <= 1'b0;

            if (start) begin
                done         <= 1'b0;
                tile_cnt     <= '0;
                rd_state     <= RD_IDLE;
                rd_done_flag <= 1'b0;
            end

            // Write path
            if (valid_in && ready_out) begin
                out_wr_en   <= 1'b1;
                out_wr_addr <= result_col;
                out_wr_data <= result_tile;
                tile_cnt    <= tile_cnt + 1;
                if (result_col == NUM_TILES - 1 && tile_cnt >= NUM_TILES - 1)
                    done <= 1'b1;
            end

            // Readout FSM
            case (rd_state)
                RD_IDLE: begin
                    if (!DISABLE_READOUT && done && !rd_done_flag) begin
                        rd_idx   <= '0;
                        rd_state <= RD_ISSUE;
                    end
                end
                RD_ISSUE: begin
                    out_rd_en   <= 1'b1;
                    out_rd_addr <= rd_idx;
                    rd_idx_pipe <= rd_idx;
                    rd_state    <= RD_RELAY;
                end
                RD_RELAY: rd_state <= RD_CAPTURE;
                RD_CAPTURE: begin
                    output_tile_data  <= out_rd_data;
                    output_tile_addr  <= rd_idx_pipe;
                    output_tile_valid <= 1'b1;
                    output_data       <= out_rd_data[0*DATA_W +: DATA_W];
                    output_valid      <= 1'b1;
                    if (rd_idx == NUM_TILES - 1) begin
                        rd_done_flag <= 1'b1;
                        rd_state     <= RD_DONE;
                    end else begin
                        rd_idx   <= rd_idx + 1;
                        rd_state <= RD_ISSUE;
                    end
                end
                RD_DONE: rd_done_flag <= 1'b1;
                default: rd_state <= RD_IDLE;
            endcase

            ready_out <= 1'b1;
        end
    end
endmodule
