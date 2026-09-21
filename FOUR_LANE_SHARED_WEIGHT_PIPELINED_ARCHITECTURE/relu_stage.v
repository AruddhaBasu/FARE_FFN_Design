//============================================================================
// relu_stage.v — Stage 3: ReLU Activation
//============================================================================
// Applies ReLU (max(0, x)) element-wise to each element of the 1×M tile.
// All M elements are processed in parallel using M comparators.
//
// The ReLU output is written to a local BRAM (relu_bram) so the DOWN
// phase can read it later. Simultaneously, it passes data through
// boundary registers for pipelined operation.
//
// Parameters:
//   D, M, DATA_W
//============================================================================

module relu_stage #(
    parameter D          = 256,
    parameter M          = 16,
    parameter HIDDEN_DIM = 4*D,
    parameter DATA_W     = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Input from Up Projection Stage ----
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [`CLOG2_MIN1(HIDDEN_DIM/M)-1:0]      tile_col,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- ReLU BRAM Write Interface ----
    output reg                           relu_wr_en,
    output reg  [`CLOG2_MIN1(HIDDEN_DIM/M)-1:0]     relu_wr_addr,
    output reg  [M*DATA_W-1:0]          relu_wr_data,

    // ---- Output (pass-through for pipeline continuity) ----
    output reg  [M*DATA_W-1:0]          result_tile,
    output reg  [`CLOG2_MIN1(HIDDEN_DIM/M)-1:0]     result_col,
    output reg                           valid_out,
    input  wire                          ready_in
);

    // -------------------------------------------------------------------
    // ReLU: max(0, x) for each element — all M in parallel
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0] relu_output;

    integer j;
    always @(*) begin
        for (j = 0; j < M; j = j + 1) begin
            if (input_tile[j*DATA_W + DATA_W - 1]) begin
                // Negative → output 0
                relu_output[j*DATA_W +: DATA_W] = {DATA_W{1'b0}};
            end else begin
                // Non-negative → pass through
                relu_output[j*DATA_W +: DATA_W] = input_tile[j*DATA_W +: DATA_W];
            end
        end
    end

    // -------------------------------------------------------------------
    // Pipeline register + BRAM write
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            result_tile <= '0;
            result_col  <= '0;
            relu_wr_en  <= 1'b0;
            relu_wr_addr<= '0;
            relu_wr_data<= '0;
        end else begin
            // Default: deassert one-shot signals
            valid_out  <= 1'b0;
            relu_wr_en <= 1'b0;

            if (valid_in && ready_out) begin
                // Latch ReLU output into boundary register
                result_tile <= relu_output;
                result_col  <= tile_col;
                valid_out   <= 1'b1;

                // Write to ReLU BRAM simultaneously
                relu_wr_en   <= 1'b1;
                relu_wr_addr <= tile_col;
                relu_wr_data <= relu_output;
            end

            // Backpressure handling
            ready_out <= 1'b1;  // ReLU is always ready (combinational, no backpressure needed internally)
        end
    end

endmodule
