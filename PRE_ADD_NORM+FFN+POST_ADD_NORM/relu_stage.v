//============================================================================
// relu_stage.v — Element-wise ReLU: max(0, x)
//============================================================================
// All M elements processed in parallel by M comparators.
// Result written to relu_bram and also forwarded downstream.
//============================================================================

module relu_stage #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire [M*DATA_W-1:0]        input_tile,
    input  wire [$clog2(4*D/M)-1:0]   tile_col,
    input  wire                       valid_in,
    output reg                        ready_out,
    output reg                        relu_wr_en,
    output reg  [$clog2(4*D/M)-1:0]  relu_wr_addr,
    output reg  [M*DATA_W-1:0]       relu_wr_data,
    output reg  [M*DATA_W-1:0]       result_tile,
    output reg  [$clog2(4*D/M)-1:0]  result_col,
    output reg                        valid_out,
    input  wire                       ready_in
);

    // Combinational ReLU
    reg [M*DATA_W-1:0] relu_output;
    integer j;
    always @(*) begin
        for (j = 0; j < M; j = j + 1) begin
            if (input_tile[j*DATA_W + DATA_W - 1])
                relu_output[j*DATA_W +: DATA_W] = {DATA_W{1'b0}};
            else
                relu_output[j*DATA_W +: DATA_W] = input_tile[j*DATA_W +: DATA_W];
        end
    end

    // Pipeline register + BRAM write
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out    <= 1'b1;
            valid_out    <= 1'b0;
            result_tile  <= '0;
            result_col   <= '0;
            relu_wr_en   <= 1'b0;
            relu_wr_addr <= '0;
            relu_wr_data <= '0;
        end else begin
            valid_out  <= 1'b0;
            relu_wr_en <= 1'b0;
            if (valid_in && ready_out) begin
                result_tile  <= relu_output;
                result_col   <= tile_col;
                valid_out    <= 1'b1;
                relu_wr_en   <= 1'b1;
                relu_wr_addr <= tile_col;
                relu_wr_data <= relu_output;
            end
            // Backpressure: accept upstream data only when downstream
            // can receive it (so we don't overwrite held output).
            ready_out <= ready_in || !valid_out;
        end
    end
endmodule
