//============================================================================
// bram_dp.v — Simple Dual-Port BRAM (1W/1R, 1-cycle registered read)
//============================================================================

module bram_dp #(
    parameter DATA_W = 16,
    parameter DEPTH  = 256,
    parameter ADDR_W = $clog2(DEPTH)
)(
    input  wire               clk,
    // Port A — write
    input  wire               wr_en,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire [DATA_W-1:0] wr_data,
    // Port B — read (synchronous, 1-cycle latency)
    input  wire               rd_en,
    input  wire [ADDR_W-1:0] rd_addr,
    output reg  [DATA_W-1:0] rd_data
);

    (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Port A: sync write
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    // Port B: sync read
    always @(posedge clk) begin
        if (rd_en) rd_data <= mem[rd_addr];
    end
endmodule
