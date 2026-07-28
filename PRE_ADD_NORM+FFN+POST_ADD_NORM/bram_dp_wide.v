//============================================================================
// bram_dp_wide.v — Wide-word Dual-Port BRAM (one tile per address)
//============================================================================
// Stores tiles of TILE_SIZE × DATA_W bits. One read + one write port.
// Read latency = 1 cycle (registered).
//============================================================================

module bram_dp_wide #(
    parameter DATA_W    = 16,
    parameter TILE_SIZE = 16,
    parameter DEPTH     = 64,
    parameter ADDR_W    = $clog2(DEPTH)
)(
    input  wire                            clk,
    input  wire                            wr_en,
    input  wire [ADDR_W-1:0]              wr_addr,
    input  wire [TILE_SIZE*DATA_W-1:0]     wr_data,
    input  wire                            rd_en,
    input  wire [ADDR_W-1:0]              rd_addr,
    output reg  [TILE_SIZE*DATA_W-1:0]     rd_data
);

    localparam WORD_W = TILE_SIZE * DATA_W;
    (* ram_style = "block" *) reg [WORD_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    always @(posedge clk) begin
        if (rd_en) rd_data <= mem[rd_addr];
    end
endmodule
