//============================================================================
// bram_dp.v — Simple Dual-Port BRAM (1 write port, 1 read port)
//============================================================================
// Port A: Write only
// Port B: Read only (synchronous read, 1-cycle latency)
//
// Parameters:
//   DATA_W : Bit width of each word
//   DEPTH  : Number of words
//   ADDR_W : Address width (`CLOG2_MIN1(DEPTH))
//============================================================================

module bram_dp #(
    parameter DATA_W = 16,
    parameter DEPTH  = 256,
    parameter ADDR_W = `CLOG2_MIN1(DEPTH)
)(
    input  wire                  clk,
    // Port A — Write
    input  wire                  wr_en,
    input  wire [ADDR_W-1:0]    wr_addr,
    input  wire [DATA_W-1:0]    wr_data,
    // Port B — Read
    input  wire                  rd_en,
    input  wire [ADDR_W-1:0]    rd_addr,
    output wire [DATA_W-1:0]    rd_data
);

    // Storage array
    (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Port A: Synchronous write
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Port B: Synchronous read (1-cycle latency)
    reg [DATA_W-1:0] rd_reg;
    always @(posedge clk) begin
        if (rd_en) begin
            rd_reg <= mem[rd_addr];
        end
    end

    assign rd_data = rd_reg;

endmodule


//============================================================================
// bram_dp_wide — Wide-word Dual-Port BRAM for tile storage
//============================================================================
// Stores one tile per address (word width = TILE_SIZE × DATA_W)
// This allows reading/writing an entire 1×M tile in one cycle.
//
// Parameters:
//   DATA_W    : Bit width of each element
//   TILE_SIZE : Number of elements per tile (= M)
//   DEPTH     : Number of tiles
//============================================================================

module bram_dp_wide #(
    parameter DATA_W    = 16,
    parameter TILE_SIZE = 16,
    parameter DEPTH     = 64,
    parameter ADDR_W    = `CLOG2_MIN1(DEPTH)
)(
    input  wire                          clk,
    // Port A — Write
    input  wire                          wr_en,
    input  wire [ADDR_W-1:0]            wr_addr,
    input  wire [TILE_SIZE*DATA_W-1:0]   wr_data,
    // Port B — Read
    input  wire                          rd_en,
    input  wire [ADDR_W-1:0]            rd_addr,
    output wire [TILE_SIZE*DATA_W-1:0]   rd_data
);

    localparam WORD_W = TILE_SIZE * DATA_W;

    (* ram_style = "block" *) reg [WORD_W-1:0] mem [0:DEPTH-1];

    // Port A: Write
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Port B: Synchronous read
    reg [WORD_W-1:0] rd_reg;
    always @(posedge clk) begin
        if (rd_en) begin
            rd_reg <= mem[rd_addr];
        end
    end

    assign rd_data = rd_reg;

endmodule
