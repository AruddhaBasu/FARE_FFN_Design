//============================================================================
// axi_read_master.v — Simplified AXI4 Read Master
//============================================================================
// One burst at a time. Accepts req (addr/len/size/valid) on a simple
// req/resp interface and drives AR/R channels. resp_data/last/valid are
// direct, gated copies of rdata/rlast/rvalid.
//============================================================================

module axi_read_master #(
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,
    // Request
    input  wire [AXI_ADDR_W-1:0]  req_addr,
    input  wire [7:0]             req_len,
    input  wire [2:0]             req_size,
    input  wire                    req_valid,
    output wire                    req_ready,
    // Response (passthrough of R channel while in S_R)
    output wire [AXI_DATA_W-1:0]  resp_data,
    output wire                    resp_last,
    output wire                    resp_valid,
    input  wire                    resp_ready,
    // AXI4 AR
    output reg  [AXI_ADDR_W-1:0]  araddr,
    output reg  [7:0]             arlen,
    output reg  [2:0]             arsize,
    output reg  [1:0]             arburst,
    output reg                     arvalid,
    input  wire                    arready,
    // AXI4 R
    input  wire [AXI_DATA_W-1:0]  rdata,
    input  wire                    rlast,
    input  wire                    rvalid,
    output reg                     rready
);

    localparam S_IDLE = 2'd0, S_AR = 2'd1, S_R = 2'd2;
    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            arvalid <= 1'b0;
            rready  <= 1'b0;
            araddr  <= '0;
            arlen   <= '0;
            arsize  <= '0;
            arburst <= 2'b01;
        end else begin
            case (state)
                S_IDLE: begin
                    arvalid <= 1'b0;
                    rready  <= 1'b0;
                    if (req_valid) begin
                        araddr  <= req_addr;
                        arlen   <= req_len;
                        arsize  <= req_size;
                        arburst <= 2'b01;
                        arvalid <= 1'b1;
                        state   <= S_AR;
                    end
                end
                S_AR: begin
                    if (arready && arvalid) begin
                        arvalid <= 1'b0;
                        rready  <= 1'b1;
                        state   <= S_R;
                    end
                end
                S_R: begin
                    // Accept a beat only when consumer is ready (backpressure)
                    if (rvalid && rready && resp_ready) begin
                        if (rlast) begin
                            rready <= 1'b0;
                            state  <= S_IDLE;
                        end
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    assign req_ready  = (state == S_IDLE);
    assign resp_data  = rdata;
    assign resp_last  = rlast;
    // Only present beats when we're in R and consumer is ready to take them
    assign resp_valid = rvalid && (state == S_R) && resp_ready;

endmodule
