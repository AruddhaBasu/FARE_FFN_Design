//============================================================================
// axi_read_master.v — Simplified AXI4 Read Master
//============================================================================
// Accepts a read request (base address + burst length), issues AXI4 AR
// channel transactions, collects R channel data, and presents results
// through a simple output interface.
//
// The module handles one burst at a time. A new request is accepted only
// after the previous burst completes.
//
// AXI4 Signals used:
//   AR channel: araddr, arlen, arsize, arburst, arvalid, arready
//   R  channel: rdata, rlast, rvalid, rready
//
// Parameters:
//   AXI_DATA_W : Width of the AXI data bus (e.g., 128)
//   AXI_ADDR_W : Width of the AXI address bus (e.g., 32)
//============================================================================

module axi_read_master #(
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Request interface (from fetch stage) ----
    input  wire [AXI_ADDR_W-1:0]        req_addr,      // Burst start address
    input  wire [7:0]                   req_len,        // Burst length (number of beats - 1)
    input  wire [2:0]                   req_size,       // Transfer size (bytes per beat: 3=8bytes, 4=16bytes, etc.)
    input  wire                          req_valid,      // Request is valid
    output wire                          req_ready,      // Can accept a new request

    // ---- Response interface (to fetch stage) ----
    output wire [AXI_DATA_W-1:0]        resp_data,      // Data beat
    output wire                          resp_last,      // Last beat of burst
    output wire                          resp_valid,     // Data beat is valid
    input  wire                          resp_ready,     // Consumer can accept data

    // ---- AXI4 Read Address Channel ----
    output reg  [AXI_ADDR_W-1:0]        araddr,
    output reg  [7:0]                   arlen,
    output reg  [2:0]                   arsize,
    output reg  [1:0]                   arburst,
    output reg                           arvalid,
    input  wire                          arready,

    // ---- AXI4 Read Data Channel ----
    input  wire [AXI_DATA_W-1:0]        rdata,
    input  wire                          rlast,
    input  wire                          rvalid,
    output reg                           rready
);

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE    = 2'd0;
    localparam S_AR      = 2'd1;  // Driving AR channel
    localparam S_R       = 2'd2;  // Receiving R channel

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            arvalid  <= 1'b0;
            rready   <= 1'b0;
            araddr   <= '0;
            arlen    <= '0;
            arsize   <= '0;
            arburst  <= 2'b01; // INCR
        end else begin
            case (state)
                S_IDLE: begin
                    arvalid <= 1'b0;
                    rready  <= 1'b0;
                    if (req_valid) begin
                        araddr   <= req_addr;
                        arlen    <= req_len;
                        arsize   <= req_size;
                        arburst  <= 2'b01; // INCR burst
                        arvalid  <= 1'b1;
                        state    <= S_AR;
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
                    if (rvalid && rready) begin
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

    // Request can be accepted only in IDLE
    assign req_ready = (state == S_IDLE);

    // Response data: directly from AXI R channel
    assign resp_data  = rdata;
    assign resp_last  = rlast;
    assign resp_valid = rvalid && (state == S_R);

endmodule
