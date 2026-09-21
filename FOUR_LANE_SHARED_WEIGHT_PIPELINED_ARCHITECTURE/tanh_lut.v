//============================================================================
// tanh_lut.v — BRAM-Based Tanh Lookup Table for Dynamic Tanh (DyT)
//============================================================================
// Implements tanh(x) using a BRAM lookup table with saturation handling.
//
// Design:
//   - Input: s_val (signed, Q_S_FRAC.W format) — the product α * z
//   - Output: tanh_out (signed Q8.8) — tanh(s_real)
//   - For |s_real| > 4.0, output = sign(s) * 1.0 = ±256 in Q8.8
//   - For |s_real| ≤ 4.0, look up in BRAM ROM
//   - Uses the property tanh(-x) = -tanh(x): LUT stores only positive values
//
// BRAM ROM initialization:
//   - 1024 entries covering real range [0, 4.0) with step 1/256
//   - Each entry: 16-bit signed Q8.8 value of tanh(index/256)
//   - Loaded from tanh_lut_init.hex via $readmemh
//
// Read timing (synchronous BRAM, 1-cycle latency):
//   Cycle N  : External module sets rd_en with combinational s_val
//   Cycle N+1: BRAM latches lut_mem[lut_addr] into rd_reg at posedge
//   Cycle N+2: rd_data available via rd_reg output
//
// Parameters:
//   DATA_W    : Output data width (16 for Q8.8)
//   FRAC_W    : Fractional bits in output (8 for Q8.8)
//   S_WIDTH   : Input width (32 for Q16.16 product)
//   S_FRAC    : Fractional bits in input (16 for Q16.16)
//   SAT_LIMIT_INT : Saturation threshold as integer (4 = 4.0 in real)
//============================================================================

module tanh_lut #(
    parameter DATA_W        = 16,
    parameter FRAC_W        = 8,
    parameter S_WIDTH       = 32,
    parameter S_FRAC        = 16,
    parameter SAT_LIMIT_INT = 4      // Integer saturation threshold (4 = 4.0)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Lookup request (combinational address, synchronous read) ----
    input  wire                          rd_en,
    input  wire  signed [S_WIDTH-1:0]    s_val,

    // ---- Lookup result (available 1 cycle after rd_en) ----
    output wire  signed [DATA_W-1:0]     tanh_out,
    output wire                          tanh_valid
);

    // -------------------------------------------------------------------
    // Derived constants — all integer arithmetic (Vivado-safe)
    // -------------------------------------------------------------------
    // LUT covers [0, SAT_LIMIT) with step 2^{-FRAC_W}
    // LUT_DEPTH = SAT_LIMIT_INT * 2^{FRAC_W} = 4 * 256 = 1024
    localparam LUT_DEPTH    = SAT_LIMIT_INT * (1 << FRAC_W);
    localparam LUT_ADDR_W   = $clog2(LUT_DEPTH);

    // SAT_LIMIT in Q_S_FRAC format: 4.0 * 2^{S_FRAC} = 4 << S_FRAC
    // Used for documentation/debug; actual saturation uses s_idx_full >= LUT_DEPTH
    localparam signed [S_WIDTH-1:0] SAT_LIMIT_Q = SAT_LIMIT_INT * (1 << S_FRAC);

    // 1.0 in Q8.8 = 256
    localparam signed [DATA_W-1:0] ONE_Q8 = (1 << FRAC_W);

    // -------------------------------------------------------------------
    // BRAM ROM for tanh values (positive inputs only)
    // -------------------------------------------------------------------
    (* ram_style = "block" *) reg [DATA_W-1:0] lut_mem [0:LUT_DEPTH-1];

    initial begin
        $readmemh("tanh_lut_init.hex", lut_mem);
    end

    // -------------------------------------------------------------------
    // Input processing: extract absolute value and LUT index
    // -------------------------------------------------------------------
    wire signed [S_WIDTH-1:0] abs_s;
    assign abs_s = (s_val < 0) ? -s_val : s_val;

    // s_idx = abs_s >> (S_FRAC - FRAC_W)
    // This converts Q_S_FRAC to Q_int.FRAC_W for LUT indexing
    wire [S_WIDTH-1:0] s_idx_full;
    assign s_idx_full = abs_s >> (S_FRAC - FRAC_W);

    // Saturation: |s_real| > SAT_LIMIT ↔ s_idx_full >= LUT_DEPTH
    wire is_saturated = (s_idx_full >= LUT_DEPTH);

    // LUT address: clamp to valid range
    wire [LUT_ADDR_W-1:0] lut_addr;
    assign lut_addr = is_saturated ? (LUT_DEPTH - 1) : s_idx_full[LUT_ADDR_W-1:0];

    // Sign of input for output reconstruction
    wire s_sign = s_val[S_WIDTH-1];

    // -------------------------------------------------------------------
    // BRAM synchronous read
    // -------------------------------------------------------------------
    reg [DATA_W-1:0] rd_reg;
    reg              sat_reg;
    reg              sign_reg;

    always @(posedge clk) begin
        if (rd_en) begin
            rd_reg   <= lut_mem[lut_addr];
            sat_reg  <= is_saturated;
            sign_reg <= s_sign;
        end
    end

    // -------------------------------------------------------------------
    // Output reconstruction: apply sign and saturation
    // -------------------------------------------------------------------
    wire signed [DATA_W-1:0] tanh_pos;
    assign tanh_pos = sat_reg ? ONE_Q8 : rd_reg;

    assign tanh_out = sign_reg ? -tanh_pos : tanh_pos;

    // Valid signal: 1 cycle after rd_en
    reg valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= 1'b0;
        else
            valid_pipe <= rd_en;
    end

    assign tanh_valid = valid_pipe;

endmodule
