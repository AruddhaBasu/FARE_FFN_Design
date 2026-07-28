//============================================================================
// tanh_lut.v — BRAM-Based Tanh Lookup Table for Dynamic Tanh (DyT)
//============================================================================
// Implements tanh(x) using a BRAM lookup table with saturation handling.
//
// Design:
//   - Input: s_val (signed, Q_S_FRAC.W format) — the product α*z
//   - Output: tanh_out (signed Q8.8) — tanh(s_real)
//   - For |s_real| >= SAT_LIMIT (= 4.0), output = sign(s) * 1.0 = ±256 in Q8.8
//   - For |s_real| <  SAT_LIMIT, look up in BRAM ROM
//   - Uses the property tanh(-x) = -tanh(x): LUT stores only positive values
//
// BRAM ROM initialization:
//   - 1024 entries covering real range [0, 4.0) with step 1/256
//   - Each entry: 16-bit signed Q8.8 value of tanh(index/256)
//   - Loaded from tanh_lut_init.hex via $readmemh
//
// Read timing (synchronous BRAM, 1-cycle latency):
//   Cycle N  : rd_en sampled high with valid address
//   Cycle N+1: lut_mem[lut_addr] appears on rd_reg, tanh_out is valid
//
// Parameters:
//   DATA_W    : Output data width (16 for Q8.8)
//   FRAC_W    : Fractional bits in output (8 for Q8.8)
//   S_WIDTH   : Input width (32 for Q16.16 product)
//   S_FRAC    : Fractional bits in input (16 for Q16.16)
//   SAT_LIMIT_INT : Saturation threshold in integer units (4 = 4.0 real)
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
    // Lookup request
    input  wire                          rd_en,
    input  wire  signed [S_WIDTH-1:0]    s_val,
    // Lookup result (valid 1 cycle after rd_en)
    output reg   signed [DATA_W-1:0]     tanh_out,
    output reg                           tanh_valid
);

    // -------------------------------------------------------------------
    // Derived constants
    // -------------------------------------------------------------------
    localparam LUT_DEPTH    = SAT_LIMIT_INT * (1 << FRAC_W);   // 1024
    localparam LUT_ADDR_W   = $clog2(LUT_DEPTH);               // 10
    localparam signed [DATA_W-1:0] ONE_Q = (1 << FRAC_W);      // 256 = 1.0 in Q8.8
    localparam signed [S_WIDTH-1:0] SAT_LIMIT_Q =
        SAT_LIMIT_INT * (1 << S_FRAC);                         // 4.0 in Q_S_FRAC
    // Most-negative safe-guard for abs()
    localparam signed [S_WIDTH-1:0] MS_NEG = - (1 << (S_WIDTH-1));

    // -------------------------------------------------------------------
    // BRAM ROM for positive tanh values
    // -------------------------------------------------------------------
    (* ram_style = "block" *) reg [DATA_W-1:0] lut_mem [0:LUT_DEPTH-1];

    initial begin
        $readmemh("tanh_lut_init.hex", lut_mem);
    end

    // Registered read data (1-cycle latency)
    reg [DATA_W-1:0] rd_reg;
    reg              sat_reg;
    reg              sign_reg;

    // -------------------------------------------------------------------
    // Input processing — ABSOLUTE VALUE (safe against MS_NEG)
    // -------------------------------------------------------------------
    // Using combinational negation with overflow-safe behavior. The worst
    // case (s_val == -2^(S_WIDTH-1)) produces a negative abs that maps into
    // saturation because s_idx_full will exceed LUT_DEPTH after the shift.
    // We use the explicit comparison instead of a unary '-', which in some
    // tools retains the negative sign for MS_NEG.
    wire signed [S_WIDTH-1:0] abs_s = (s_val < 0 && s_val != MS_NEG) ? -s_val :
                                      (s_val == MS_NEG) ? SAT_LIMIT_Q : s_val;

    // Convert Q_S_FRAC to Q_FRAC.FRAC for LUT indexing
    wire [S_WIDTH-1:0] s_idx_full = abs_s >> (S_FRAC - FRAC_W);

    // Saturation flag and clamped address
    wire is_saturated = (s_idx_full >= LUT_DEPTH);
    wire [LUT_ADDR_W-1:0] lut_addr =
        is_saturated ? {LUT_ADDR_W{1'b0}} : s_idx_full[LUT_ADDR_W-1:0];

    // Sign of input
    wire s_sign = s_val[S_WIDTH-1];

    // -------------------------------------------------------------------
    // Synchronous BRAM read — registered address path (Vivado-inferable)
    // -------------------------------------------------------------------
    // To infer BRAM (not distributed RAM/LUTRAM), the BRAM read address and
    // controls must be consumed inside the clocked always block. The earlier
    // combinational-address version caused Vivado to infer distributed RAM
    // or fail BRAM inference when rd_en was not registered identically.
    // This block registers the BRAM read so synthesis can recognize a BRAM.
    // -------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_reg     <= {DATA_W{1'b0}};
            sat_reg    <= 1'b0;
            sign_reg   <= 1'b0;
            tanh_out   <= {DATA_W{1'b0}};
            tanh_valid <= 1'b0;
        end else begin
            // Default: data not valid
            tanh_valid <= 1'b0;
            if (rd_en) begin
                // Perform the BRAM read synchronously. If saturated,
                // lut_mem access is to addr 0 (safe, ignored anyway).
                rd_reg   <= lut_mem[lut_addr];
                sat_reg  <= is_saturated;
                sign_reg <= s_sign;
                // Reconstruct output (registered) using the *captured* flags
                // so that tanh_out is available 1 cycle after rd_en rises,
                // matching the documented 1-cycle latency.
                begin : out_recon
                    reg signed [DATA_W-1:0] tanh_pos;
                    tanh_pos = is_saturated ? ONE_Q : lut_mem[lut_addr];
                    tanh_out <= s_sign ? -tanh_pos : tanh_pos;
                end
                tanh_valid <= 1'b1;
            end
        end
    end

endmodule
