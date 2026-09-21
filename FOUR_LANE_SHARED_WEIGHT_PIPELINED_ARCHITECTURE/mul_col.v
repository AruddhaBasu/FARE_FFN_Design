//============================================================================
// mul_col.v — Multiply-Add Column: M multipliers + 1 adder tree
//============================================================================
// Computes one output element of a tile partial product:
//   result = Σ_{k=0}^{M-1} input_tile[k] * weight_col[k]
//
// This hierarchical module replaces the flat M×M multiplier array.
// Instead of one massive 32768-bit product bus, we have M independent
// column modules, each with M multipliers and one adder tree.
//
// Benefits:
//   - No M*M*PROD_W flat bus (was 32768 bits for M=32)
//   - Only M*DATA_W weight input (per column, not the full tile)
//   - Vivado can place & route each column independently
//   - Fixes HDConfig::lookup() BelGrid crash
//============================================================================

module mul_col #(
    parameter M      = 32,
    parameter DATA_W = 16,
    parameter PIPE   = 1     // 1 = pipelined adder tree (recommended for M>8)
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // 1×M input tile
    input  wire [M*DATA_W-1:0]           input_tile,

    // 1×M weight column (k-th element of each row for this output column)
    input  wire [M*DATA_W-1:0]           weight_col,

    // Adder tree valid input
    input  wire                          valid_in,

    // Result
    output wire [DATA_W + $clog2(M) + DATA_W-1:0]  result,
    output wire                          result_valid
);

    localparam PROD_W = 2 * DATA_W;
    localparam SUM_W  = PROD_W + $clog2(M);

    // -------------------------------------------------------------------
    // M multipliers: products[k] = input_tile[k] * weight_col[k]
    // -------------------------------------------------------------------
    wire [M*PROD_W-1:0] products;

    genvar k;
    generate
        for (k = 0; k < M; k = k + 1) begin : gen_mul
            wire signed [DATA_W-1:0] a;
            wire signed [DATA_W-1:0] b;
            wire signed [PROD_W-1:0] p;

            assign a = input_tile[k*DATA_W +: DATA_W];
            assign b = weight_col[k*DATA_W +: DATA_W];
            assign p = a * b;

            assign products[k*PROD_W +: PROD_W] = p;
        end
    endgenerate

    // -------------------------------------------------------------------
    // Adder tree: M products → 1 sum
    // -------------------------------------------------------------------
    adder_tree #(
        .N     (M),
        .IN_W  (PROD_W),
        .PIPE  (PIPE)
    ) u_adder_tree (
        .clk       (clk),
        .rst_n     (rst_n),
        .data_in   (products),
        .valid_in  (valid_in),
        .data_out  (result),
        .valid_out (result_valid)
    );

endmodule
