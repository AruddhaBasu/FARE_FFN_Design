//============================================================================
// mul_col.v — Multiply-Add Column (M multipliers + adder tree)
//============================================================================
// Computes result = Σ_{k=0..M-1} input_tile[k] * weight_col[k].
// Uses WIDTH-SAFE multipliers: operands sign-extended to 2*DATA_W before
// multiply so the product is full 2*DATA_W (no truncation).
//============================================================================

module mul_col #(
    parameter M      = 32,
    parameter DATA_W = 16,
    parameter PIPE   = 1
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [M*DATA_W-1:0]           weight_col,
    input  wire                          valid_in,
    output wire [2*DATA_W+$clog2(M)-1:0] result,
    output wire                          result_valid
);

    localparam PROD_W = 2 * DATA_W;
    // Width-safe products (full PROD_W bits).
    wire [M*PROD_W-1:0] products;

    genvar k;
    generate
        for (k = 0; k < M; k = k + 1) begin : gen_mul
            wire signed [DATA_W-1:0]   a = input_tile [k*DATA_W +: DATA_W];
            wire signed [DATA_W-1:0]   b = weight_col [k*DATA_W +: DATA_W];
            // Widen to PROD_W BEFORE multiplying to keep the full product.
            wire signed [PROD_W-1:0]   aw = {{DATA_W{a[DATA_W-1]}}, a};
            wire signed [PROD_W-1:0]   bw = {{DATA_W{b[DATA_W-1]}}, b};
            wire signed [PROD_W-1:0]   p  = aw * bw;
            assign products[k*PROD_W +: PROD_W] = p;
        end
    endgenerate

    adder_tree #(.N(M), .IN_W(PROD_W), .PIPE(PIPE)) u_tree (
        .clk(clk), .rst_n(rst_n),
        .data_in(products), .valid_in(valid_in),
        .data_out(result), .valid_out(result_valid)
    );

endmodule
