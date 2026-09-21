//============================================================================
// adder_tree.v — Parameterized Pipelined Adder Tree (Vivado-Safe)
//============================================================================
// Reduces N inputs to a single sum using a binary tree.
//
// This version uses a FLAT structure with explicit wires at each level
// instead of large 2D arrays. This avoids creating massive intermediate
// buses that crash Vivado's HDConfig/BelGrid database.
//
// For N=32, PIPE=1: 5 pipeline stages, 5-cycle latency.
// Output width = IN_W + $clog2(N)
//============================================================================

module adder_tree #(
    parameter N     = 16,
    parameter IN_W  = 32,
    parameter PIPE  = 0
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire  [N*IN_W-1:0]          data_in,
    input  wire                        valid_in,
    output wire  [IN_W+$clog2(N)-1:0]  data_out,
    output wire                        valid_out
);

    localparam LEVELS = $clog2(N);
    localparam OUT_W  = IN_W + LEVELS;

    // ===================================================================
    // Stage 0: Sign-extend all N inputs from IN_W to OUT_W
    // ===================================================================
    wire [N*OUT_W-1:0] stage0;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_se0
            assign stage0[i*OUT_W +: OUT_W] =
                {{LEVELS{data_in[i*IN_W + IN_W - 1]}}, data_in[i*IN_W +: IN_W]};
        end
    endgenerate

    // ===================================================================
    // Build each level as a flat bus of half the entries
    //
    // At level l:  inputs  = N/(2^l)  values, each (IN_W+l) bits
    //              outputs = N/(2^(l+1)) values, each (IN_W+l+1) bits
    //
    // We pad each value to OUT_W for consistent indexing.
    // Active bits at level l: (IN_W+l) per entry.
    // ===================================================================

    // We need intermediate buses between levels.
    // For Vivado compatibility, we declare them with maximum width
    // and only use the first (count * OUT_W) bits at each level.

    // Stage buses: stage_l_out has (N >> (l+1)) entries padded to OUT_W
    // We use a generate chain: each level reads from the previous stage.

    // Pipeline registers for each level (used when PIPE=1)
    // We declare them as flat buses too.

    // Chain: stage0 → (level 0 logic) → stage1 → (level 1 logic) → ... → result

    // For clean Vivado synthesis, we build a Generate chain where
    // each level instantiates its own local wires and regs.

    wire [N*OUT_W-1:0]  level_bus [0:LEVELS];
    reg  [N*OUT_W-1:0]  level_pipe [0:LEVELS];
    reg                  level_valid_pipe [0:LEVELS];

    assign level_bus[0] = stage0;

    genvar l;
    generate
        for (l = 0; l < LEVELS; l = l + 1) begin : gen_level

            localparam COUNT_IN  = N >> l;
            localparam COUNT_OUT = N >> (l + 1);
            localparam W_CUR     = IN_W + l;        // Active width at this level
            localparam W_NEXT    = IN_W + l + 1;    // Active width after addition

            // Source: either pipelined or combinational from previous level
            wire [N*OUT_W-1:0] src;
            if (PIPE && l > 0) begin : gen_src_pipe
                assign src = level_pipe[l];
            end else begin : gen_src_comb
                assign src = level_bus[l];
            end

            // Generate (COUNT_OUT) adder pairs
            genvar j;
            for (j = 0; j < COUNT_OUT; j = j + 1) begin : gen_add

                wire signed [W_CUR-1:0] a;
                wire signed [W_CUR-1:0] b;
                wire signed [W_NEXT-1:0] sum;

                // Extract pair from source bus (each entry padded to OUT_W)
                assign a   = src[(2*j)*OUT_W +: W_CUR];
                assign b   = src[(2*j+1)*OUT_W +: W_CUR];
                assign sum = $signed(a) + $signed(b);

                if (PIPE) begin : gen_pipelined
                    // Register the sum
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n)
                            level_pipe[l+1][j*OUT_W +: OUT_W] <= '0;
                        else if (level_valid_pipe[l])
                            level_pipe[l+1][j*OUT_W +: OUT_W] <=
                                {{(OUT_W - W_NEXT){1'b0}}, sum};
                    end
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        level_pipe[l+1][j*OUT_W +: OUT_W];
                end else begin : gen_combinational
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        {{(OUT_W - W_NEXT){1'b0}}, sum};
                end
            end

            // Fill unused entries with zero (for clean simulation)
            for (j = COUNT_OUT; j < N; j = j + 1) begin : gen_pad
                if (PIPE) begin : gen_pad_pipe
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        level_pipe[l+1][j*OUT_W +: OUT_W];
                end else begin : gen_pad_comb
                    assign level_bus[l+1][j*OUT_W +: OUT_W] = '0;
                end
            end
        end
    endgenerate

    // ===================================================================
    // Valid signal pipeline
    // ===================================================================
    integer vp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (vp = 0; vp <= LEVELS; vp = vp + 1)
                level_valid_pipe[vp] <= 1'b0;
        end else begin
            level_valid_pipe[0] <= valid_in;
            for (vp = 1; vp <= LEVELS; vp = vp + 1)
                level_valid_pipe[vp] <= level_valid_pipe[vp-1];
        end
    end

    // ===================================================================
    // Output
    // ===================================================================
    assign data_out  = level_bus[LEVELS][OUT_W-1:0];
    assign valid_out = PIPE ? level_valid_pipe[LEVELS] : valid_in;

endmodule
