//============================================================================
// adder_tree.v — Pipelined Binary Adder Tree (Vivado-safe flat layout)
//============================================================================
// Reduces N signed inputs to a single sum.
// Output width = IN_W + $clog2(N). When PIPE=1, latency = $clog2(N) cycles
// (valid chain matches pipeline stages exactly).
//
// This implementation uses flat per-level buses with explicit sign-extension
// to avoid implicit 1-bit wires and to play nicely with Vivado synthesis.
//============================================================================

module adder_tree #(
    parameter N    = 16,
    parameter IN_W = 32,
    parameter PIPE = 0
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire [N*IN_W-1:0]           data_in,
    input  wire                        valid_in,
    output wire [IN_W+$clog2(N)-1:0]   data_out,
    output wire                        valid_out
);

    localparam LEVELS = $clog2(N);
    localparam OUT_W  = IN_W + LEVELS;

    // Stage 0: sign-extend all N inputs to OUT_W
    wire [N*OUT_W-1:0] stage0;
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_se0
            assign stage0[i*OUT_W +: OUT_W] =
                {{(OUT_W-IN_W){data_in[i*IN_W + IN_W - 1]}}, data_in[i*IN_W +: IN_W]};
        end
    endgenerate

    // Per-level buses (padded to N*OUT_W for uniform indexing)
    wire [N*OUT_W-1:0] level_bus [0:LEVELS];
    reg  [N*OUT_W-1:0] level_pipe [0:LEVELS];
    reg                level_valid_pipe [0:LEVELS];

    assign level_bus[0] = stage0;

    genvar l;
    generate
        for (l = 0; l < LEVELS; l = l + 1) begin : gen_level
            localparam COUNT_IN  = N >> l;
            localparam COUNT_OUT = N >> (l+1);
            localparam W_CUR     = IN_W + l;
            localparam W_NEXT    = IN_W + l + 1;

            wire [N*OUT_W-1:0] src;
            if (PIPE && l > 0) begin : gen_src_pipe
                assign src = level_pipe[l];
            end else begin : gen_src_comb
                assign src = level_bus[l];
            end

            genvar j;
            for (j = 0; j < COUNT_OUT; j = j + 1) begin : gen_add
                wire signed [W_CUR-1:0]  a, b;
                wire signed [W_NEXT-1:0] sum;
                assign a   = src[(2*j)*OUT_W   +: W_CUR];
                assign b   = src[(2*j+1)*OUT_W +: W_CUR];
                assign sum = a + b;
                if (PIPE) begin : gen_r
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n)
                            level_pipe[l+1][j*OUT_W +: OUT_W] <= '0;
                        else if (level_valid_pipe[l])
                            level_pipe[l+1][j*OUT_W +: OUT_W] <=
                                {{(OUT_W - W_NEXT){1'b0}}, sum};
                    end
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        level_pipe[l+1][j*OUT_W +: OUT_W];
                end else begin : gen_c
                    assign level_bus[l+1][j*OUT_W +: OUT_W] =
                        {{(OUT_W - W_NEXT){1'b0}}, sum};
                end
            end

            // zero-pad unused upper slots
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

    // Valid pipeline (LEVELS registers deep for PIPE=1)
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

    assign data_out  = level_bus[LEVELS][OUT_W-1:0];
    assign valid_out = PIPE ? level_valid_pipe[LEVELS] : valid_in;

endmodule
