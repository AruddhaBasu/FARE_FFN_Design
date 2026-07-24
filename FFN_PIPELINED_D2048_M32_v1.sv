//============================================================================
//  Common parameters and macros for the Pipelined FFN
//============================================================================
// Default parameters:
//   D        = 256   : Input/output vector dimension
//   M        = 16    : Tile dimension
//   DATA_W   = 16    : Fixed-point data width (signed)
//   FRAC_W   = 8     : Fractional bits in fixed-point representation
//   AXI_DATA_W = 128 : AXI data bus width (must be >= DATA_W)
//   ADDR_W   = 32    : AXI address width
//
// Derived:
//   NUM_TILES_D  = D/M     : Number of tiles along D dimension
//   NUM_TILES_4D = 4*D/M   : Number of tiles along 4D dimension
//   PROD_W       = 2*DATA_W: Multiplier product width
//   ACC_UP_W     = PROD_W + $clog2(M) + 2 : Up-proj accumulator width
//   ACC_DOWN_W   = PROD_W + $clog2(M) + 2 : Down-proj accumulator width
//============================================================================

// Macro to index into a flat 1×M tile vector
// tile[k] = tile_flat[k*DATA_W +: DATA_W]
`define TILE_IDX(flat, k, DW) flat[k*DW +: DW]

// Macro to index into a flat M×M weight tile vector
// weight[k][j] = weight_flat[(k*M+j)*DW +: DW]
`define WEIGHT_IDX(flat, k, j, M, DW) flat[(k*(M)+(j))*(DW) +: (DW)]

//============================================================================
//  Simplified AXI4 Read Master
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
    parameter AXI_DATA_W = 128,        // AXI data bus width
    parameter AXI_ADDR_W = 32           // AXI address bus width
)(
    input  wire                          clk,            // Reference clock signal
    input  wire                          rst_n,           // Reset signal 

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
    localparam S_IDLE    = 2'd0;    // Idle state
    localparam S_AR      = 2'd1;  // Driving AR channel
    localparam S_R       = 2'd2;  // Receiving R channel

    reg [1:0] state;              //state register

    always @(posedge clk or negedge rst_n) begin         
        if (!rst_n) begin            // When reset is applied
            state    <= S_IDLE;        // state = Idle state
            arvalid  <= 1'b0;         // Address valid signal is reset
            rready   <= 1'b0;        // Data ready signal is reset
            araddr   <= '0;            // Read address is reset to 0, from where the data reading would start
            arlen    <= '0;            // Address length is reset to 0
            arsize   <= '0;            // Address size is reset to 0
            arburst  <= 2'b01; // Increment burst length by 1
        end else begin
            case (state)
                S_IDLE: begin                // Idle state
                    arvalid <= 1'b0;            // Valid ready handshake signals are reset
                    rready  <= 1'b0;
                    if (req_valid) begin            // If valid request is found
                        araddr   <= req_addr;        // AR channel address = required address
                        arlen    <= req_len;          // AR channel length = required length
                        arsize   <= req_size;         //AR channel size = required size
                        arburst  <= 2'b01;             // Inrease AR channel burst by 1
                        arvalid  <= 1'b1;            // AR channel valid is set to 1
                        state    <= S_AR;            // Next state = AXI AR(address read) state
                    end
                end

                S_AR: begin                            // AXI address read state
                    if (arready && arvalid) begin      // If arready and arvalid is high   
                        arvalid <= 1'b0;                // Make arvalid = 0 to preserve the address
                        rready  <= 1'b1;                // rready = 1 to ensure the FSM is ready to read data
                        state   <= S_R;                // Next state = AXI R(data read) state
                    end
                end

                S_R: begin                            // AXI data read state
                    if (rvalid && rready) begin        // If rvalid and rready are both high
                        if (rlast) begin                // and rlast is high, idicating rdata is finished
                            rready <= 1'b0;                // rready is reset to 0, as the data has been read completely
                            state  <= S_IDLE;            // Next state = Idle state
                        end
                    end
                end

                default: state <= S_IDLE; // The default state is idle
            endcase
        end
    end

    // Request can be accepted only in IDLE
    assign req_ready = (state == S_IDLE);

    // Response data: directly from AXI R channel
    assign resp_data  = rdata;                        // Response data = rdata
    assign resp_last  = rlast;                        // Response last = rlast
    assign resp_valid = rvalid && (state == S_R);       // Response valid is high only when rvalid is high and the state is AXI data read state

endmodule

//============================================================================
// Simple Dual-Port BRAM (1 write port, 1 read port)
//============================================================================
// Port A: Write only
// Port B: Read only (synchronous read, 1-cycle latency)
//
// Parameters:
//   DATA_W : Bit width of each word
//   DEPTH  : Number of words
//   ADDR_W : Address width ($clog2(DEPTH))
//============================================================================

module bram_dp #(
    parameter DATA_W = 16,
    parameter DEPTH  = 256,
    parameter ADDR_W = $clog2(DEPTH)
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
//  Wide-word Dual-Port BRAM for tile storage
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
    parameter ADDR_W    = $clog2(DEPTH)
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

//============================================================================
// Parameterized Pipelined Adder Tree (Efficient)
//============================================================================
// Reduces N input values to a single sum using a binary tree.
//
// Parameters:
//   N       : Number of inputs (must be a power of 2)
//   IN_W    : Input data width (signed)
//   PIPE    : 1 = insert pipeline register at every tree level
//             0 = purely combinational
//
// Output width = IN_W + $clog2(N)
//
// Efficient storage: only allocates (N >> l) entries at level l,
// each (IN_W + l) bits wide. This avoids the O(N*LEVELS) waste
// of the naïve flat-array approach — critical for N=32.
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

    localparam LEVELS = $clog2(N); // Number of adder tree stages

    // -------------------------------------------------------------------
    // Level 0: sign-extend inputs from IN_W to (IN_W + LEVELS)
    // -------------------------------------------------------------------
    // We use a flat bus: level0 has N words, each (IN_W+LEVELS) bits wide
    localparam FULL_W = IN_W + LEVELS;

    wire [N*FULL_W-1:0]  level0;
    reg  [N*FULL_W-1:0]  level0_p;   // pipelined version

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_signext_l0
            assign level0[i*FULL_W +: FULL_W] =
                {{LEVELS{data_in[i*IN_W + IN_W - 1]}}, data_in[i*IN_W +: IN_W]};
        end
    endgenerate

    // -------------------------------------------------------------------
    // Build each level using a flat bus
    // At level l:  count = N >> l,  word_width = IN_W + l
    // At level l+1: count = N >> (l+1), word_width = IN_W + l + 1
    //
    // For efficiency we store only the active entries per level.
    // But since Verilog generate can't create variable-width 2D arrays,
    // we use flat buses with computed indices.
    // -------------------------------------------------------------------

    // We'll chain levels: level_l (combinational output) → level_l_p (pipeline reg) → next level input

    // For PIPE=0: wire chain (level0 → level1_comb → level2_comb → ... → result)
    // For PIPE=1: reg  chain (level0_p → level1_comb → level1_p → level2_comb → ...)

    // Implementation: create wires for each level output, and optional pipeline regs
    // We use a generate loop with intermediate buses.

    // Maximum bus size at each level: (N >> l) * FULL_W (padded to FULL_W per word)
    // This is wasteful but necessary for clean indexing in generate.
    // Total wire bits: Σ (N>>l) * FULL_W ≈ 2N * FULL_W — acceptable.

    // Level output buses (only first (N>>l)*FULL_W bits used at each level)
    wire [N*FULL_W-1:0] level_out [0:LEVELS];
    reg  [N*FULL_W-1:0] level_p   [0:LEVELS];  // pipeline registers
    reg                 level_valid_p [0:LEVELS];

    // Assign level 0
    assign level_out[0] = level0;

    // -------------------------------------------------------------------
    // Generate each tree level
    // -------------------------------------------------------------------
    genvar l;
    generate
        for (l = 0; l < LEVELS; l = l + 1) begin : gen_tree_level
            localparam COUNT_IN  = N >> l;
            localparam COUNT_OUT = N >> (l + 1);
            localparam W_IN      = IN_W + l;
            localparam W_OUT     = IN_W + l + 1;

            // Select input source: pipelined register or combinational wire
            wire [N*FULL_W-1:0] src;
            assign src = (PIPE && l > 0) ? level_p[l] : level_out[l];

            // Generate adder pairs for this level
            genvar j;
            for (j = 0; j < COUNT_OUT; j = j + 1) begin : gen_pair
                wire signed [W_IN-1:0] a, b;
                wire signed [W_OUT-1:0] sum;

                assign a   = src[(2*j)*FULL_W +: W_IN];
                assign b   = src[(2*j+1)*FULL_W +: W_IN];
                assign sum = $signed(a) + $signed(b);

                if (PIPE) begin : gen_pipe_reg
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n)
                            level_p[l+1][j*FULL_W +: FULL_W] <= '0;
                        else if (level_valid_p[l])
                            level_p[l+1][j*FULL_W +: FULL_W] <= {{(FULL_W-W_OUT){1'b0}}, sum};
                    end
                    assign level_out[l+1][j*FULL_W +: FULL_W] = level_p[l+1][j*FULL_W +: FULL_W];
                end else begin : gen_comb_only
                    assign level_out[l+1][j*FULL_W +: FULL_W] = {{(FULL_W-W_OUT){1'b0}}, sum};
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------
    // Valid signal propagation
    // -------------------------------------------------------------------
    // For PIPE=1: valid takes LEVELS clock cycles to propagate through
    // the tree. We use a shift register of depth LEVELS.
    // For PIPE=0: valid passes through combinationally.
    // -------------------------------------------------------------------
    reg [LEVELS:0] valid_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_shift <= '0;
        else begin
            valid_shift[0] <= valid_in;
            valid_shift[LEVELS:1] <= valid_shift[LEVELS-1:0];
        end
    end

    // Pipeline register enables follow the same latency
    integer vp;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (vp = 0; vp <= LEVELS; vp = vp + 1)
                level_valid_p[vp] <= 1'b0;
        end else begin
            level_valid_p[0] <= valid_in;
            for (vp = 1; vp <= LEVELS; vp = vp + 1)
                level_valid_p[vp] <= level_valid_p[vp-1];
        end
    end

    // -------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------
    localparam OUT_W = IN_W + LEVELS;
    assign data_out  = level_out[LEVELS][OUT_W-1:0];
    assign valid_out = PIPE ? valid_shift[LEVELS] : valid_in;

endmodule

//============================================================================
//  Stage 1: Fetch & Address Generation 
//============================================================================
// Two-phase operation:
//   UP_PHASE   : Fetches input tiles + up-projection weight tiles → Stage 2
//   DOWN_PHASE : Reads ReLU tiles from local BRAM + fetches down-projection
//                weight tiles from external memory → Stage 4
//============================================================================

module fetch_addr_gen #(
    parameter D          = 256,
    parameter M          = 16,
    parameter DATA_W     = 16,
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output reg                           done,
    output reg                           phase_up,
    output reg                           phase_down,

    // ---- AXI Read Master Interface ----
    output reg  [AXI_ADDR_W-1:0]        axi_req_addr,
    output reg  [7:0]                   axi_req_len,
    output reg  [2:0]                   axi_req_size,
    output reg                          axi_req_valid,
    input  wire                          axi_req_ready,
    input  wire [AXI_DATA_W-1:0]        axi_resp_data,
    input  wire                          axi_resp_last,
    input  wire                          axi_resp_valid,
    output wire                          axi_resp_ready,

    // ---- ReLU BRAM Read Interface ----
    output reg                           relu_rd_en,
    output reg  [$clog2(4*D/M)-1:0]     relu_rd_addr,
    input  wire [M*DATA_W-1:0]          relu_rd_data,

    // ---- To Up Projection Stage ----
    output reg  [M*DATA_W-1:0]          up_input_tile,
    output reg  [M*M*DATA_W-1:0]        up_weight_tile,
    output reg  [$clog2(4*D/M)-1:0]     up_tile_col,
    output reg  [$clog2(D/M)-1:0]       up_inner_idx,
    output reg                           up_is_last,
    output reg                           up_valid,
    input  wire                          up_ready,

    // ---- To Down Projection Stage ----
    output reg  [M*DATA_W-1:0]          down_relu_tile,
    output reg  [M*M*DATA_W-1:0]        down_weight_tile,
    output reg  [$clog2(D/M)-1:0]       down_tile_col,
    output reg  [$clog2(4*D/M)-1:0]     down_inner_idx,
    output reg                           down_is_last,
    output reg                           down_valid,
    input  wire                          down_ready
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;
    localparam WORDS_PER_BEAT = AXI_DATA_W / DATA_W;

    localparam INPUT_BASE   = 32'h0000_0000;
    localparam WUP_BASE     = 32'h1000_0000;
    localparam WDOWN_BASE   = 32'h2000_0000;
    localparam TILE_BYTES   = M * (DATA_W / 8);
    localparam WTILE_BYTES  = M * M * (DATA_W / 8);

    localparam INPUT_BEATS  = (M + WORDS_PER_BEAT - 1) / WORDS_PER_BEAT;
    localparam WEIGHT_BEATS = (M * M + WORDS_PER_BEAT - 1) / WORDS_PER_BEAT;

    // AXI size: log2(AXI_DATA_W/8)
    function [2:0] axi_size;
        input integer bytes;
        integer s;
        begin
            s = 0;
            while ((1 << s) < bytes) s = s + 1;
            axi_size = s[2:0];
        end
    endfunction

    localparam AXI_SZ = axi_size(AXI_DATA_W / 8);

    // -------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------
    localparam S_IDLE           = 4'd0;        // Idle state
    localparam S_UP_REQ_INPUT   = 4'd1;        // Up projection input request state
    localparam S_UP_WAIT_INPUT  = 4'd2;        // Up projection wait state for input arrival
    localparam S_UP_REQ_WEIGHT  = 4'd3;        // Up projection request state for weight
    localparam S_UP_WAIT_WEIGHT = 4'd4;        // Up projection wait state for weight arrival
    localparam S_UP_SEND        = 4'd5;        // input tile and weight tile send state
    localparam S_DOWN_REQ_RELU  = 4'd6;        // Down projection input request state ( from ReLU BRAM)
    localparam S_DOWN_RELAY     = 4'd7;        // Wait for BRAM read latency
    localparam S_DOWN_WAIT_RELU = 4'd8;        //  Down projection wait state for input arrival ( from ReLU BRAM)
    localparam S_DOWN_REQ_WGT   = 4'd9;        // Down projection weight tile request state
    localparam S_DOWN_WAIT_WGT  = 4'd10;        // Down projection wait state for weight arrival
    localparam S_DOWN_SEND      = 4'd11;         // Input tile and down tile send state
    localparam S_DONE           = 4'd12;        // Done state

    reg [3:0] state;                            // State register

    // -------------------------------------------------------------------
    // Iteration counters
    // -------------------------------------------------------------------
    reg [$clog2(NUM_TILES_4D)-1:0] tile_col;
    reg [$clog2(NUM_TILES_D)-1:0]  inner_idx;
    reg [$clog2(NUM_TILES_4D)-1:0] inner_idx_wide;

    // -------------------------------------------------------------------
    // Tile data buffers
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0]     input_tile_buf;
    reg [M*M*DATA_W-1:0]   weight_tile_buf;
    reg [M*DATA_W-1:0]     relu_tile_buf;
    reg [$clog2(M*M)-1:0]  beat_cnt;

    integer k;

    // -------------------------------------------------------------------
    // axi_resp_ready: combinational — high when in a wait state
    // -------------------------------------------------------------------
    assign axi_resp_ready = (state == S_UP_WAIT_INPUT) ||
                            (state == S_UP_WAIT_WEIGHT) ||
                            (state == S_DOWN_WAIT_WGT);

    // -------------------------------------------------------------------
    // Main state machine
    // The pattern is:
    //   Cycle N  : Set req_addr/len/size AND state <= REQ_STATE
    //   Cycle N+1: Request signals are now stable; assert req_valid
    //   Cycle N+2: AXI master sees valid=1 with correct data
    // -------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            phase_up        <= 1'b0;
            phase_down      <= 1'b0;
            tile_col        <= '0;
            inner_idx       <= '0;
            inner_idx_wide  <= '0;
            input_tile_buf  <= '0;
            weight_tile_buf <= '0;
            relu_tile_buf   <= '0;
            beat_cnt        <= '0;
            axi_req_addr    <= '0;
            axi_req_len     <= '0;
            axi_req_size    <= '0;
            axi_req_valid   <= 1'b0;
            up_input_tile   <= '0;
            up_weight_tile  <= '0;
            up_tile_col     <= '0;
            up_inner_idx    <= '0;
            up_is_last      <= 1'b0;
            up_valid        <= 1'b0;
            down_relu_tile  <= '0;
            down_weight_tile<= '0;
            down_tile_col   <= '0;
            down_inner_idx  <= '0;
            down_is_last    <= 1'b0;
            down_valid      <= 1'b0;
            relu_rd_en      <= 1'b0;
            relu_rd_addr    <= '0;
        end else begin
            // Default: deassert one-shot signals
            up_valid     <= 1'b0;
            down_valid   <= 1'b0;
            relu_rd_en   <= 1'b0;
            axi_req_valid <= 1'b0;  // Default: deassert valid each cycle

            case (state)
                // =====================================================
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        phase_up   <= 1'b1;
                        phase_down <= 1'b0;
                        tile_col   <= '0;
                        inner_idx  <= '0;
                        // Pre-set request signals for first input request
                        axi_req_addr  <= INPUT_BASE;
                        axi_req_len   <= INPUT_BEATS - 1;
                        axi_req_size  <= AXI_SZ;
                        axi_req_valid <= 1'b1;  // Will be stable next cycle
                        state         <= S_UP_REQ_INPUT;
                    end
                end

                // =====================================================
                // UP PHASE: Request input tile
                // =====================================================
                S_UP_REQ_INPUT: begin
                    // Request signals were set in previous cycle; keep asserting valid
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_UP_WAIT_INPUT;
                    end
                end

                S_UP_WAIT_INPUT: begin
                    if (axi_resp_valid) begin
                        for (k = 0; k < WORDS_PER_BEAT; k = k + 1) begin
                            if (beat_cnt * WORDS_PER_BEAT + k < M) begin
                                input_tile_buf[(beat_cnt * WORDS_PER_BEAT + k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                            end
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            // Pre-set request signals for weight request
                            axi_req_addr  <= WUP_BASE +
                                             (inner_idx * NUM_TILES_4D + tile_col) * WTILE_BYTES;
                            axi_req_len   <= WEIGHT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_UP_REQ_WEIGHT;
                        end
                    end
                end

                // =====================================================
                // UP PHASE: Request weight tile
                // =====================================================
                S_UP_REQ_WEIGHT: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_UP_WAIT_WEIGHT;
                    end
                end

                S_UP_WAIT_WEIGHT: begin
                    if (axi_resp_valid) begin
                        for (k = 0; k < WORDS_PER_BEAT; k = k + 1) begin
                            if (beat_cnt * WORDS_PER_BEAT + k < M * M) begin
                                weight_tile_buf[(beat_cnt * WORDS_PER_BEAT + k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                            end
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            state <= S_UP_SEND;
                        end
                    end
                end

                // =====================================================
                // UP PHASE: Send tile pair to up projection
                // =====================================================
                S_UP_SEND: begin
                    if (up_ready) begin
                        up_input_tile   <= input_tile_buf;
                        up_weight_tile  <= weight_tile_buf;
                        up_tile_col     <= tile_col[$clog2(NUM_TILES_4D)-1:0];
                        up_inner_idx    <= inner_idx;
                        up_is_last      <= (inner_idx == NUM_TILES_D - 1);
                        up_valid        <= 1'b1;

                        // Advance counters and pre-set next request
                        if (inner_idx == NUM_TILES_D - 1) begin
                            if (tile_col == NUM_TILES_4D - 1) begin
                                phase_up   <= 1'b0;
                                phase_down <= 1'b1;
                                tile_col   <= '0;
                                inner_idx_wide <= '0;
                                state <= S_DOWN_REQ_RELU;
                            end else begin
                                tile_col   <= tile_col + 1;
                                inner_idx  <= '0;
                                // Pre-set next input request
                                axi_req_addr  <= INPUT_BASE;
                                axi_req_len   <= INPUT_BEATS - 1;
                                axi_req_size  <= AXI_SZ;
                                axi_req_valid <= 1'b1;
                                state         <= S_UP_REQ_INPUT;
                            end
                        end else begin
                            inner_idx <= inner_idx + 1;
                            // Pre-set next input request
                            axi_req_addr  <= INPUT_BASE + (inner_idx + 1) * TILE_BYTES;
                            axi_req_len   <= INPUT_BEATS - 1;
                            axi_req_size  <= AXI_SZ;
                            axi_req_valid <= 1'b1;
                            state         <= S_UP_REQ_INPUT;
                        end
                    end
                end

                // =====================================================
                // DOWN PHASE: Read ReLU tile from local BRAM
                // =====================================================
                S_DOWN_REQ_RELU: begin
                    relu_rd_addr <= inner_idx_wide[$clog2(NUM_TILES_4D)-1:0];
                    relu_rd_en   <= 1'b1;
                    state        <= S_DOWN_RELAY;
                end

                S_DOWN_RELAY: begin
                    // Wait one cycle for BRAM synchronous read to complete.
                    // rd_en was asserted in S_DOWN_REQ_RELU (takes effect at
                    // end of that cycle).  The BRAM sees rd_en at the NEXT
                    // posedge (this cycle) and latches the data into rd_reg.
                    // The data will be available in S_DOWN_WAIT_RELU.
                    state <= S_DOWN_WAIT_RELU;
                end

                S_DOWN_WAIT_RELU: begin
                    relu_tile_buf <= relu_rd_data;
                    // Pre-set down weight request
                    axi_req_addr  <= WDOWN_BASE +
                                    (inner_idx_wide * NUM_TILES_D + tile_col) * WTILE_BYTES;
                    axi_req_len   <= WEIGHT_BEATS - 1;
                    axi_req_size  <= AXI_SZ;
                    axi_req_valid <= 1'b1;
                    beat_cnt      <= '0;
                    state         <= S_DOWN_REQ_WGT;
                end

                // =====================================================
                // DOWN PHASE: Request down-projection weight tile
                // =====================================================
                S_DOWN_REQ_WGT: begin
                    axi_req_valid <= 1'b1;
                    if (axi_req_ready) begin
                        axi_req_valid <= 1'b0;
                        beat_cnt      <= '0;
                        state         <= S_DOWN_WAIT_WGT;
                    end
                end

                S_DOWN_WAIT_WGT: begin
                    if (axi_resp_valid) begin
                        for (k = 0; k < WORDS_PER_BEAT; k = k + 1) begin
                            if (beat_cnt * WORDS_PER_BEAT + k < M * M) begin
                                weight_tile_buf[(beat_cnt * WORDS_PER_BEAT + k)*DATA_W +: DATA_W]
                                    <= axi_resp_data[k*DATA_W +: DATA_W];
                            end
                        end
                        beat_cnt <= beat_cnt + 1;
                        if (axi_resp_last) begin
                            state <= S_DOWN_SEND;
                        end
                    end
                end

                // =====================================================
                // DOWN PHASE: Send tile pair to down projection
                // =====================================================
                S_DOWN_SEND: begin
                    if (down_ready) begin
                        down_relu_tile   <= relu_tile_buf;
                        down_weight_tile <= weight_tile_buf;
                        down_tile_col    <= tile_col[$clog2(NUM_TILES_D)-1:0];
                        down_inner_idx   <= inner_idx_wide[$clog2(NUM_TILES_4D)-1:0];
                        down_is_last     <= (inner_idx_wide == NUM_TILES_4D - 1);
                        down_valid       <= 1'b1;

                        if (inner_idx_wide == NUM_TILES_4D - 1) begin
                            if (tile_col == NUM_TILES_D - 1) begin
                                phase_down <= 1'b0;
                                done       <= 1'b1;
                                state      <= S_DONE;
                            end else begin
                                tile_col       <= tile_col + 1;
                                inner_idx_wide <= '0;
                                state          <= S_DOWN_REQ_RELU;
                            end
                        end else begin
                            inner_idx_wide <= inner_idx_wide + 1;
                            state          <= S_DOWN_REQ_RELU;
                        end
                    end
                end

                // =====================================================
                S_DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

//============================================================================
//  Stage 2: Up Projection (d × 4d)
//============================================================================
// Computes one partial product per cycle:
//   partial[j] = Sum of {k=0}^{M-1} input_tile[k] * weight_tile[k][j], j=0..M-1
//
// Hardware:
//   - M×M parallel signed multipliers (one per (k,j) pair)
//   - M parallel adder trees (one per output column j), each summing M products
//   - Internal accumulator: adds partial products across inner iterations
//     to produce the complete output tile for a given tile_col
//
// Pipeline adder trees (PIPE=1) for timing closure at large M.
// Pipeline depth = $clog2(M) + 1 cycles through the adder tree.
//
// Parameters:
//   D, M, DATA_W
//============================================================================

module up_projection #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Input from Fetch Stage ----
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [M*M*DATA_W-1:0]         weight_tile,
    input  wire [$clog2(4*D/M)-1:0]      tile_col,
    input  wire [$clog2(D/M)-1:0]        inner_idx,
    input  wire                           is_last,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- Output to ReLU Stage ----
    output reg  [M*DATA_W-1:0]           result_tile,
    output reg  [$clog2(4*D/M)-1:0]      result_col,
    output reg                            valid_out,
    input  wire                           ready_in
);

    localparam PROD_W  = 2 * DATA_W;
    localparam SUM_W   = PROD_W + $clog2(M);
    localparam ACC_W   = SUM_W + $clog2(D/M) + 2;
    localparam MAX_POS = (1 << (DATA_W - 1)) - 1;
    localparam MIN_NEG = -(1 << (DATA_W - 1));

    integer j;
    reg signed [ACC_W-1:0] acc_val;

    // -------------------------------------------------------------------
    // M×M Parallel Multipliers
    // -------------------------------------------------------------------
    wire [M*M*PROD_W-1:0] products;

    genvar gk, gj;
    generate
        for (gk = 0; gk < M; gk = gk + 1) begin : gen_mul_row
            for (gj = 0; gj < M; gj = gj + 1) begin : gen_mul_col
                wire signed [DATA_W-1:0] a;
                wire signed [DATA_W-1:0] b;
                wire signed [PROD_W-1:0] p;

                assign a = input_tile[gk*DATA_W +: DATA_W];
                assign b = weight_tile[(gk*M+gj)*DATA_W +: DATA_W];
                assign p = a * b;

                assign products[(gk*M+gj)*PROD_W +: PROD_W] = p;
            end
        end
    endgenerate

    // -------------------------------------------------------------------
    // M Adder Trees (pipelined for timing closure)
    // -------------------------------------------------------------------
    wire [M*SUM_W-1:0]  adder_tree_results;
    wire                 adder_tree_valid;

    generate
        for (gj = 0; gj < M; gj = gj + 1) begin : gen_adder_tree
            wire [M*PROD_W-1:0] tree_input;

            for (gk = 0; gk < M; gk = gk + 1) begin : gen_gather
                assign tree_input[gk*PROD_W +: PROD_W] =
                    products[(gk*M+gj)*PROD_W +: PROD_W];
            end

            adder_tree #(
                .N     (M),
                .IN_W  (PROD_W),
                .PIPE  (1)     //  PIPELINED for M=32 timing closure 
            ) u_adder_tree (
                .clk       (clk),
                .rst_n     (rst_n),
                .data_in   (tree_input),
                .valid_in  (valid_in),
                .data_out  (adder_tree_results[gj*SUM_W +: SUM_W]),
                .valid_out ()
            );
        end
    endgenerate

    // -------------------------------------------------------------------
    // Pipeline valid through adder tree
    // -------------------------------------------------------------------
    localparam TREE_DEPTH = $clog2(M);
    reg [TREE_DEPTH:0] valid_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= '0;
        else begin
            valid_pipe[0] <= valid_in & ready_out;
            valid_pipe[TREE_DEPTH:1] <= valid_pipe[TREE_DEPTH-1:0];
        end
    end

    // -------------------------------------------------------------------
    // Internal Accumulator
    // -------------------------------------------------------------------
    reg [M*ACC_W-1:0]    acc;
    reg                  acc_valid;
    reg [$clog2(4*D/M)-1:0] acc_col;

    // Pipeline inner_idx and is_last through the adder tree
    reg [$clog2(D/M)-1:0]    inner_idx_pipe [0:TREE_DEPTH];
    reg                       is_last_pipe   [0:TREE_DEPTH];
    reg [$clog2(4*D/M)-1:0]  tile_col_pipe  [0:TREE_DEPTH];

    integer p;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (p = 0; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= '0;
                is_last_pipe[p]   <= '0;
                tile_col_pipe[p]  <= '0;
            end
        end else begin
            inner_idx_pipe[0] <= inner_idx;
            is_last_pipe[0]   <= is_last;
            tile_col_pipe[0]  <= tile_col;
            for (p = 1; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= inner_idx_pipe[p-1];
                is_last_pipe[p]   <= is_last_pipe[p-1];
                tile_col_pipe[p]  <= tile_col_pipe[p-1];
            end
        end
    end

    // -------------------------------------------------------------------
    // Accumulation logic (occurs after adder tree pipeline)
    // -------------------------------------------------------------------
    wire tree_out_valid = valid_pipe[TREE_DEPTH];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            acc         <= '0;
            acc_valid   <= 1'b0;
            acc_col     <= '0;
            result_tile <= '0;
            result_col  <= '0;
        end else begin
            valid_out <= 1'b0;

            // Accept accumulated results from adder tree pipeline
            if (tree_out_valid) begin
                for (j = 0; j < M; j = j + 1) begin
                    if (inner_idx_pipe[TREE_DEPTH] == 0) begin
                        acc[j*ACC_W +: ACC_W] <=
                            {{(ACC_W-SUM_W){adder_tree_results[j*SUM_W + SUM_W - 1]}},
                             adder_tree_results[j*SUM_W +: SUM_W]};
                    end else begin
                        acc[j*ACC_W +: ACC_W] <=
                            $signed(acc[j*ACC_W +: ACC_W]) +
                            $signed({{(ACC_W-SUM_W){adder_tree_results[j*SUM_W + SUM_W - 1]}},
                                     adder_tree_results[j*SUM_W +: SUM_W]});
                    end
                end
                acc_col <= tile_col_pipe[TREE_DEPTH];

                if (is_last_pipe[TREE_DEPTH]) begin
                    acc_valid <= 1'b1;
                end
            end

            // Output to ReLU stage (with saturating truncation)
            if (acc_valid && ready_in) begin
                for (j = 0; j < M; j = j + 1) begin
                    acc_val = $signed(acc[j*ACC_W +: ACC_W]);
                    if (acc_val > MAX_POS)
                        result_tile[j*DATA_W +: DATA_W] <= MAX_POS[DATA_W-1:0];
                    else if (acc_val < MIN_NEG)
                        result_tile[j*DATA_W +: DATA_W] <= MIN_NEG[DATA_W-1:0];
                    else
                        result_tile[j*DATA_W +: DATA_W] <= acc[j*ACC_W +: DATA_W];
                end
                result_col <= acc_col;
                valid_out  <= 1'b1;
                acc_valid  <= 1'b0;
            end

            // Backpressure: stall input when accumulator is waiting to output
            ready_out <= !(acc_valid && !ready_in);
        end
    end

endmodule

//============================================================================
// Stage 3: ReLU Activation
//============================================================================
// Applies ReLU (max(0, x)) element-wise to each element of the 1×M tile.
// All M elements are processed in parallel using M comparators.
//
// The ReLU output is written to a local BRAM (relu_bram) so the DOWN
// phase can read it later. Simultaneously, it passes data through
// boundary registers for pipelined operation.
//
// Parameters:
//   D, M, DATA_W
//============================================================================

module relu_stage #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Input from Up Projection Stage ----
    input  wire [M*DATA_W-1:0]           input_tile,
    input  wire [$clog2(4*D/M)-1:0]      tile_col,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- ReLU BRAM Write Interface ----
    output reg                           relu_wr_en,
    output reg  [$clog2(4*D/M)-1:0]     relu_wr_addr,
    output reg  [M*DATA_W-1:0]          relu_wr_data,

    // ---- Output (pass-through for pipeline continuity) ----
    output reg  [M*DATA_W-1:0]          result_tile,
    output reg  [$clog2(4*D/M)-1:0]     result_col,
    output reg                           valid_out,
    input  wire                          ready_in
);

    // -------------------------------------------------------------------
    // ReLU: max(0, x) for each element — all M in parallel
    // -------------------------------------------------------------------
    reg [M*DATA_W-1:0] relu_output;

    integer j;
    always @(*) begin
        for (j = 0; j < M; j = j + 1) begin
            if (input_tile[j*DATA_W + DATA_W - 1]) begin
                // Negative → output 0
                relu_output[j*DATA_W +: DATA_W] = {DATA_W{1'b0}};
            end else begin
                // Non-negative → pass through
                relu_output[j*DATA_W +: DATA_W] = input_tile[j*DATA_W +: DATA_W];
            end
        end
    end

    // -------------------------------------------------------------------
    // Pipeline register + BRAM write
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            result_tile <= '0;
            result_col  <= '0;
            relu_wr_en  <= 1'b0;
            relu_wr_addr<= '0;
            relu_wr_data<= '0;
        end else begin
            // Default: deassert one-shot signals
            valid_out  <= 1'b0;
            relu_wr_en <= 1'b0;

            if (valid_in && ready_out) begin
                // Latch ReLU output into boundary register
                result_tile <= relu_output;
                result_col  <= tile_col;
                valid_out   <= 1'b1;

                // Write to ReLU BRAM simultaneously
                relu_wr_en   <= 1'b1;
                relu_wr_addr <= tile_col;
                relu_wr_data <= relu_output;
            end

            // Backpressure handling
            ready_out <= 1'b1;  // ReLU is always ready (combinational, no backpressure needed internally)
        end
    end

endmodule

//============================================================================
//  Stage 4: Down Projection (4d × d)
//============================================================================
// Same structure as up_projection but:
//   - Inner loop: NUM_TILES_4D iterations
//   - Output columns: NUM_TILES_D
//   - Wider accumulator for the larger inner loop
//   - Pipelined adder trees (PIPE=1) for timing closure
//============================================================================

module down_projection #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire [M*DATA_W-1:0]           relu_tile,
    input  wire [M*M*DATA_W-1:0]         weight_tile,
    input  wire [$clog2(D/M)-1:0]        tile_col,
    input  wire [$clog2(4*D/M)-1:0]      inner_idx,
    input  wire                           is_last,
    input  wire                           valid_in,
    output reg                            ready_out,

    output reg  [M*DATA_W-1:0]           result_tile,
    output reg  [$clog2(D/M)-1:0]        result_col,
    output reg                            valid_out,
    input  wire                           ready_in
);

    localparam PROD_W  = 2 * DATA_W;
    localparam SUM_W   = PROD_W + $clog2(M);
    localparam ACC_W   = SUM_W + $clog2(4*D/M) + 2;
    localparam MAX_POS = (1 << (DATA_W - 1)) - 1;
    localparam MIN_NEG = -(1 << (DATA_W - 1));

    integer j;
    reg signed [ACC_W-1:0] acc_val;

    // -------------------------------------------------------------------
    // M×M Parallel Multipliers
    // -------------------------------------------------------------------
    wire [M*M*PROD_W-1:0] products;

    genvar gk, gj;
    generate
        for (gk = 0; gk < M; gk = gk + 1) begin : gen_mul_row
            for (gj = 0; gj < M; gj = gj + 1) begin : gen_mul_col
                wire signed [DATA_W-1:0] a;
                wire signed [DATA_W-1:0] b;
                wire signed [PROD_W-1:0] p;

                assign a = relu_tile[gk*DATA_W +: DATA_W];
                assign b = weight_tile[(gk*M+gj)*DATA_W +: DATA_W];
                assign p = a * b;

                assign products[(gk*M+gj)*PROD_W +: PROD_W] = p;
            end
        end
    endgenerate

    // -------------------------------------------------------------------
    // M Adder Trees (pipelined for timing closure)
    // -------------------------------------------------------------------
    wire [M*SUM_W-1:0] adder_tree_results;

    generate
        for (gj = 0; gj < M; gj = gj + 1) begin : gen_adder_tree
            wire [M*PROD_W-1:0] tree_input;

            for (gk = 0; gk < M; gk = gk + 1) begin : gen_gather
                assign tree_input[gk*PROD_W +: PROD_W] =
                    products[(gk*M+gj)*PROD_W +: PROD_W];
            end

            adder_tree #(
                .N     (M),
                .IN_W  (PROD_W),
                .PIPE  (1)     // *** PIPELINED for M=32 timing closure ***
            ) u_adder_tree (
                .clk       (clk),
                .rst_n     (rst_n),
                .data_in   (tree_input),
                .valid_in  (valid_in),
                .data_out  (adder_tree_results[gj*SUM_W +: SUM_W]),
                .valid_out ()
            );
        end
    endgenerate

    // -------------------------------------------------------------------
    // Pipeline valid through adder tree
    // -------------------------------------------------------------------
    localparam TREE_DEPTH = $clog2(M);
    reg [TREE_DEPTH:0] valid_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= '0;
        else begin
            valid_pipe[0] <= valid_in & ready_out;
            valid_pipe[TREE_DEPTH:1] <= valid_pipe[TREE_DEPTH-1:0];
        end
    end

    // -------------------------------------------------------------------
    // Pipeline control signals through adder tree
    // -------------------------------------------------------------------
    reg [$clog2(4*D/M)-1:0] inner_idx_pipe [0:TREE_DEPTH];
    reg                       is_last_pipe   [0:TREE_DEPTH];
    reg [$clog2(D/M)-1:0]    tile_col_pipe  [0:TREE_DEPTH];

    integer p;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (p = 0; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= '0;
                is_last_pipe[p]   <= '0;
                tile_col_pipe[p]  <= '0;
            end
        end else begin
            inner_idx_pipe[0] <= inner_idx;
            is_last_pipe[0]   <= is_last;
            tile_col_pipe[0]  <= tile_col;
            for (p = 1; p <= TREE_DEPTH; p = p + 1) begin
                inner_idx_pipe[p] <= inner_idx_pipe[p-1];
                is_last_pipe[p]   <= is_last_pipe[p-1];
                tile_col_pipe[p]  <= tile_col_pipe[p-1];
            end
        end
    end

    // -------------------------------------------------------------------
    // Internal Accumulator
    // -------------------------------------------------------------------
    reg [M*ACC_W-1:0]        acc;
    reg                      acc_valid;
    reg [$clog2(D/M)-1:0]    acc_col;

    wire tree_out_valid = valid_pipe[TREE_DEPTH];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out   <= 1'b1;
            valid_out   <= 1'b0;
            acc         <= '0;
            acc_valid   <= 1'b0;
            acc_col     <= '0;
            result_tile <= '0;
            result_col  <= '0;
        end else begin
            valid_out <= 1'b0;

            if (tree_out_valid) begin
                for (j = 0; j < M; j = j + 1) begin
                    if (inner_idx_pipe[TREE_DEPTH] == 0) begin
                        acc[j*ACC_W +: ACC_W] <=
                            {{(ACC_W-SUM_W){adder_tree_results[j*SUM_W + SUM_W - 1]}},
                             adder_tree_results[j*SUM_W +: SUM_W]};
                    end else begin
                        acc[j*ACC_W +: ACC_W] <=
                            $signed(acc[j*ACC_W +: ACC_W]) +
                            $signed({{(ACC_W-SUM_W){adder_tree_results[j*SUM_W + SUM_W - 1]}},
                                     adder_tree_results[j*SUM_W +: SUM_W]});
                    end
                end
                acc_col <= tile_col_pipe[TREE_DEPTH];

                if (is_last_pipe[TREE_DEPTH]) begin
                    acc_valid <= 1'b1;
                end
            end

            if (acc_valid && ready_in) begin
                for (j = 0; j < M; j = j + 1) begin
                    acc_val = $signed(acc[j*ACC_W +: ACC_W]);
                    if (acc_val > MAX_POS)
                        result_tile[j*DATA_W +: DATA_W] <= MAX_POS[DATA_W-1:0];
                    else if (acc_val < MIN_NEG)
                        result_tile[j*DATA_W +: DATA_W] <= MIN_NEG[DATA_W-1:0];
                    else
                        result_tile[j*DATA_W +: DATA_W] <= acc[j*ACC_W +: DATA_W];
                end
                result_col <= acc_col;
                valid_out  <= 1'b1;
                acc_valid  <= 1'b0;
            end

            ready_out <= !(acc_valid && !ready_in);
        end
    end

endmodule

//============================================================================
//  Stage 5: Output Accumulation 
//============================================================================
// Receives completed 1×M tiles from the Down Projection stage.
// Decodes the tile_col index to determine where in the 1×D output
// vector each tile should be written.
//
// After all tiles are written, a readout FSM streams them out through
// output_tile_data / output_tile_addr / output_tile_valid with proper
// BRAM latency alignment.
//
// Timing analysis (critical path):
//
//   Because out_rd_en and out_rd_addr are registered outputs (non-blocking
//   in this always block), and the BRAM also has a registered read:
//
//   Cycle N   : FSM in RD_ISSUE  → sets out_rd_en<=1, out_rd_addr<=idx
//   Cycle N+1 : out_rd_en=1 reaches BRAM → BRAM latches mem[idx] into rd_reg
//   Cycle N+2 : out_rd_data = mem[idx]  → FSM in RD_CAPTURE reads it
//
//   Total: 2 cycles from RD_ISSUE to valid data.
//   FSM path: RD_ISSUE → RD_RELAY → RD_CAPTURE → (next RD_ISSUE or RD_DONE)
//
//============================================================================

module accumulator #(
    parameter D      = 256,
    parameter M      = 16,
    parameter DATA_W = 16
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output reg                           done,

    // ---- Input from Down Projection Stage ----
    input  wire [M*DATA_W-1:0]           result_tile,
    input  wire [$clog2(D/M)-1:0]        result_col,
    input  wire                           valid_in,
    output reg                            ready_out,

    // ---- Output BRAM Write Interface ----
    output reg                           out_wr_en,
    output reg  [$clog2(D/M)-1:0]       out_wr_addr,
    output reg  [M*DATA_W-1:0]          out_wr_data,

    // ---- Output BRAM Read Interface ----
    output reg                           out_rd_en,
    output reg  [$clog2(D/M)-1:0]       out_rd_addr,
    input  wire [M*DATA_W-1:0]          out_rd_data,

    // ---- Streaming output port (latency-aligned) ----
    output reg  [M*DATA_W-1:0]          output_tile_data,
    output reg  [$clog2(D/M)-1:0]       output_tile_addr,
    output reg                           output_tile_valid,

    // ---- Legacy single-element output (kept for compatibility) ----
    output reg  [DATA_W-1:0]            output_data,
    output reg                           output_valid
);

    localparam NUM_TILES = D / M;

    // -------------------------------------------------------------------
    // Write-side state
    // -------------------------------------------------------------------
    reg [$clog2(NUM_TILES)-1:0] tile_cnt;

    // -------------------------------------------------------------------
    // Readout FSM states
    // -------------------------------------------------------------------
    localparam RD_IDLE    = 3'd0;
    localparam RD_ISSUE   = 3'd1;  // Assert rd_en for current index (registered)
    localparam RD_RELAY   = 3'd2;  // Wait for registered rd_en to reach BRAM
    localparam RD_CAPTURE = 3'd3;  // BRAM data is now valid — capture & output
    localparam RD_DONE    = 3'd4;  // All tiles streamed out

    reg [2:0] rd_state;
    reg [$clog2(NUM_TILES)-1:0] rd_idx;
    reg [$clog2(NUM_TILES)-1:0] rd_idx_pipe;  // Pipelined address (aligns with data)
    reg        rd_done_flag;

    // -------------------------------------------------------------------
    // Main sequential logic
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready_out       <= 1'b1;
            done            <= 1'b0;
            tile_cnt        <= 0;
            out_wr_en       <= 1'b0;
            out_wr_addr     <= 0;
            out_wr_data     <= 0;
            out_rd_en       <= 1'b0;
            out_rd_addr     <= 0;
            output_tile_data  <= 0;
            output_tile_addr  <= 0;
            output_tile_valid <= 1'b0;
            output_data     <= 0;
            output_valid    <= 1'b0;
            rd_state        <= RD_IDLE;
            rd_idx          <= 0;
            rd_idx_pipe     <= 0;
            rd_done_flag    <= 1'b0;
        end else begin
            // Defaults: one-cycle pulses
            out_wr_en        <= 1'b0;
            out_rd_en        <= 1'b0;
            output_tile_valid <= 1'b0;
            output_valid     <= 1'b0;

            // -----------------------------------------------------------
            // Reset on new start
            // -----------------------------------------------------------
            if (start) begin
                done         <= 1'b0;
                tile_cnt     <= 0;
                rd_state     <= RD_IDLE;
                rd_done_flag <= 1'b0;
            end

            // -----------------------------------------------------------
            // Write path: accept result tile from down projection
            // -----------------------------------------------------------
            if (valid_in && ready_out) begin
                out_wr_en   <= 1'b1;
                out_wr_addr <= result_col;
                out_wr_data <= result_tile;

                tile_cnt <= tile_cnt + 1;

                if (tile_cnt == NUM_TILES - 1) begin
                    done <= 1'b1;
                end
            end

            // -----------------------------------------------------------
            // Readout FSM: stream output tiles after computation done
            // -----------------------------------------------------------
            case (rd_state)
                RD_IDLE: begin
                    if (done && !rd_done_flag) begin
                        rd_idx    <= 0;
                        rd_state  <= RD_ISSUE;
                    end
                end

                RD_ISSUE: begin
                    // Issue BRAM read for rd_idx.
                    // out_rd_en and out_rd_addr are non-blocking, so they
                    // take effect at the END of this cycle.
                    out_rd_en   <= 1'b1;
                    out_rd_addr <= rd_idx;
                    // Pipeline the address to align with data arrival
                    rd_idx_pipe <= rd_idx;
                    rd_state    <= RD_RELAY;
                end

                RD_RELAY: begin
                    // Cycle after RD_ISSUE: out_rd_en is now 1 at the BRAM
                    // input. The BRAM latches mem[rd_idx] into rd_reg at the
                    // end of THIS cycle. Data will be on out_rd_data next cycle.
                    // Deassert rd_en (already default 0).
                    rd_state <= RD_CAPTURE;
                end

                RD_CAPTURE: begin
                    // out_rd_data now holds mem[rd_idx_pipe].
                    // Capture and present on the output port.
                    output_tile_data  <= out_rd_data;
                    output_tile_addr  <= rd_idx_pipe;
                    output_tile_valid <= 1'b1;
                    // Legacy single-element output
                    output_data  <= out_rd_data[0*DATA_W +: DATA_W];
                    output_valid <= 1'b1;

                    if (rd_idx < NUM_TILES - 1) begin
                        rd_idx   <= rd_idx + 1;
                        rd_state <= RD_ISSUE;
                    end else begin
                        rd_done_flag <= 1'b1;
                        rd_state     <= RD_DONE;
                    end
                end

                RD_DONE: begin
                    // Stay here until next start resets us
                    rd_done_flag <= 1'b1;
                end

                default: rd_state <= RD_IDLE;
            endcase

            ready_out <= 1'b1;
        end
    end

endmodule


//============================================================================
// ffn_top.v — Top-Level Pipelined Feed-Forward Network
//============================================================================
// Integrates all five pipeline stages with valid/ready handshaking,
// AXI4 read interface to external memory, and local BRAMs for
// ReLU output buffering and final output storage.
//   ReLU BRAM connects Stage 3 output → Stage 4 input (via fetch stage reads)
//   Output BRAM stores the final 1×D result
//
// Pipeline Phases:
//   UP   phase: Stages 1→2→3 active; computes input × W_up, applies ReLU
//   DOWN phase: Stages 1→4→5 active; computes ReLU_out × W_down, accumulates
//
// Parameters:
//   D          : Input/output dimension
//   M          : Tile dimension
//   DATA_W     : Fixed-point data width
//   AXI_DATA_W : AXI data bus width
//   AXI_ADDR_W : AXI address width
//============================================================================

module ffn_top #(
    parameter D          = 256,
    parameter M          = 16,
    parameter DATA_W     = 16,
    parameter AXI_DATA_W = 128,
    parameter AXI_ADDR_W = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,          // Assert to begin computation
    output wire                          done,           // Computation complete

    // ---- AXI4 Read Master Interface ----
    output wire [AXI_ADDR_W-1:0]        araddr,
    output wire [7:0]                   arlen,
    output wire [2:0]                   arsize,
    output wire [1:0]                   arburst,
    output wire                          arvalid,
    input  wire                          arready,

    input  wire [AXI_DATA_W-1:0]        rdata,
    input  wire                          rlast,
    input  wire                          rvalid,
    output wire                          rready,

    // ---- Output Vector (1×D) ----
    output wire [M*DATA_W-1:0]          output_tile_data,
    output wire [$clog2(D/M)-1:0]       output_tile_addr,
    output wire                          output_tile_valid
);

    // -------------------------------------------------------------------
    // Derived parameters
    // -------------------------------------------------------------------
    localparam NUM_TILES_D  = D / M;
    localparam NUM_TILES_4D = 4 * D / M;

    // -------------------------------------------------------------------
    // Internal wires: Fetch → Up Projection
    // -------------------------------------------------------------------
    wire [M*DATA_W-1:0]           up_input_tile;
    wire [M*M*DATA_W-1:0]         up_weight_tile;
    wire [$clog2(NUM_TILES_4D)-1:0] up_tile_col;
    wire [$clog2(NUM_TILES_D)-1:0]  up_inner_idx;
    wire                           up_is_last;
    wire                           up_valid;
    wire                           up_ready;

    // -------------------------------------------------------------------
    // Internal wires: Up Projection → ReLU
    // -------------------------------------------------------------------
    wire [M*DATA_W-1:0]           relu_input_tile;
    wire [$clog2(NUM_TILES_4D)-1:0] relu_tile_col;
    wire                           relu_valid_in;
    wire                           relu_ready_out;

    // -------------------------------------------------------------------
    // Internal wires: ReLU → BRAM (write)
    // -------------------------------------------------------------------
    wire                           relu_wr_en;
    wire [$clog2(NUM_TILES_4D)-1:0] relu_wr_addr;
    wire [M*DATA_W-1:0]           relu_wr_data;

    // -------------------------------------------------------------------
    // Internal wires: ReLU BRAM → Fetch (read, in DOWN phase)
    // -------------------------------------------------------------------
    wire                           relu_rd_en;
    wire [$clog2(NUM_TILES_4D)-1:0] relu_rd_addr;
    wire [M*DATA_W-1:0]           relu_rd_data;

    // -------------------------------------------------------------------
    // Internal wires: Fetch → Down Projection
    // -------------------------------------------------------------------
    wire [M*DATA_W-1:0]           down_relu_tile;
    wire [M*M*DATA_W-1:0]         down_weight_tile;
    wire [$clog2(NUM_TILES_D)-1:0]  down_tile_col;
    wire [$clog2(NUM_TILES_4D)-1:0] down_inner_idx;
    wire                           down_is_last;
    wire                           down_valid;
    wire                           down_ready;

    // -------------------------------------------------------------------
    // Internal wires: Down Projection → Accumulator
    // -------------------------------------------------------------------
    wire [M*DATA_W-1:0]           acc_result_tile;
    wire [$clog2(NUM_TILES_D)-1:0]  acc_result_col;
    wire                           acc_valid_in;
    wire                           acc_ready_out;

    // -------------------------------------------------------------------
    // Internal wires: Accumulator → Output BRAM
    // -------------------------------------------------------------------
    wire                           out_wr_en;
    wire [$clog2(NUM_TILES_D)-1:0]  out_wr_addr;
    wire [M*DATA_W-1:0]           out_wr_data;
    wire                           out_rd_en;
    wire [$clog2(NUM_TILES_D)-1:0]  out_rd_addr;
    wire [M*DATA_W-1:0]           out_rd_data;

    // -------------------------------------------------------------------
    // Internal wires: AXI Read Master
    // -------------------------------------------------------------------
    wire [AXI_ADDR_W-1:0]        axi_req_addr;
    wire [7:0]                   axi_req_len;
    wire [2:0]                   axi_req_size;
    wire                          axi_req_valid;
    wire                          axi_req_ready;
    wire [AXI_DATA_W-1:0]        axi_resp_data;
    wire                          axi_resp_last;
    wire                          axi_resp_valid;
    wire                          axi_resp_ready;

    // -------------------------------------------------------------------
    // Phase signals from fetch stage
    // -------------------------------------------------------------------
    wire                          phase_up;
    wire                          phase_down;

    // ===================================================================
    // Module Instantiations
    // ===================================================================

    // -------------------------------------------------------------------
    // AXI Read Master
    // -------------------------------------------------------------------
    axi_read_master #(
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_axi_read_master (
        .clk        (clk),
        .rst_n      (rst_n),

        .req_addr   (axi_req_addr),
        .req_len    (axi_req_len),
        .req_size   (axi_req_size),
        .req_valid  (axi_req_valid),
        .req_ready  (axi_req_ready),

        .resp_data  (axi_resp_data),
        .resp_last  (axi_resp_last),
        .resp_valid (axi_resp_valid),
        .resp_ready (axi_resp_ready),

        .araddr     (araddr),
        .arlen      (arlen),
        .arsize     (arsize),
        .arburst    (arburst),
        .arvalid    (arvalid),
        .arready    (arready),

        .rdata      (rdata),
        .rlast      (rlast),
        .rvalid     (rvalid),
        .rready     (rready)
    );

    // -------------------------------------------------------------------
    // Stage 1: Fetch & Address Generation
    // -------------------------------------------------------------------
    fetch_addr_gen #(
        .D          (D),
        .M          (M),
        .DATA_W     (DATA_W),
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_fetch_addr_gen (
        .clk            (clk),
        .rst_n          (rst_n),

        .start          (start),
        .done           (),
        .phase_up       (phase_up),
        .phase_down     (phase_down),

        .axi_req_addr   (axi_req_addr),
        .axi_req_len    (axi_req_len),
        .axi_req_size   (axi_req_size),
        .axi_req_valid  (axi_req_valid),
        .axi_req_ready  (axi_req_ready),
        .axi_resp_data  (axi_resp_data),
        .axi_resp_last  (axi_resp_last),
        .axi_resp_valid (axi_resp_valid),
        .axi_resp_ready (axi_resp_ready),

        .relu_rd_en     (relu_rd_en),
        .relu_rd_addr   (relu_rd_addr),
        .relu_rd_data   (relu_rd_data),

        .up_input_tile  (up_input_tile),
        .up_weight_tile (up_weight_tile),
        .up_tile_col    (up_tile_col),
        .up_inner_idx   (up_inner_idx),
        .up_is_last     (up_is_last),
        .up_valid       (up_valid),
        .up_ready       (up_ready),

        .down_relu_tile   (down_relu_tile),
        .down_weight_tile (down_weight_tile),
        .down_tile_col    (down_tile_col),
        .down_inner_idx   (down_inner_idx),
        .down_is_last     (down_is_last),
        .down_valid       (down_valid),
        .down_ready       (down_ready)
    );

    // -------------------------------------------------------------------
    // Stage 2: Up Projection
    // -------------------------------------------------------------------
    up_projection #(
        .D      (D),
        .M      (M),
        .DATA_W (DATA_W)
    ) u_up_projection (
        .clk        (clk),
        .rst_n      (rst_n),

        .input_tile  (up_input_tile),
        .weight_tile (up_weight_tile),
        .tile_col    (up_tile_col),
        .inner_idx   (up_inner_idx),
        .is_last     (up_is_last),
        .valid_in    (up_valid),
        .ready_out   (up_ready),

        .result_tile (relu_input_tile),
        .result_col  (relu_tile_col),
        .valid_out   (relu_valid_in),
        .ready_in    (relu_ready_out)
    );

    // -------------------------------------------------------------------
    // Stage 3: ReLU
    // -------------------------------------------------------------------
    relu_stage #(
        .D      (D),
        .M      (M),
        .DATA_W (DATA_W)
    ) u_relu_stage (
        .clk        (clk),
        .rst_n      (rst_n),

        .input_tile  (relu_input_tile),
        .tile_col    (relu_tile_col),
        .valid_in    (relu_valid_in),
        .ready_out   (relu_ready_out),

        .relu_wr_en   (relu_wr_en),
        .relu_wr_addr (relu_wr_addr),
        .relu_wr_data (relu_wr_data),

        .result_tile (),
        .result_col  (),
        .valid_out   (),
        .ready_in    (1'b1)       // ReLU always pushes through
    );

    // -------------------------------------------------------------------
    // ReLU Output BRAM (stores 1×4D intermediate results)
    // -------------------------------------------------------------------
    bram_dp_wide #(
        .DATA_W    (DATA_W),
        .TILE_SIZE (M),
        .DEPTH     (NUM_TILES_4D)
    ) u_relu_bram (
        .clk      (clk),

        .wr_en    (relu_wr_en),
        .wr_addr  (relu_wr_addr),
        .wr_data  (relu_wr_data),

        .rd_en    (relu_rd_en),
        .rd_addr  (relu_rd_addr),
        .rd_data  (relu_rd_data)
    );

    // -------------------------------------------------------------------
    // Stage 4: Down Projection
    // -------------------------------------------------------------------
    down_projection #(
        .D      (D),
        .M      (M),
        .DATA_W (DATA_W)
    ) u_down_projection (
        .clk        (clk),
        .rst_n      (rst_n),

        .relu_tile   (down_relu_tile),
        .weight_tile (down_weight_tile),
        .tile_col    (down_tile_col),
        .inner_idx   (down_inner_idx),
        .is_last     (down_is_last),
        .valid_in    (down_valid),
        .ready_out   (down_ready),

        .result_tile (acc_result_tile),
        .result_col  (acc_result_col),
        .valid_out   (acc_valid_in),
        .ready_in    (acc_ready_out)
    );

    // -------------------------------------------------------------------
    // Stage 5: Accumulator
    // -------------------------------------------------------------------
    accumulator #(
        .D      (D),
        .M      (M),
        .DATA_W (DATA_W)
    ) u_accumulator (
        .clk        (clk),
        .rst_n      (rst_n),

        .start       (start),
        .done        (done),

        .result_tile (acc_result_tile),
        .result_col  (acc_result_col),
        .valid_in    (acc_valid_in),
        .ready_out   (acc_ready_out),

        .out_wr_en   (out_wr_en),
        .out_wr_addr (out_wr_addr),
        .out_wr_data (out_wr_data),

        .out_rd_en   (out_rd_en),
        .out_rd_addr (out_rd_addr),
        .out_rd_data (out_rd_data),

        .output_tile_data  (output_tile_data),
        .output_tile_addr  (output_tile_addr),
        .output_tile_valid (output_tile_valid),

        .output_data (),
        .output_valid()
    );

    // -------------------------------------------------------------------
    // Output BRAM (stores 1×D final results)
    // -------------------------------------------------------------------
    bram_dp_wide #(
        .DATA_W    (DATA_W),
        .TILE_SIZE (M),
        .DEPTH     (NUM_TILES_D)
    ) u_output_bram (
        .clk      (clk),

        .wr_en    (out_wr_en),
        .wr_addr  (out_wr_addr),
        .wr_data  (out_wr_data),

        .rd_en    (out_rd_en),
        .rd_addr  (out_rd_addr),
        .rd_data  (out_rd_data)
    );

    // -------------------------------------------------------------------
    // Output port — driven directly by accumulator 
    // -------------------------------------------------------------------
    // output_tile_data, output_tile_addr, output_tile_valid are now
    // registered outputs from the accumulator module, properly aligned
    // with BRAM read latency. No combinational assignments needed.

endmodule


//============================================================================
//  D=2048, M=32
//============================================================================
// Wrapper around ffn_top with  parameters.
//
// Key parameter choices:
//   D          = 2048    : Full input/output dimension
//   M          = 32      : Tile dimension (64 tiles along D)
//   DATA_W     = 16      : 16-bit fixed-point (Q8.8)
//   AXI_DATA_W = 512     : Wide AXI bus for throughput
//                         (1 beat per input tile, 32 beats per weight tile)
//   AXI_ADDR_W = 32      : 4 GB address space
//
// Resource estimates :
//   DSP48E2    : 2048  (32×32 multipliers × 2 projection stages)
//  DRAM size : ~64 MB (4KB input + 32MB W_up + 32MB W_down)
//============================================================================

module ffn_top_2048 (
    input  wire                          clk,
    input  wire                          rst_n,

    // ---- Control ----
    input  wire                          start,
    output wire                          done,

    // ---- AXI4 Read Master Interface (512-bit data bus) ----
    output wire [31:0]                   araddr,
    output wire [7:0]                   arlen,
    output wire [2:0]                   arsize,
    output wire [1:0]                   arburst,
    output wire                          arvalid,
    input  wire                          arready,

    input  wire [511:0]                 rdata,
    input  wire                          rlast,
    input  wire                          rvalid,
    output wire                          rready,

    // ---- Output Vector (1×2048, streamed as 64 tiles of 1×32) ----
    output wire [32*16-1:0]             output_tile_data,   // 512 bits
    output wire [5:0]                   output_tile_addr,   // 0..63
    output wire                          output_tile_valid
);

    ffn_top #(
        .D          (2048),
        .M          (32),
        .DATA_W     (16),
        .AXI_DATA_W (512),
        .AXI_ADDR_W (32)
    ) u_ffn (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .done           (done),

        .araddr         (araddr),
        .arlen          (arlen),
        .arsize         (arsize),
        .arburst        (arburst),
        .arvalid        (arvalid),
        .arready        (arready),

        .rdata          (rdata),
        .rlast          (rlast),
        .rvalid         (rvalid),
        .rready         (rready),

        .output_tile_data  (output_tile_data),
        .output_tile_addr  (output_tile_addr),
        .output_tile_valid (output_tile_valid)
    );

endmodule

