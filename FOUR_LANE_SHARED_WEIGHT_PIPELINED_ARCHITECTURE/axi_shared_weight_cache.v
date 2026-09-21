//=============================================================================
// axi_shared_weight_cache.v
//
// Functional shared AXI read arbiter/cache for the multi-lane wrapper.
// One lane owns the external AXI read channel at a time. Shared parameter and
// weight bursts are retained in a one-tile cache; an exact address/length hit
// is replayed locally. Private x/residual bursts are passed through.
//=============================================================================

module axi_shared_weight_cache #(
    parameter LANES      = 4,
    parameter AXI_DATA_W = 64,
    parameter AXI_ADDR_W = 32
)(
    input  wire [LANES*AXI_ADDR_W-1:0] lane_araddr,
    input  wire [LANES*8-1:0]          lane_arlen,
    input  wire [LANES*3-1:0]          lane_arsize,
    input  wire [LANES-1:0]            lane_arvalid,
    output reg  [LANES-1:0]            lane_arready,
    output reg  [LANES*AXI_DATA_W-1:0] lane_rdata,
    output reg  [LANES-1:0]            lane_rvalid,
    output reg  [LANES-1:0]            lane_rlast,
    input  wire [LANES-1:0]            lane_rready,
    input  wire [LANES-1:0]            active_mask,
    output reg  [AXI_ADDR_W-1:0]       araddr,
    output reg  [7:0]                  arlen,
    output reg  [2:0]                  arsize,
    output reg  [1:0]                  arburst,
    output reg                         arvalid,
    input  wire                         arready,
    input  wire [AXI_DATA_W-1:0]        rdata,
    input  wire                         rlast,
    input  wire                         rvalid,
    output reg                         rready,
    input  wire                         clk,
    input  wire                         rst_n
);

    localparam S_IDLE  = 2'd0;
    localparam S_EXT_R = 2'd1;
    localparam S_CACHE = 2'd2;

    reg [1:0] state;
    integer owner;
    integer selected;
    integer i;
    reg [7:0] beat_idx;
    reg [7:0] burst_len_r;
    reg [AXI_ADDR_W-1:0] burst_addr_r;
    reg [2:0] burst_size_r;
    reg shared_active;

    reg cache_valid;
    reg [AXI_ADDR_W-1:0] cache_addr;
    reg [7:0] cache_len;
    reg [2:0] cache_size;
    reg [AXI_DATA_W-1:0] cache_mem [0:255];

    function is_shared_address;
        input [AXI_ADDR_W-1:0] a;
        begin
            is_shared_address =
                ((a >= 32'h0C00_0000) && (a < 32'h0E00_0000)) ||
                ((a >= 32'h1000_0000) && (a < 32'h3200_0000));
        end
    endfunction

    reg selected_shared;
    reg selected_hit;
    reg owner_ready;

    always @(*) begin
        lane_arready = '0;
        lane_rdata   = '0;
        lane_rvalid  = '0;
        lane_rlast   = '0;
        araddr       = '0;
        arlen        = '0;
        arsize       = '0;
        arburst      = 2'b01;
        arvalid      = 1'b0;
        rready       = 1'b0;
        selected     = -1;
        selected_shared = 1'b0;
        selected_hit = 1'b0;
        owner_ready  = 1'b0;

        for (i = 0; i < LANES; i = i + 1) begin
            if ((selected < 0) && active_mask[i] && lane_arvalid[i])
                selected = i;
        end

        if (selected >= 0) begin
            selected_shared = is_shared_address(
                lane_araddr[selected*AXI_ADDR_W +: AXI_ADDR_W]);
            selected_hit = selected_shared && cache_valid &&
                (cache_addr == lane_araddr[selected*AXI_ADDR_W +: AXI_ADDR_W]) &&
                (cache_len  == lane_arlen[selected*8 +: 8]) &&
                (cache_size == lane_arsize[selected*3 +: 3]);
        end

        if (state == S_IDLE && selected >= 0) begin
            araddr = lane_araddr[selected*AXI_ADDR_W +: AXI_ADDR_W];
            arlen  = lane_arlen[selected*8 +: 8];
            arsize = lane_arsize[selected*3 +: 3];
            if (selected_hit) begin
                lane_arready[selected] = 1'b1;
            end else begin
                arvalid = 1'b1;
                lane_arready[selected] = arready;
            end
        end else if (state == S_EXT_R) begin
            lane_rdata[owner*AXI_DATA_W +: AXI_DATA_W] = rdata;
            lane_rvalid[owner] = rvalid;
            lane_rlast[owner]  = rlast;
            owner_ready = lane_rready[owner];
            rready = owner_ready;
        end else if (state == S_CACHE) begin
            lane_rdata[owner*AXI_DATA_W +: AXI_DATA_W] = cache_mem[beat_idx];
            lane_rvalid[owner] = 1'b1;
            lane_rlast[owner]  = (beat_idx == cache_len);
            owner_ready = lane_rready[owner];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            owner       <= 0;
            beat_idx    <= 0;
            burst_len_r <= 0;
            burst_addr_r<= 0;
            burst_size_r<= 0;
            shared_active <= 1'b0;
            cache_valid <= 1'b0;
            cache_addr  <= 0;
            cache_len   <= 0;
            cache_size  <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (selected >= 0) begin
                        if (selected_hit) begin
                            if (lane_arvalid[selected] && lane_arready[selected]) begin
                                owner    <= selected;
                                beat_idx <= 0;
                                state    <= S_CACHE;
                            end
                        end else if ((selected >= 0) && lane_arvalid[selected] && lane_arready[selected]) begin
                            owner        <= selected;
                            beat_idx     <= 0;
                            burst_len_r  <= arlen;
                            burst_addr_r <= araddr;
                            burst_size_r <= arsize;
                            shared_active <= selected_shared;
                            if (selected_shared) begin
                                cache_valid <= 1'b0;
                                cache_addr  <= araddr;
                                cache_len   <= arlen;
                                cache_size  <= arsize;
                            end
                            state <= S_EXT_R;
                        end
                    end
                end

                S_EXT_R: begin
                    if (rvalid && rready) begin
                        if (shared_active)
                            cache_mem[beat_idx] <= rdata;
                        if (rlast) begin
                            if (shared_active) cache_valid <= 1'b1;
                            shared_active <= 1'b0;
                            state <= S_IDLE;
                        end else begin
                            beat_idx <= beat_idx + 1'b1;
                        end
                    end
                end

                S_CACHE: begin
                    if (owner_ready) begin
                        if (beat_idx == cache_len)
                            state <= S_IDLE;
                        else
                            beat_idx <= beat_idx + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
