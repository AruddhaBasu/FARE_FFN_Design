`timescale 1ns/1ps
module tb_ffn_4lane_smoke;
    localparam D=8, M=4, HIDDEN_DIM=16, N=4, LANES=2, NUM_COLS=4;
    localparam DW=16, AW=32, AXW=64;
    reg clk=0, rst_n=0, start=0;
    always #1 clk=~clk;
    wire done;
    wire [AW-1:0] araddr; wire [7:0] arlen; wire [2:0] arsize; wire [1:0] arburst;
    wire arvalid; reg arready; reg [AXW-1:0] rdata; reg rvalid; wire rlast; wire rready;
    wire [AXW-1:0] out_data; wire [`CLOG2_MIN1(D/M)-1:0] out_addr;
    wire [2:0] out_offset; wire [`CLOG2_MIN1(N)-1:0] out_seq_idx;
    wire out_valid,out_row_last,out_last;
    ffn_block_zynq_4lane #(.D(D),.M(M),.HIDDEN_DIM(HIDDEN_DIM),.N(N),.LANES(LANES),.NUM_COLS(NUM_COLS),.DATA_W(DW),.AXI_DATA_W(AXW),.AXI_ADDR_W(AW)) dut(
      .clk(clk),.rst_n(rst_n),.start(start),.done(done),.alpha_pre(16'sd128),.alpha_post(16'sd128),
      .araddr(araddr),.arlen(arlen),.arsize(arsize),.arburst(arburst),.arvalid(arvalid),.arready(arready),
      .rdata(rdata),.rlast(rlast),.rvalid(rvalid),.rready(rready),.out_data(out_data),.out_addr(out_addr),.out_offset(out_offset),.out_seq_idx(out_seq_idx),.out_valid(out_valid),.out_row_last(out_row_last),.out_last(out_last));
    reg busy; reg [8:0] left; integer out_count;
    assign rlast = busy && (left == 9'd1);
    always @(posedge clk or negedge rst_n) begin
      if(!rst_n) begin arready<=1; rvalid<=0; rdata<='0; busy<=0; left<=0; end
      else if(!busy) begin arready<=1; if(arvalid&&arready) begin busy<=1; arready<=0; left<={1'b0,arlen}+1; rvalid<=1; rdata<='0; end end
      else if(rvalid&&rready) begin if(left==1) begin busy<=0; arready<=1; rvalid<=0; left<=0; end else left<=left-1; end
    end
    always @(posedge clk) if(out_valid) begin out_count=out_count+1; $display("OUT seq=%0d row_last=%0d last=%0d",out_seq_idx,out_row_last,out_last); end
    integer c;
    initial begin out_count=0; repeat(5) @(posedge clk); rst_n=1; repeat(2) @(posedge clk); @(negedge clk) start=1; @(negedge clk) start=0; c=0; while(!done&&c<1000000) begin @(posedge clk); c=c+1; end @(posedge clk); if(!done) $display("FAIL timeout"); else $display("PASS done cycles=%0d outputs=%0d",c,out_count); $finish; end
endmodule
