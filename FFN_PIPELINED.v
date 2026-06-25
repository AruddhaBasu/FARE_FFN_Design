//Verilog Implementation of Feed Forward Network 
//Where up and down projection matrices are stored 
//in BRAM
module ffn_bram_pipeline #(
    parameter N = 8,             //Input vector size
    parameter BIT_WIDTH = 16,	 //Number of bits in each element of the input vector, up and down projection matrices and the output vector
    parameter FRAC_BIT = 8,		  //Number of bits which are alloted to fractional part of a number out of the total bits alloted in BIT_WIDTH
    parameter TILE_WIDTH = 1024  //As 8x8 up and down projection matrices are used in each pass of the ffn, so the total bit capacity of the BRAM = no. of rows x no. of columns x BIT_WIDTH = 8x8x16 = 1024
)(
    input clk,// Clock input 
    input rst,//Restart input
    input start,//Start input
    input signed [(N*BIT_WIDTH)-1:0] row_input,//Signed Input Vector 
    
// BRAM Interface for Up-Projection Weights
  output reg [1:0] up_bram_addr,//BRAM Address Bus for up projection. The adress bus consists of 2 bits as the entire up matrix is partitioned into 4 tiles and 2 bits are required to address them.
  input signed [TILE_WIDTH-1:0] up_bram_rdata,//BRAM data bus. Instead of reading the matrix element-by-element over 64 cycles, the architecture fetches complete 8x8 matrix all at once.
    
    // BRAM Interface for Down-Projection Weights (Transposed)
  output reg [1:0] down_bram_addr,//BRAM Address Bus for down projection
  input signed [TILE_WIDTH-1:0] down_bram_rdata,//BRAM Data Bus for down projection
    
    // Outputs
  output reg signed [(N*BIT_WIDTH)-1:0] accumulator_bank,//Register Bank where the output of the FFN would stored. It has the same size as of the input.
    output reg done//Control signal conveying the completion message for one row vector input
);

    // FSM Control States
  reg [2:0] cycle_count;//3 bit counter to track the number of cycles when running is active. 
    reg running;// 1 bit status register. When running = 1, the FSM transitions from idle state to active state
    
    // Control Path Pipeline Shift Registers: In a pipelined architecture, data 
  reg [1:0] tile_idx_s1, tile_idx_s2, tile_idx_s3, tile_idx_s4;//Tracks the specific tile index currently being processed. 
    reg valid_s1, valid_s2, valid_s3, valid_s4;//Act as data valid qualifiers for each stage. As a valid input vector moves through the network, the valid bit shifts right every clock cycle.

    // Pipeline Stage Data Registers
  reg signed [(N*BIT_WIDTH)-1:0] row_input_latch;// Holds original input vector. When the FSM starts calculation, it latches the incoming row vector here.
  reg signed [(N*BIT_WIDTH)-1:0] up_proj_out;//Holds the intermediate result of the up projection matrix multiplication
  reg signed [(N*BIT_WIDTH)-1:0] relu_out;//Holds the rectified linear activation output vector.

    // -----------------------------------------------------------
    // STRUCTURAL UNPACKING (Eliminates Bit-Slicing Simulator Bugs)
    // -----------------------------------------------------------
    wire signed [BIT_WIDTH-1:0] row_in_elements [0:N-1];
    wire signed [BIT_WIDTH-1:0] up_w_matrix [0:N-1][0:N-1];
    wire signed [BIT_WIDTH-1:0] up_proj_elements [0:N-1];
    wire signed [BIT_WIDTH-1:0] relu_elements [0:N-1];
    wire signed [BIT_WIDTH-1:0] down_w_matrix [0:N-1][0:N-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : unpack_loops
            // Unpack latch input row vector
            assign row_in_elements[gi] = row_input_latch[gi*BIT_WIDTH +: BIT_WIDTH];
            // Unpack intermediate stage vectors
            assign up_proj_elements[gi] = up_proj_out[gi*BIT_WIDTH +: BIT_WIDTH];
            assign relu_elements[gi] = relu_out[gi*BIT_WIDTH +: BIT_WIDTH];
            
            for (gj = 0; gj < N; gj = gj + 1) begin : unpack_matrices
                // Unpack 1024-bit flat BRAM signals into true 2D signed matrices
                assign up_w_matrix[gi][gj]   = up_bram_rdata[((gi*N)+gj)*BIT_WIDTH +: BIT_WIDTH];
                assign down_w_matrix[gi][gj] = down_bram_rdata[((gi*N)+gj)*BIT_WIDTH +: BIT_WIDTH];
            end
        end
    endgenerate

    // -----------------------------------------------------------
    // STAGE 1: FSM Controller & Address Generation
    // -----------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cycle_count     <= 0;
            running         <= 1'b0;
            valid_s1        <= 1'b0;
            tile_idx_s1     <= 0;
            up_bram_addr    <= 0;
            row_input_latch <= 0;
        end else if (start) begin
            running         <= 1'b1;
            cycle_count     <= 0;
            valid_s1        <= 1'b1;
            tile_idx_s1     <= 0;
            up_bram_addr    <= 0;
            row_input_latch <= row_input; 
        end else if (running) begin
            if (cycle_count < 3) begin
                cycle_count  <= cycle_count + 1;
                tile_idx_s1  <= tile_idx_s1 + 1;
                up_bram_addr <= tile_idx_s1 + 1;
                valid_s1     <= 1'b1;
            end else begin
                valid_s1     <= 1'b0; 
                if (cycle_count == 6) begin 
                    running  <= 1'b0;
                end else begin
                    cycle_count <= cycle_count + 1;
                end
            end
        end else begin
            valid_s1 <= 1'b0;
        end
    end

    // -----------------------------------------------------------
    // CONTROL PATH PROPAGATION
    // -----------------------------------------------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_s2    <= 0; tile_idx_s2 <= 0;
            valid_s3    <= 0; tile_idx_s3 <= 0;
            valid_s4    <= 0; tile_idx_s4 <= 0;
            done        <= 0;
        end else begin
            valid_s2    <= valid_s1;
            tile_idx_s2 <= tile_idx_s1;
            
            valid_s3    <= valid_s2;
            tile_idx_s3 <= tile_idx_s2;
            
            valid_s4    <= valid_s3;
            tile_idx_s4 <= tile_idx_s3;
            
            done        <= valid_s4 && (tile_idx_s4 == 2'b11); 
        end
    end

    // -----------------------------------------------------------
    // STAGE 2: Up-Projection Math Engine (Purely Clocked Loops)
    // -----------------------------------------------------------
    integer i, j;
    reg signed [(2*BIT_WIDTH)-1:0] up_mult;
    reg signed [BIT_WIDTH-1:0] up_acc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            up_proj_out <= 0;
        end else if (valid_s2) begin
            for (i = 0; i < N; i = i + 1) begin
                up_acc = 0;
                for (j = 0; j < N; j = j + 1) begin
                    up_mult = row_in_elements[j] * up_w_matrix[i][j];
                    up_acc  = up_acc + (up_mult >>> FRAC_BIT);
                end
                up_proj_out[i*BIT_WIDTH +: BIT_WIDTH] <= up_acc;
            end
        end
    end

    // -----------------------------------------------------------
    // STAGE 3: ReLU Activation Function
    // -----------------------------------------------------------
    integer k;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            relu_out <= 0;
        end else if (valid_s3) begin
            for (k = 0; k < N; k = k + 1) begin
                if (up_proj_elements[k][BIT_WIDTH - 1] == 1'b1) // Signed MSB Check
                    relu_out[k*BIT_WIDTH +: BIT_WIDTH] <= {BIT_WIDTH{1'b0}};
                else
                    relu_out[k*BIT_WIDTH +: BIT_WIDTH] <= up_proj_elements[k];
            end
        end
    end

    // Direct assignment linking down-BRAM lookup to Stage 3 timing slot
    always @(*) begin
        down_bram_addr = tile_idx_s3; 
    end

    // -----------------------------------------------------------
    // STAGE 4: Down-Projection Math & Accumulation Bank
    // -----------------------------------------------------------
    integer x, y;
    reg signed [(2*BIT_WIDTH)-1:0] down_mult;
    reg signed [BIT_WIDTH-1:0] down_acc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulator_bank <= 0;
        end else begin
            if (start) begin
                accumulator_bank <= 0; // Clear values for fresh execution
            end else if (valid_s4) begin
                for (x = 0; x < N; x = x + 1) begin
                    down_acc = 0;
                    for (y = 0; y < N; y = y + 1) begin
                        down_mult = relu_elements[y] * down_w_matrix[x][y];
                        down_acc  = down_acc + (down_mult >>> FRAC_BIT);
                    end
                    // Accumulate directly across clocks safely
                    accumulator_bank[x*BIT_WIDTH +: BIT_WIDTH] <= 
                        accumulator_bank[x*BIT_WIDTH +: BIT_WIDTH] + down_acc;
                end
            end
        end
    end

endmodule