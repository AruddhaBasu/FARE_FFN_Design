//============================================================================
// defines.v — Common parameters and macros for the Pipelined FFN
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
//   HIDDEN_DIM   = 4*D     : Configurable FFN hidden dimension
//   NUM_TILES_H  = HIDDEN_DIM/M : Number of hidden-dimension tiles
//   PROD_W       = 2*DATA_W: Multiplier product width
//   ACC_UP_W     = PROD_W + $clog2(M) + 2 : Up-proj accumulator width
//   ACC_DOWN_W   = PROD_W + $clog2(M) + 2 : Down-proj accumulator width
//============================================================================

// Safe clog2 helper.  A one-entry memory/index still needs a one-bit
// address port in synthesizable RTL; plain $clog2(1) evaluates to zero.
`define CLOG2_MIN1(x) (((x) <= 1) ? 1 : $clog2(x))

// Macro to index into a flat 1×M tile vector
// tile[k] = tile_flat[k*DATA_W +: DATA_W]
`define TILE_IDX(flat, k, DW) flat[k*DW +: DW]

// Macro to index into a flat M×M weight tile vector
// weight[k][j] = weight_flat[(k*M+j)*DW +: DW]
`define WEIGHT_IDX(flat, k, j, M, DW) flat[(k*(M)+(j))*(DW) +: (DW)]
