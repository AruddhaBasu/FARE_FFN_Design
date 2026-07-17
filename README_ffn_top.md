# Implementation 1: ffn_top — Full Parallel FFN (D=2048, M=32)

## Overview

`ffn_top.v` is the **parameterized top-level module** for the pipelined feed-forward
network. When configured with `NUM_COLS=32` (equal to M), every output column is
computed in parallel — no time-multiplexing. This is the **highest-throughput**
configuration but also the most resource-intensive.

```
y = ReLU(x · W_up) · W_down
```

## Module Hierarchy

```
ffn_top
├── u_axi_read_master     (AXI4 read master, 512-bit data bus)
├── u_fetch_addr_gen      (Stage 1: fetch tiles + address generation)
├── u_up_projection       (Stage 2: tm_proj_stage, NUM_COLS=32)
│   └── gen_col[0:31]
│       └── u_mul_col     (32 multipliers + 1 pipelined adder tree each)
├── u_relu_stage          (Stage 3: parallel ReLU)
├── u_relu_bram           (256 × 512-bit BRAM for intermediate results)
├── u_down_projection     (Stage 4: tm_proj_stage, NUM_COLS=32)
│   └── gen_col[0:31]
│       └── u_mul_col     (32 multipliers + 1 pipelined adder tree each)
├── u_accumulator         (Stage 5: tile index decode + output BRAM write)
└── u_output_bram         (64 × 512-bit BRAM for final output)
```

## Parameters for D=2048, M=32

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D` | 2048 | Input/output vector dimension |
| `M` | 32 | Tile dimension |
| `NUM_COLS` | 32 | Parallel columns (full parallel = M) |
| `DATA_W` | 16 | Signed fixed-point width (Q8.8) |
| `AXI_DATA_W` | 512 | AXI data bus width |
| `AXI_ADDR_W` | 32 | AXI address width |

### Derived Constants

| Constant | Value | Calculation |
|----------|-------|-------------|
| NUM_TILES_D | 64 | D/M = 2048/32 |
| NUM_TILES_4D | 256 | 4D/M = 8192/32 |
| MAX_INNER | 256 | max(64, 256) |
| PROD_W | 32 | 2 × DATA_W |
| SUM_W | 37 | PROD_W + log₂(32) = 32 + 5 |
| ACC_W (up) | 45 | SUM_W + log₂(64) + 2 = 37 + 6 + 2 |
| ACC_W (down) | 47 | SUM_W + log₂(256) + 2 = 37 + 8 + 2 |
| NUM_SUB | 1 | ceil(32/32) — no time-multiplexing |
| TREE_DEPTH | 5 | log₂(32) — pipelined adder tree levels |

## Port Interface

### Control
| Port | Width | Dir | Description |
|------|-------|-----|-------------|
| `clk` | 1 | input | System clock |
| `rst_n` | 1 | input | Active-low reset |
| `start` | 1 | input | Assert to begin computation |
| `done` | 1 | output | Computation complete |

### AXI4 Read Master (512-bit)
| Port | Width | Dir | Description |
|------|-------|-----|-------------|
| `araddr` | 32 | output | Read address |
| `arlen` | 8 | output | Burst length − 1 |
| `arsize` | 3 | output | Transfer size (6 = 64 bytes) |
| `arburst` | 2 | output | Burst type (01 = INCR) |
| `arvalid` | 1 | output | Address valid |
| `arready` | 1 | input | Address ready |
| `rdata` | 512 | input | Read data |
| `rlast` | 1 | input | Last beat of burst |
| `rvalid` | 1 | input | Data valid |
| `rready` | 1 | output | Data ready |

### Output (512-bit tiles)
| Port | Width | Dir | Description |
|------|-------|-----|-------------|
| `output_tile_data` | 512 | output | 1×M output tile (32 × 16 bits) |
| `output_tile_addr` | 6 | output | Tile address (0..63) |
| `output_tile_valid` | 1 | output | Tile data is valid |

## Resource Estimates (D=2048, M=32, NUM_COLS=32)

| Resource | Estimated | Notes |
|----------|-----------|-------|
| DSP48E2 | 2048 | 32 cols × 32 muls × 2 stages |
| LUT | ~150K | Adder trees, control, pipeline regs |
| FF | ~100K | Pipeline registers, accumulators |
| BRAM36 | ~24 | ReLU BRAM: 16 KB + Output BRAM: 4 KB + buffers |
| IOB | ~1085 | 512-bit AXI + 512-bit output + control |

### Target FPGA

**Not suitable for Zynq-7000** (needs UltraScale+ or larger):
- Xilinx VU9P (6,840 DSP, 1,182K LUT)
- Xilinx VU13P (11,520 DSP, 1,728K LUT)
- Xilinx AU50 (1,344 DSP, 865K LUT) — tight fit
- Xilinx V70 (1,872 DSP, 851K LUT) — fits with margin

## External Memory Map

| Region | Base Address | Size | Description |
|--------|-------------|------|-------------|
| Input `x` | `0x0000_0000` | 4 KB | 2048 × 2 bytes |
| W_up | `0x1000_0000` | 32 MB | 2048 × 8192 × 2 bytes |
| W_down | `0x2000_0000` | 32 MB | 8192 × 2048 × 2 bytes |
| **Total** | | **~64 MB** | |

## Computation Flow

### UP Phase (64 output tiles × 64 inner iterations = 4,096 tile pairs)

```
For tile_col = 0 to 255:                    (256 output column tiles)
  For inner_idx = 0 to 63:                  (64 input row tiles)
    Fetch: input_tile[inner_idx] + W_up_tile[inner_idx][tile_col]
    Compute: up_proj(input_tile, W_up_tile) → partial sum
  ReLU: max(0, result) → store in ReLU BRAM at tile_col
```

### DOWN Phase (64 output tiles × 256 inner iterations = 16,384 tile pairs)

```
For tile_col = 0 to 63:                     (64 output column tiles)
  For inner_idx = 0 to 255:                 (256 ReLU row tiles)
    Fetch: relu_tile[inner_idx] from BRAM + W_down_tile[inner_idx][tile_col] from AXI
    Compute: down_proj(relu_tile, W_down_tile) → partial sum
  Accumulator: write result to output BRAM at tile_col
```

**Total AXI tile fetches**: 4,096 (UP) + 16,384 (DOWN) = **20,480 tile pairs**

### AXI Burst Details

| Data Type | Beats per Tile | Bytes per Beat | Bytes per Tile |
|-----------|---------------|----------------|----------------|
| Input tile (1×32) | 1 | 64 | 64 |
| Weight tile (32×32) | 32 | 64 | 2,048 |
| ReLU tile | — | — | Local BRAM (no AXI) |

## Pipeline Timing

```
                  ┌───────────┐   ┌──────────┐   ┌──────┐
  UP Phase:       │ Fetch     │──▶│ Up Proj  │──▶│ ReLU │──▶ BRAM
  (per tile pair) │ 3-33 cyc  │   │ 8 cyc    │   │ 1 cyc│
                  └───────────┘   └──────────┘   └──────┘

                  ┌───────────┐   ┌──────────┐   ┌────────────┐
  DOWN Phase:     │ Fetch     │──▶│ Down Proj│──▶│ Accumulator│──▶ BRAM
  (per tile pair) │ 3-33 cyc  │   │ 8 cyc    │   │ 1 cyc      │
                  └───────────┘   └──────────┘   └────────────┘
```

- Up projection latency (per inner iteration): ~8 cycles (5-level pipeline + FSM overhead)
- Down projection latency (per inner iteration): ~8 cycles
- With NUM_SUB=1 (full parallel): **no sub-cycle overhead**

## Output Readout

The accumulator's internal 4-state FSM streams output tiles:
```
RD_IDLE → RD_ISSUE → RD_RELAY → RD_CAPTURE → (next RD_ISSUE or RD_DONE)
```

- 2-cycle BRAM read latency per tile
- 64 tiles × ~3 cycles/tile ≈ **192 cycles** for full output readout
- Output appears on `output_tile_data` / `output_tile_addr` / `output_tile_valid`



## Source Files

```
rtl/defines.v           — Common macros
rtl/adder_tree.v        — Pipelined adder tree (5 levels for M=32)
rtl/mul_col.v           — Hierarchical 32 multipliers + 1 adder tree
rtl/bram_dp.v           — Dual-port BRAM (wide-word variant)
rtl/axi_read_master.v   — AXI4 read master (512-bit)
rtl/fetch_addr_gen.v    — Two-phase tile fetch + address generation
rtl/tm_proj_stage.v     — Time-multiplexed projection (NUM_COLS=32 → full parallel)
rtl/relu_stage.v        — Parallel ReLU (32 comparators)
rtl/accumulator.v       — Output accumulation + readout FSM (DISABLE_READOUT=0)
rtl/ffn_top.v           — This module (top)
```
