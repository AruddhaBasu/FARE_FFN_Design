# Implementation 2: ffn_top_2048 — Production Wrapper (D=2048, M=32, NUM_COLS=16)

## Overview

`ffn_top_2048.v` is a **thin wrapper** around `ffn_top.v` with hardcoded production
parameters. It uses `NUM_COLS=16` (half of M=32), requiring **2 sub-cycles** per
inner iteration. This trades ~50% DSP savings for ~2× longer computation time
compared to the full-parallel configuration.

```
y = ReLU(x · W_up) · W_down
```

## Key Design Choice: NUM_COLS=16

With M=32 output columns but only 16 parallel mul_col instances, each inner
iteration is split into 2 sub-cycles:
- **Sub-cycle 0**: Computes output columns 0–15
- **Sub-cycle 1**: Computes output columns 16–31

The weight tile is latched and different column groups are extracted each sub-cycle.

## Module Hierarchy

```
ffn_top_2048
└── u_ffn (ffn_top)
    ├── u_axi_read_master     (AXI4 read master, 512-bit)
    ├── u_fetch_addr_gen      (Stage 1)
    ├── u_up_projection       (Stage 2: tm_proj_stage, NUM_COLS=16)
    │   └── gen_col[0:15]
    │       └── u_mul_col     (32 multipliers + pipelined adder tree each)
    ├── u_relu_stage          (Stage 3)
    ├── u_relu_bram           (256 × 512-bit)
    ├── u_down_projection     (Stage 4: tm_proj_stage, NUM_COLS=16)
    │   └── gen_col[0:15]
    │       └── u_mul_col     (32 multipliers + pipelined adder tree each)
    ├── u_accumulator         (Stage 5)
    └── u_output_bram         (64 × 512-bit)
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D` | 2048 | Input/output vector dimension |
| `M` | 32 | Tile dimension |
| `NUM_COLS` | 16 | Parallel columns (from ffn_top default) |
| `DATA_W` | 16 | Signed fixed-point width (Q8.8) |
| `AXI_DATA_W` | 512 | AXI data bus width |
| `AXI_ADDR_W` | 32 | AXI address width |

### Derived Constants

| Constant | Value | Calculation |
|----------|-------|-------------|
| NUM_TILES_D | 64 | D/M = 2048/32 |
| NUM_TILES_4D | 256 | 4D/M = 8192/32 |
| MAX_INNER | 256 | max(64, 256) |
| NUM_SUB | 2 | ceil(32/16) — 2 sub-cycles per inner iteration |
| TREE_DEPTH | 5 | log₂(32) — pipelined adder tree levels |
| ACC_W (up) | 45 | SUM_W + log₂(64) + 2 |
| ACC_W (down) | 47 | SUM_W + log₂(256) + 2 |

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

## Resource Estimates (D=2048, M=32, NUM_COLS=16)

| Resource | Estimated | Notes |
|----------|-----------|-------|
| DSP48E2 | 1024 | 16 cols × 32 muls × 2 stages |
| LUT | ~90K | Adder trees, control, pipeline regs |
| FF | ~60K | Pipeline registers, accumulators |
| BRAM36 | ~24 | ReLU BRAM: 16 KB + Output BRAM: 4 KB |
| IOB | ~1085 | 512-bit AXI + 512-bit output + control |

### Target FPGAs

| FPGA | DSP | LUT | Fits? |
|------|-----|-----|-------|
| Xilinx VU9P | 6,840 | 1,182K | ✅ Easy |
| Xilinx AU50 | 1,344 | 865K | ✅ Fits (76% DSP) |
| Xilinx V70 | 1,872 | 851K | ✅ Fits (55% DSP) |
| Zynq xc7z045 | 900 | 218K | ❌ DSP overflow + IOB overflow |

**Key constraint**: The 512-bit AXI bus + 512-bit output requires **1,085 IOBs**,
which exceeds most mid-range FPGAs. This configuration is best suited for
FPGAs with high IOB count or on-chip AXI interconnect (e.g., Xilinx HBM parts).

## External Memory Map

| Region | Base Address | Size | Description |
|--------|-------------|------|-------------|
| Input `x` | `0x0000_0000` | 4 KB | 2048 × 2 bytes |
| W_up | `0x1000_0000` | 32 MB | 2048 × 8192 × 2 bytes |
| W_down | `0x2000_0000` | 32 MB | 8192 × 2048 × 2 bytes |
| **Total** | | **~64 MB** | |

## Computation Flow

### UP Phase (256 output tiles × 64 inner iterations)

```
For tile_col = 0 to 255:
  For inner_idx = 0 to 63:
    Fetch: input_tile[inner_idx] + W_up_tile[inner_idx][tile_col]
    Sub-cycle 0: up_proj(input_tile, W_up_tile[cols 0-15])
    Sub-cycle 1: up_proj(input_tile, W_up_tile[cols 16-31])
    → Accumulate into columns 0-31 of accumulator
  ReLU: max(0, result) → store in ReLU BRAM
```

### DOWN Phase (64 output tiles × 256 inner iterations)

```
For tile_col = 0 to 63:
  For inner_idx = 0 to 255:
    Fetch: relu_tile[inner_idx] + W_down_tile[inner_idx][tile_col]
    Sub-cycle 0: down_proj(relu_tile, W_down_tile[cols 0-15])
    Sub-cycle 1: down_proj(relu_tile, W_down_tile[cols 16-31])
    → Accumulate into columns 0-31 of accumulator
  Write result to output BRAM
```

### Sub-Cycle Detail

```
Inner iteration with NUM_SUB=2:

  Cycle  0: Latch input + weight tile
  Cycle  1: Feed sub-cycle 0 (cols 0-15) to mul_cols → S_FEED
  Cycle  2: Pipeline wait → S_PIPE
  Cycles 3-7: Adder tree pipeline (TREE_DEPTH=5)
  Cycle  8: Accumulate sub-cycle 0 results → S_ACC
  Cycle  9: Feed sub-cycle 1 (cols 16-31) to mul_cols → S_FEED
  Cycle 10: Pipeline wait → S_PIPE
  Cycles 11-15: Adder tree pipeline
  Cycle 16: Accumulate sub-cycle 1 results → S_ACC
  (If is_last: output result; else: ready for next tile)
```

**Per inner iteration**: ~16 cycles (2 sub-cycles × ~8 cycles each)

**Total UP computation**: 256 × 64 × ~16 ≈ 262,144 cycles
**Total DOWN computation**: 64 × 256 × ~16 ≈ 262,144 cycles
**Estimated total**: ~524K cycles (+ AXI fetch overhead)

## Comparison with Full Parallel

| Metric | NUM_COLS=32 | NUM_COLS=16 | Ratio |
|--------|-------------|-------------|-------|
| DSP | 2,048 | 1,024 | 0.5× |
| LUT | ~150K | ~90K | 0.6× |
| Sub-cycles/iter | 1 | 2 | 2× slower |
| Computation time | ~262K cycles | ~524K cycles | 2× |
| IOB | 1,085 | 1,085 | same |

## Output Readout

Same as ffn_top — accumulator's internal 4-state FSM:
```
RD_IDLE → RD_ISSUE → RD_RELAY → RD_CAPTURE → (next or RD_DONE)
```

64 tiles × ~3 cycles/tile ≈ **192 cycles** for full output readout.

## Synthesis

```bash
vivado -mode batch -source vivado_synth.tcl
```

The script uses `ffn_top` as top. To use the wrapper:
```tcl
set_property top ffn_top_2048 [current_fileset]
```

Or override NUM_COLS on ffn_top:
```tcl
set_property generic {D=2048 M=32 NUM_COLS=16 DATA_W=16 AXI_DATA_W=512 AXI_ADDR_W=32} [current_fileset]
```

## Simulation (scaled down)

```bash
# D=8, M=4 (fast smoke test)
iverilog -g2005-sv -o ffn_tb \
    rtl/defines.v rtl/adder_tree.v rtl/mul_col.v rtl/bram_dp.v \
    rtl/axi_read_master.v rtl/fetch_addr_gen.v rtl/tm_proj_stage.v \
    rtl/relu_stage.v rtl/accumulator.v rtl/ffn_top.v \
    tb/tb_ffn_complete.v
vvp ffn_tb

# D=32, M=16 (exercises NUM_SUB=1 for NUM_COLS=16)
iverilog -g2005-sv -o ffn_2048_tb \
    rtl/defines.v rtl/adder_tree.v rtl/mul_col.v rtl/bram_dp.v \
    rtl/axi_read_master.v rtl/fetch_addr_gen.v rtl/tm_proj_stage.v \
    rtl/relu_stage.v rtl/accumulator.v rtl/ffn_top.v \
    tb/tb_ffn_2048.v
vvp ffn_2048_tb
```

## Source Files

```
rtl/defines.v           — Common macros
rtl/adder_tree.v        — Pipelined adder tree (5 levels for M=32)
rtl/mul_col.v           — Hierarchical 32 multipliers + 1 adder tree
rtl/bram_dp.v           — Dual-port BRAM (wide-word variant)
rtl/axi_read_master.v   — AXI4 read master (512-bit)
rtl/fetch_addr_gen.v    — Two-phase tile fetch + address generation
rtl/tm_proj_stage.v     — Time-multiplexed projection (NUM_COLS=16, NUM_SUB=2)
rtl/relu_stage.v        — Parallel ReLU (32 comparators)
rtl/accumulator.v       — Output accumulation + readout FSM (DISABLE_READOUT=0)
rtl/ffn_top.v           — Parameterized top module
rtl/ffn_top_2048.v      — This module (production wrapper)
```
