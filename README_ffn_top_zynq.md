# Implementation 3: ffn_top_zynq — Zynq-7000 Optimized FFN (D=2048, M=32)

## Overview

`ffn_top_zynq.v` is the **Zynq-7000 resource-optimized** top module targeting the
xc7z045ffv900-1. It uses aggressive time-multiplexing (`NUM_COLS=14`), a narrow
64-bit AXI bus, and a 64-bit output serializer to fit within the Zynq's constrained
DSP (900), IOB (362), and LUT (218K) budgets.

```
y = ReLU(x · W_up) · W_down
```

## Why This Configuration Exists

The full-parallel design (NUM_COLS=32, 512-bit AXI) requires:
- **2,048 DSPs** → only 900 available on Zynq (1148 overflow to LUTs)
- **1,085 IOBs** → only 362 available (300% over budget)
- **449K LUTs** → only 218K available (205% over budget)

This configuration reduces all three to fit.

## Key Design Decisions

### 1. NUM_COLS=14 (Time-Multiplexed Projection)

Each projection stage uses 14 parallel mul_col instances instead of 32.
This requires `ceil(32/14) = 3` sub-cycles per inner iteration.

| Sub-cycle | Columns Processed |
|-----------|-------------------|
| 0 | 0–13 (14 columns) |
| 1 | 14–27 (14 columns) |
| 2 | 28–31 (4 columns, partial) |

### 2. 64-bit AXI Bus (8× narrower than 512-bit)

Reduces IOB from 1,085 to ~189 pins. Weight tiles that required 32 beats
on 512-bit now require 256 beats on 64-bit. Throughput reduced by 8× but
fits within pin budget.

### 3. 64-bit Output Serializer

Instead of 512-bit parallel output tiles, tiles are streamed as 8 beats
of 64 bits each. The accumulator's internal readout FSM is disabled
(DISABLE_READOUT=1) and replaced by a dedicated serializer FSM.

## Module Hierarchy

```
ffn_top_zynq
├── u_axi_read_master     (AXI4 read master, 64-bit data bus)
├── u_fetch_addr_gen      (Stage 1: fetch tiles + address generation)
├── u_up_projection       (Stage 2: tm_proj_stage, NUM_COLS=14)
│   └── gen_col[0:13]
│       └── u_mul_col     (32 multipliers + pipelined adder tree each)
├── u_relu_stage          (Stage 3: parallel ReLU)
├── u_relu_bram           (256 × 512-bit BRAM)
├── u_down_projection     (Stage 4: tm_proj_stage, NUM_COLS=14)
│   └── gen_col[0:13]
│       └── u_mul_col     (32 multipliers + pipelined adder tree each)
├── u_accumulator         (Stage 5: DISABLE_READOUT=1)
├── u_output_bram         (64 × 512-bit BRAM)
└── [output serializer]   (5-state FSM, reads BRAM → 64-bit beats)
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `D` | 2048 | Input/output vector dimension |
| `M` | 32 | Tile dimension |
| `NUM_COLS` | 14 | Parallel columns per projection stage |
| `DATA_W` | 16 | Signed fixed-point width (Q8.8) |
| `AXI_DATA_W` | 64 | AXI data bus width |
| `AXI_ADDR_W` | 32 | AXI address width |

### Derived Constants

| Constant | Value | Calculation |
|----------|-------|-------------|
| NUM_TILES_D | 64 | D/M = 2048/32 |
| NUM_TILES_4D | 256 | 4D/M = 8192/32 |
| MAX_INNER | 256 | max(64, 256) |
| NUM_SUB | 3 | ceil(32/14) — 3 sub-cycles per inner iteration |
| TREE_DEPTH | 5 | log₂(32) — pipelined adder tree levels |
| ELEMS_PER_BEAT | 4 | AXI_DATA_W / DATA_W = 64/16 |
| BEATS_PER_TILE | 8 | M / ELEMS_PER_BEAT = 32/4 |
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

### AXI4 Read Master (64-bit)
| Port | Width | Dir | Description |
|------|-------|-----|-------------|
| `araddr` | 32 | output | Read address |
| `arlen` | 8 | output | Burst length − 1 |
| `arsize` | 3 | output | Transfer size (3 = 8 bytes) |
| `arburst` | 2 | output | Burst type (01 = INCR) |
| `arvalid` | 1 | output | Address valid |
| `arready` | 1 | input | Address ready |
| `rdata` | 64 | input | Read data |
| `rlast` | 1 | input | Last beat of burst |
| `rvalid` | 1 | input | Data valid |
| `rready` | 1 | output | Data ready |

### Streamed Output (64-bit)
| Port | Width | Dir | Description |
|------|-------|-----|-------------|
| `out_data` | 64 | output | 4 elements per beat (4 × 16 bits) |
| `out_addr` | 6 | output | Tile address (0..63) |
| `out_offset` | 3 | output | Beat offset within tile (0..7) |
| `out_valid` | 1 | output | Beat is valid |
| `out_last` | 1 | output | Last beat of last tile |

## Resource Estimates (D=2048, M=32, NUM_COLS=14)

| Resource | Estimated | Available | Utilization |
|----------|-----------|-----------|-------------|
| DSP48E1 | 896 | 900 | 99.6% |
| LUT | ~80K | 218K | 37% |
| FF | ~40K | 437K | 9% |
| BRAM36 | ~30 | 545 | 6% |
| IOB | ~189 | 362 | 52% |

### DSP Budget Detail

| Stage | Columns | Multipliers | DSP |
|-------|---------|-------------|-----|
| Up Projection | 14 | 14 × 32 | 448 |
| Down Projection | 14 | 14 × 32 | 448 |
| **Total** | | | **896** |

Remaining: 4 DSPs (margin for control logic)

### IOB Budget Detail

| Signal Group | Pins |
|-------------|------|
| araddr | 32 |
| arlen + arsize + arburst | 13 |
| arvalid + arready | 2 |
| rdata | 64 |
| rlast + rvalid + rready | 3 |
| out_data | 64 |
| out_addr + out_offset | 9 |
| out_valid + out_last | 2 |
| clk + rst_n + start + done | 4 |
| **Total** | **~193** |

## Output Serializer FSM

The serializer reads output BRAM tiles after `done` is asserted and streams
them as 64-bit beats:

```
SER_IDLE ──(done)──▶ SER_WAIT ──▶ SER_CAPTURE ──▶ SER_STREAM ──┐
     ▲                (1 cycle)    (capture        (8 beats     │
     │                             BRAM data)      per tile)    │
     │                                                         │
     │                    ┌─── last tile done ───────────────▶ SER_DONE
     └── (start) ◀───────┘
                          next tile: SER_WAIT
```

### BRAM Read Timing

```
Cycle N  : Serializer sets rd_en<=1, rd_addr<=idx
Cycle N+1: BRAM sees rd_en=1, latches mem[idx] into rd_reg
Cycle N+2: out_rd_data = mem[idx] → captured in SER_CAPTURE
```

### Output Beat Format

Each 64-bit beat contains 4 elements (ELEMS_PER_BEAT=4):

```
out_data[63:48] = element[3]  (16-bit signed)
out_data[47:32] = element[2]  (16-bit signed)
out_data[31:16] = element[1]  (16-bit signed)
out_data[15:0]  = element[0]  (16-bit signed)
```

For M=32: 8 beats per tile, 64 tiles total = **512 output beats**

### Serializer Timing

| State | Duration | Notes |
|-------|----------|-------|
| SER_IDLE | Until done | Waiting for computation |
| SER_WAIT | 1 cycle | BRAM read latency |
| SER_CAPTURE | 1 cycle | Capture tile data |
| SER_STREAM | 8 cycles | Output 8 beats |
| **Per tile** | **10 cycles** | |
| **64 tiles** | **640 cycles** | Full output streaming |

## External Memory Map

| Region | Base Address | Size | Description |
|--------|-------------|------|-------------|
| Input `x` | `0x0000_0000` | 4 KB | 2048 × 2 bytes |
| W_up | `0x1000_0000` | 32 MB | 2048 × 8192 × 2 bytes |
| W_down | `0x2000_0000` | 32 MB | 8192 × 2048 × 2 bytes |
| **Total** | | **~64 MB** | |

### AXI Burst Details (64-bit bus)

| Data Type | Beats per Tile | Bytes per Beat | Bytes per Tile |
|-----------|---------------|----------------|----------------|
| Input tile (1×32) | 8 | 8 | 64 |
| Weight tile (32×32) | 256 | 8 | 2,048 |
| ReLU tile | — | — | Local BRAM (no AXI) |

## Computation Flow

### UP Phase (256 output tiles × 64 inner iterations × 3 sub-cycles)

```
For tile_col = 0 to 255:
  For inner_idx = 0 to 63:
    Fetch: input_tile[inner_idx] + W_up_tile[inner_idx][tile_col]
    Sub-cycle 0: up_proj(input_tile, W_up_tile[cols 0-13])
    Sub-cycle 1: up_proj(input_tile, W_up_tile[cols 14-27])
    Sub-cycle 2: up_proj(input_tile, W_up_tile[cols 28-31])
    → Accumulate into all 32 accumulator entries
  ReLU: max(0, result) → store in ReLU BRAM
```

### DOWN Phase (64 output tiles × 256 inner iterations × 3 sub-cycles)

```
For tile_col = 0 to 63:
  For inner_idx = 0 to 255:
    Fetch: relu_tile[inner_idx] + W_down_tile[inner_idx][tile_col]
    Sub-cycle 0: down_proj(relu_tile, W_down_tile[cols 0-13])
    Sub-cycle 1: down_proj(relu_tile, W_down_tile[cols 14-27])
    Sub-cycle 2: down_proj(relu_tile, W_down_tile[cols 28-31])
    → Accumulate into all 32 accumulator entries
  Write result to output BRAM
```

### Sub-Cycle Detail (NUM_SUB=3)

```
Inner iteration with NUM_SUB=3:

  Cycle  0: Latch input + weight tile
  Cycle  1: Feed sub-cycle 0 (cols 0-13) → S_FEED
  Cycles 2-7: Adder tree pipeline + accumulate → S_PIPE → S_ACC
  Cycle  8: Feed sub-cycle 1 (cols 14-27) → S_FEED
  Cycles 9-14: Pipeline + accumulate
  Cycle 15: Feed sub-cycle 2 (cols 28-31) → S_FEED
  Cycles 16-21: Pipeline + accumulate
  (If is_last: output result; else: ready for next tile)
```

**Per inner iteration**: ~22 cycles (3 sub-cycles × ~7 cycles each + overhead)

**Total UP computation**: 256 × 64 × ~22 ≈ 360,448 cycles
**Total DOWN computation**: 64 × 256 × ~22 ≈ 360,448 cycles
**Estimated total**: ~721K cycles (+ AXI fetch overhead)

## Comparison with Other Configurations

| Metric | NUM_COLS=32 | NUM_COLS=16 | NUM_COLS=14 (Zynq) |
|--------|-------------|-------------|---------------------|
| DSP | 2,048 | 1,024 | **896** |
| LUT | ~150K | ~90K | **~80K** |
| IOB | 1,085 | 1,085 | **~189** |
| AXI width | 512 | 512 | **64** |
| Output | 512-bit parallel | 512-bit parallel | **64-bit serializer** |
| Sub-cycles/iter | 1 | 2 | 3 |
| Computation time | ~262K cyc | ~524K cyc | ~721K cyc |
| BRAM readout | Accum. FSM | Accum. FSM | **Serializer FSM** |
| Target | UltraScale+ | Mid-range FPGA | **Zynq-7000** |

## Synthesis

```bash
vivado -mode batch -source vivado_synth_zynq.tcl
```

The script targets `xc7z045ffv900-1` with:
- `FLATTEN_HIERARCHY none` (prevents Vivado BelGrid crash)
- `synth.maxNumOfProcessedInstances 5000`

## Simulation

```bash
# D=16, M=8, NUM_COLS=4 (4 beats/tile, serializer verification)
iverilog -g2005-sv -o ffn_zynq_tb \
    rtl/defines.v rtl/adder_tree.v rtl/mul_col.v rtl/bram_dp.v \
    rtl/axi_read_master.v rtl/fetch_addr_gen.v rtl/tm_proj_stage.v \
    rtl/relu_stage.v rtl/accumulator.v rtl/ffn_top_zynq.v \
    tb/tb_ffn_zynq.v
vvp ffn_zynq_tb
```

Expected output:
```
*** ALL 16 PASS (BRAM snoop) ***
*** ALL 16 PASS (Serializer) ***
*** ALL 64 ReLU PASS ***
*** Beat count CORRECT: 8 (=2 tiles × 4 beats) ***
```

## Important Implementation Notes

### 1. Accumulator DISABLE_READOUT=1

The accumulator's internal readout FSM is **disabled** in this configuration.
The output BRAM read port is exclusively controlled by the serializer FSM.
Do **not** connect the accumulator's `out_rd_en`/`out_rd_addr` outputs to
the output BRAM — the serializer drives these signals.

### 2. Accumulator Initialization Bug (Fixed)

The `tm_proj_stage` had a bug where only the first sub-cycle's accumulator
entries were initialized for `inner_idx==0`. The fix ensures ALL sub-cycles
initialize their entries. This is critical when `NUM_SUB > 1` (i.e., when
`M > NUM_COLS`).

### 3. Vivado BelGrid Crash

The design uses hierarchical `mul_col.v` modules and flat-bus adder trees
to avoid the Vivado internal error `HDConfig::lookup() BelGrid`. The
synthesis script sets `FLATTEN_HIERARCHY none`.

### 4. Weight Tile Layout on AXI

On a 64-bit AXI bus, each weight tile (32×32 = 1,024 elements = 2,048 bytes)
requires **256 beats** of 64 bits (4 elements per beat). The fetch stage
re-assembles the tile from these beats into the 512-bit `weight_tile_buf`.

## Source Files

```
rtl/defines.v           — Common macros
rtl/adder_tree.v        — Pipelined adder tree (5 levels for M=32)
rtl/mul_col.v           — Hierarchical 32 multipliers + 1 adder tree
rtl/bram_dp.v           — Dual-port BRAM (wide-word variant)
rtl/axi_read_master.v   — AXI4 read master (64-bit)
rtl/fetch_addr_gen.v    — Two-phase tile fetch + address generation
rtl/tm_proj_stage.v     — Time-multiplexed projection (NUM_COLS=14, NUM_SUB=3)
rtl/relu_stage.v        — Parallel ReLU (32 comparators)
rtl/accumulator.v       — Output accumulation (DISABLE_READOUT=1)
rtl/ffn_top_zynq.v      — This module (Zynq top + output serializer)
```
