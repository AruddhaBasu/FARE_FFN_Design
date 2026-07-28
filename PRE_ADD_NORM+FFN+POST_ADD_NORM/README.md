# Transformer FFN Block with Dynamic Tanh (DyT) — FPGA Implementation

A Zynq-optimized, fixed-point (Q8.8) implementation of a Transformer Feed-Forward
Network (FFN) block that replaces Layer Normalization with **Dynamic Tanh (DyT)**
from *"Transformers without Normalization"* (Zhu et al., 2025). The design is a
fully streaming, tile-based matmul with two DyT stages (pre and post),
time-multiplexed DSP columns, BRAM-based tanh lookup, and an AXI4-Full read
master for weight/parameter fetch.

---

## 1. Architectural Overview

### 1.1 Top-level block diagram

```
                           ┌────────────────────────────────────────────┐
   DRAM (AXI4-Full)        │  ffn_block_zynq  (D=2048, M=32, NUM_COLS)   |
   ────────────────┐       │                                             │
   x, res, γ₁,β₁,  │       │   ┌────────────┐   norm_input_bram (1W/1R)  |
   γ₂,β₂, W_up,    ├─AR/R─▶│   │  add_dyt   │──y_in───────────┐         │
   W_down          ◀───────│   │  _stage    │──z₁─residual_bram│        │
                           │   └────────────┘                 ▼         │
                           │                        ┌─────────────┐     │
                           │                        │  fetch_addr │     │
                           │                        │  _gen_dyt   │     │
                           │                        └──┬───────┬──┘     │
                           │                           ▼       ▼        │
                           │                    ┌──────────┐ ┌────────┐  │
                           │    ffn_output_bram │tm_proj   │ │tm_proj │  │
                           │    ◀─── y_ffn ──── │  _up     │ │ _down  │  │
                           │         ┌────┐      │(ReLU in- │ │(proj d)│  │
                           │         │acc │      │ between) │ │        │  │
                           │         └─┬──┘      └──────────┘ └───┬────┘  │
                           │           ▼                            │     │
                           │   ┌─────────────┐  ffn_output_bram    │     │
                           │   │ add_dyt_post│◀────y_ffn───────────┘     │
                           │   └──────┬──────┘◀────z₁(residual_bram)    │
                           │          │                                 │
                           └──────────┼─────────────────────────────────┘
                                      ▼
                             Streamed output (y_out)
                             AXI_DATA_W beats × NUM_TILES
```

### 1.2 Three-phase operation

| Phase | Function | Computation | Output |
|-------|----------|-------------|--------|
| **PRE_DYT** | Pre-add + DyT | y_in[k] = γ₁[k]·tanh(α₁·(x[k]+res[k]))+β₁[k] | norm_input_bram; also stores z₁=x+res → residual_bram for post-DyT |
| **FFN** | Up-ReLU-Down matmul | h = ReLU(y_in·W_up); y_ffn = h·W_down | ffn_output_bram |
| **POST_DYT** | Post-add + DyT + stream | y_out[k] = γ₂[k]·tanh(α₂·(y_ffn[k]+z₁[k]))+β₂[k] | Streaming output port |

A simple 4-state phase controller FSM starts each phase with a **one-cycle
start pulse** (critical — modules such as `accumulator` and `add_dyt_post`
would misbehave if given a level-based `start`), so phase transitions are:

```
IDLE --start--> PRE_DYT --pre_done--> FFN --ffn_done--> POST_DYT --post_done--> IDLE
```

The AXI read master is **time-multiplexed** between the three phases via a
request mux; only one phase owns the AR/R channels at a time.

### 1.3 Why DyT instead of LayerNorm?

Standard LayerNorm requires per-token mean/variance:

```
LN(x) = γ · (x - μ) / √(σ²+ε) + β
```

This is expensive in hardware:

- Two reductions (mean, variance) → large adder trees
- Division & square root → expensive DSP chains, iterative pipelines, or LUTs
- Two-pass per tile (compute stats, then scale)
- ~64+ DSPs per normalization layer

DyT replaces this with an *element-wise* operation:

```
DyT(x) = γ ⊙ tanh(α · x) + β
```

- No statistics, no reductions
- Single-pass, fully element-wise
- One multiply, one BRAM LUT lookup, one MAC per element → **2 DSPs + 1 BRAM**
- Trainable α (scalar), γ and β (per-channel vectors) match LayerNorm's
  parameter count (2D + 1 vs 2D + 2 — negligible)

In the residual variant (this design), the pre/post DyT are applied to the
sum `x + residual`, mirroring Pre-LN/Post-LN placement.

---

## 2. Algorithmic Details

### 2.1 Tile-based blocked matmul

The full D-dimensional FFN is:

```
h = ReLU( x · W_up  + b_up )        // W_up ∈ ℝ^{D×4D}
y = h · W_down + b_down              // W_down ∈ ℝ^{4D×D}
```

Tiled with tile size M (default 32), each weight matrix is divided into M×M
blocks:

```
W_up   = [ W_up[i,j]   ] ∈ ℝ^{(D/M) × (4D/M)}    i ∈ [0,D/M), j ∈ [0,4D/M)
W_down = [ W_down[j,k] ] ∈ ℝ^{(4D/M) × (D/M)}
```

For each output column tile `j` (UP) or `k` (DOWN), we stream over inner
tile index `i` and **accumulate** partial products:

```
UP   output tile[:,j] = Σ_i x[:,i] @ W_up[i,j]
DOWN output tile[:,k] = Σ_j ReLU(h)[:,j] @ W_down[j,k]
```

The matmul operates on a **1×M input row tile** streaming against **M×M weight
tiles** — a classic output-stationary dataflow.

### 2.2 Time-multiplexed column projection (`tm_proj_stage`)

Instead of instantiating M parallel `mul_col` modules (which would cost
M×M = 1024 DSPs per projection for M=32), we process `NUM_COLS` columns per
sub-cycle. Each sub-cycle:

1. A dynamic MUX extracts `NUM_COLS` weight columns (of width M×DATA_W)
   from the latched M×M weight tile.
2. `NUM_COLS` parallel `mul_col` instances each compute one partial output
   element via M multipliers + a 5-level pipelined binary adder tree.
3. After `TREE_DEPTH+1` wait cycles for the adder-tree pipeline, results
   are accumulated into the corresponding entries of a M×ACC_W register
   file.
4. After `NUM_SUB = ⌈M/NUM_COLS⌉` sub-cycles, all M columns for this inner
   iteration are complete.
5. On `is_last`, the accumulator is saturated and output as a Q8.8 tile.

For `NUM_COLS=14, M=32`: NUM_SUB=3 sub-cycles × ~TREE_DEPTH+2 ≈ 3×7 ≈ 21 cycles
per inner iteration. DSP count: 14 columns × 32 multipliers = **448 DSPs per
projection** (vs 1024 fully parallel). `NUM_COLS` is a synthesis-time knob
that directly trades DSPs for latency.

### 2.3 Element-wise pipeline in DyT stages

Each DyT stage has an element-processing loop of 3 cycles per element:

```
Cycle N    (S_DSP_SCALE) : s = α · z[k]                 → Q16.16, fire tanh LUT rd_en
Cycle N+1  (S_DSP_TANH)  : tanh_val = LUT[s]            → Q8.8 (1-cycle BRAM latency)
Cycle N+2  (S_DSP_AFFINE): y[k] = sat(γ·tanh_val + β)   → Q8.8
```

This is a simple unpipelined loop — not fully pipelined because each iteration
depends on the previous result and shares the two DSPs. Adding a pipeline
register on elem_idx would make it 1 element/cycle (single-iteration-rate),
but at ~98 cycles/tile and 64 tiles (~6.4 Kcycles) this is already negligible
versus the FFN's ~950 Kcycle compute.

#### 2.3.1 The tanh BRAM LUT (`tanh_lut.v`)

Exploiting odd symmetry `tanh(-x) = -tanh(x)`, the ROM stores only positive
values:

- **Depth:** 1024 entries = 4.0 × 256
- **Step:** 2⁻⁸ ≈ 0.0039 real units (1 LSB of Q8.8)
- **Range:** [0, 4.0) — tanh(4) ≈ 0.9993 ≈ 1.0 in Q8.8 (0x0100 = 256)
- **Word:** 16-bit signed Q8.8
- **Initialization:** `$readmemh("tanh_lut_init.hex")` — generated offline by
  `gen_tanh_lut.py` using `math.tanh`
- **Saturation:** when `|s| ≥ 4.0`, output is forced to ±ONE_Q (±256 = ±1.0)
- **Addressing:** convert Q16.16 input `s` to Qint.8 index by right-shifting
  `(S_FRAC-FRAC_W)=8` bits; take absolute value; clamp; mux sign at output.

**BRAM usage:** One 36Kb BRAM36E1 (1024 × 16 = 16 Kbit → fits in one 18Kb with
room to spare; synthesis will pick the smallest).

#### 2.3.2 Fixed-point arithmetic (Q8.8)

All activations, weights, α/γ/β use signed Q8.8:

```
bit:  15 14 ... 8 7 ... 0
      S  integer  fractional
```

Range: [−128.0, +127.996], resolution: 2⁻⁸ ≈ 0.0039.

Multipliers produce Q16.16 (32-bit) products — sign-extended to 32 bits
**before** multiplication (a critical fix; naïve 16×16→16 in Verilog
truncates). β is sign-extended to 32 bits then left-shifted 8 (arithmetic)
to land in Q16.16 before addition. The final sum saturates to Q8.8:

```
MAX_THRESH = 0x007FFF00 = +127.99609375  (= 32767 << 8)
MIN_THRESH = 0xFF800000 = −128.0
```

Bits `[23:8]` of the saturated sum give the Q8.8 result.

### 2.4 Adder tree (`adder_tree.v`)

Pipelined binary reduction tree for `mul_col`'s M products:

- Inputs: M × PROD_W = M × 32-bit Q16.16 products
- Output: 1 × (PROD_W + log₂M)-bit sum (37-bit for M=32)
- Pipeline: `PIPE=1` inserts a register after every level, giving
  `LEVELS = $clog2(M)` = 5 cycles latency, 1 result/cycle throughput.
- Vivado-safe "flat" bus layout: each level is a flat N×OUT_W-bit wire,
  avoiding 2D-array packing that crashes Vivado's BelGrid database for
  large M.

### 2.5 AXI4-Full read master (`axi_read_master.v`)

A minimal single-burst AXI4 read master with:

- AR channel: drives `araddr/arlen/arsize/arburst=INCR/arvalid`; waits for
  `arready` handshake.
- R channel: accepts beats with `rready ∧ resp_ready`; passes data through
  combinational to the active phase.
- One burst at a time; `req_ready = (state==IDLE)`.
- Back-pressure safe: R beats are only acknowledged when the consumer
  (phase) is in its WAIT state (`resp_ready` is gated by phase FSM).

### 2.6 BRAM primitives

| BRAM | Width | Depth | Purpose |
|------|-------|-------|---------|
| `norm_input_bram` | M×16 = 512 b | D/M = 64 | Holds y_in (pre-DyT output) for FFN UP phase input |
| `residual_bram`  | M×16 = 512 b | 64 | Holds z₁ = x+res for post-DyT residual add |
| `relu_bram`      | M×16 = 512 b | 4D/M = 256 | Holds ReLU(h) for DOWN phase |
| `ffn_output_bram`| M×16 = 512 b | 64 | Holds y_ffn (accumulator output) for post-DyT input |
| `tanh_lut`       | 16 b | 1024 | tanh(x) ROM |

All BRAMs use synchronous (registered) reads with 1-cycle latency, inferred
via `(* ram_style = "block" *)` and registered read-enable/address (required
for Vivado to infer block RAM instead of LUTRAM).

### 2.7 AXI memory map

| Base address | Content | Size per tile | Total size |
|--------------|---------|---------------|------------|
| `0x0000_0000` | Input activations `x`          | M·2 bytes | D·2 = 4 KB |
| `0x0800_0000` | Residual input `res`           | M·2 bytes | D·2 = 4 KB |
| `0x0C00_0000` | γ₁ (pre-DyT per-channel scale) | M·2 bytes | D·2 = 4 KB |
| `0x0D00_0000` | β₁ (pre-DyT per-channel bias)  | M·2 bytes | D·2 = 4 KB |
| `0x1000_0000` | W_up weights                   | M²·2 bytes | D·4D·2 = 32 MB |
| `0x2000_0000` | W_down weights                 | M²·2 bytes | 4D·D·2 = 32 MB |
| `0x3000_0000` | γ₂ (post-DyT per-channel scale)| M·2 bytes | D·2 = 4 KB |
| `0x3100_0000` | β₂ (post-DyT per-channel bias) | M·2 bytes | D·2 = 4 KB |

Weights dominate memory — each FFN weight matrix is ~32 MB for D=2048.

---

## 3. Parameterization

All architectural knobs are Verilog parameters on `ffn_block_zynq`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `D`         | 2048 | Model hidden dimension |
| `M`         | 32   | Tile size (elements/tile) |
| `NUM_COLS`  | 8    | Parallel mul_col instances per projection (DSP knob) |
| `DATA_W`    | 16   | Element bit-width |
| `FRAC_W`    | 8    | Fractional bits (Q8.8) |
| `AXI_DATA_W`| 64   | AXI data bus width (bits) — beats carry AXI_DATA_W/DATA_W elements |
| `AXI_ADDR_W`| 32   | AXI address width |

**Sub-derivations:**

- `NUM_TILES_D  = D/M = 64`
- `NUM_TILES_4D = 4D/M = 256`
- `ELEMS_PER_BEAT = AXI_DATA_W/DATA_W = 4` (64/16)
- `BEATS_PER_TILE = M/ELEMS_PER_BEAT = 8`
- `NUM_SUB = ⌈M/NUM_COLS⌉` (3 for NUM_COLS=14, 4 for NUM_COLS=8)
- `TREE_DEPTH = $clog2(M) = 5`
- `ACC_W = 2·DATA_W + $clog2(M) + $clog2(MAX_INNER) + 2 ≈ 47 bits`

---

## 4. Performance Analysis

All numbers below use default parameters D=2048, M=32, AXI_DATA_W=64, on a
Zynq US+ with DSP48E1s and 36Kb BRAMs, running at a target 200 MHz (5 ns period).
Latency formulae generalize.

### 4.1 Cycle budget per phase

#### 4.1.1 PRE_DYT (`add_dyt_stage`)

Per tile of M=32 elements:
- AXI fetches (x, res, γ, β): 4 × BEATS_PER_TILE = 4×8 = 32 beats, plus
  AR handshake overhead ≈ ~36 cycles
- Parallel add z=x+res: 1 cycle
- Element loop: M × 3 cycles = 32×3 = 96 cycles
- Write (y→norm_bram, z→res_bram) + stream: 1 cycle

Total ≈ **~135 cycles/tile × 64 tiles = 8,640 cycles ≈ 43 µs @ 200 MHz**

AXI is the bottleneck here; compute is negligible. If weights/biases are
cached in on-chip BRAM this drops to ~100 cycles/tile.

#### 4.1.2 FFN matmul (up-ReLU-down)

**UP projection** per output column tile j ∈ [0, 256):
- 1 norm_bram read: 2 cycles (REQ+RELAY+WAIT state machine; BRAM has 1-cycle
  latency, FSM adds 1-cycle bubble)
- 1 weight fetch via AXI: WEIGHT_BEATS = (32×32)/4 = 256 beats ≈ 260 cycles
- Inner loop over i ∈ [0, 64):
  - NUM_SUB=3 sub-cycles × (1 feed + TREE_DEPTH+1 pipe + 1 acc) ≈ 3×8 = 24 cycles
  - 64 × 24 = 1536 cycles
- Total per UP column: ~1800 cycles; ×256 columns = **~460 Kcycles**

**ReLU:** 1 cycle/tile, completely shadowed by pipeline; **free**.

**DOWN projection** per output column tile k ∈ [0, 64):
- 1 ReLU_bram read: 2 cycles
- 1 weight fetch: 256 beats ≈ 260 cycles
- Inner loop over j ∈ [0, 256):
  - NUM_SUB × 8 = 24 cycles per inner; 256 × 24 = 6144 cycles
- Total per DOWN column: ~6400 cycles; ×64 columns = **~410 Kcycles**

**FFN total ≈ 870 Kcycles ≈ 4.4 ms @ 200 MHz.**

*AXI bandwidth sensitivity:* if weights are streamed over a 64-bit AXI at 200
MHz with 100% utilization, the bandwidth is 1.6 GB/s. W_up+W_down total 64 MB,
so minimum AXI fetch time ≈ 40 ms — but the design overlaps compute with AXI
fetch of the next tile. The actual wall-clock is dominated by whichever is
slower between compute (~4.4 ms) and AXI streaming; with 64-bit AXI and full
pipeline overlap, achievable throughput is compute-bound. Widen AXI to 128b
or 256b to get out of the memory-bound regime.

#### 4.1.3 POST_DYT (`add_dyt_post`)

Same compute as PRE_DYT, but ffn_output and residual are read from local
BRAMs instead of AXI; only γ₂, β₂ are AXI-fetched.

- Per tile: 2 local reads (2+2 cycles) + 2×8 beats AXI + 32×3 elem cycles +
  8 stream beats ≈ ~110 cycles
- ×64 tiles ≈ **7,000 cycles ≈ 35 µs**

#### 4.1.4 End-to-end

| Phase | Cycles | Time @ 200 MHz |
|-------|--------|----------------|
| PRE_DYT  | ~8.6 K   | 43 µs |
| FFN      | ~870 K   | 4.4 ms |
| POST_DYT | ~7.0 K   | 35 µs |
| **Total**| **~886 K** | **~4.5 ms** |

Equivalent throughput: one D=2048 token/FFN per 4.5 ms → **~222 tokens/s per
block** (single-block). With batching or wider AXI/DATA_W this scales up.
For a full Transformer layer (attention + FFN), the FFN is typically ⅔–¾
of the compute, so expect ~6 ms/layer at 200 MHz.

### 4.2 Resource utilization (post-synthesis estimates)

With NUM_COLS=8, M=32, D=2048:

| Resource | Count | Notes |
|----------|-------|-------|
| **DSP48E1** | ~2 + 2·NUM_COLS·M + 0 = **~514** | 2 for DyT MACs, 2·(8×32) = 512 for matmul. ReLU is compare-only, adder tree is LUT/FF. (NUM_COLS=14 → ~898 DSPs.) |
| **BRAM36E1** | **~7** | tanh LUT (1) + norm_input_bram (1×36) + residual_bram (1×36) + relu_bram (4×36 for 256×512b) + ffn_output_bram (1×36). ReLU BRAM needs the most storage: 256×512b = 128 Kbit, ≈4×36Kb BRAMs. |
| **LUT** | **~8–10 K** | FSMs (~1 K), adders (M 16-bit adds per DyT = ~320), multiplier partial-product logic (dominant for DSP-poor configs), address muxes, adder trees (37-bit adds × 5 levels × NUM_COLS × 2 projections ≈ 3–4 K). |
| **FF**  | **~5–7 K** | Pipeline registers in adder trees (dominant), tile buffers (5 tiles × 512b = ~320 FF), FSM state. |

*Compared with LayerNorm-based designs using ~64 DSPs per LN plus division/sqrt
logic (10–20 more DSPs and LUT-heavy sqrt), DyT saves ~80+ DSPs per layer at
a cost of 1 BRAM and ~100 LUT/FF.*

### 4.3 Throughput vs NUM_COLS (DSP vs latency trade-off)

| NUM_COLS | DSP/proj | DSP total | Cycles/inner (UP) | Cycles/tile-col (UP) | UP total cycles |
|----------|----------|-----------|--------------------|-----------------------|------------------|
| 32 (full) | 1024 | ~2050 | ~8 | ~10 | ~2.6 K |
| 16        | 512  | ~1026 | ~16 | ~18 | ~4.6 K |
| 14        | 448  | ~898  | ~24 | ~26 | ~6.7 K |
| 8         | 256  | ~514  | ~32 | ~34 | ~8.7 K |
| 4         | 128  | ~258  | ~64 | ~66 | ~17 K |
| 1         | 32   | ~66   | ~256 | ~258 | ~66 K |

For NUM_COLS=8 (default), the design uses ~514 DSPs (≈25% of a ZU7EV's
2020 DSPs) and completes the FFN in ~4.4 ms. For a smaller device, NUM_COLS=4
fits in a Z-7020 (220 DSPs) with ~2× latency penalty.

### 4.4 Timing / critical path

Expected critical path (post-placement, ~ns):

```
mul_col multiplier (16b×16b→32b) → first-level adder → 37-bit register
```

The multiplier in Xilinx DSP48E1 is 0.6 ns; the 37-bit add following is ~0.8 ns,
plus routing ~2–3 ns, giving **~4–5 ns** comfortably meeting a 200 MHz
(5 ns) clock.

The DyT path is shorter (16-bit multiply, 16-bit BRAM lookup, 32-bit add
→ saturation mux) and is not critical.

### 4.5 Numerical precision (Q8.8)

- tanh LUT quantization: 1-LSB step in input → max error ~(1/2) LSB in output
  ≈ 0.002; tanh is monotonic and smooth so no artifact risk.
- Multiplier truncation: Q16.16→Q8.8 uses rounding-via-truncation after
  saturation; worst-case per-element quantization error ½ LSB = 0.002.
- Accumulator is 47-bit — no risk of overflow across 256 inner iterations
  (each partial product ≤ 32767×32767 ≈ 1e9, sum over 256 iters ≈ 2.8e11,
  fits in 38 bits, 47 bits has 9 bits of headroom).
- Compared to fp32, expected per-element output error **< 0.01** (1.2 LSB),
  well within typical BERT/Transformer quantization tolerance (8-bit
  weight/activation quantization in literature shows <1% accuracy drop).

### 4.6 Power (rough estimate, Zynq US+)

At 200 MHz, 500 DSPs toggling ~30% of cycles, ~7 BRAMs active, ~10K LUTs:

- DSP dynamic power: ~500 × ~3 mW/MHz × 200 MHz × 0.3 ≈ **~90 mW**
- BRAM: ~7 × ~1 mW/MHz × 200 MHz ≈ **~1.4 mW**
- Logic/clock: **~50–100 mW**
- Static: ~100–200 mW (device-dependent)

Total FFN block ≈ **250–400 mW** — far below a GPU or even a CPU
implementation (30–100× more efficient per token) and suitable for
edge/real-time inference.

### 4.7 Scaling

- **Increasing D:** linear growth in tile count (NUM_TILES_D = D/M),
  quadratic in weight memory. BRAM sizes auto-scale via parameters.
- **Widening M:** reduces tile count but blows up M² weight-tile size and
  NUM_COLS·M DSP products. M=32 is the sweet spot for 36Kb BRAMs
  (M×DATA_W=512 bits fits perfectly per BRAM column).
- **Widening AXI_DATA_W:** directly reduces weight fetch time; recommended
  to use the full PS/PL AXI width (128b or 256b).
- **Multi-block pipelines:** instantiating multiple `ffn_block_zynq` for
  multiple Transformer layers is straightforward; add FIFO-based streaming
  between them.

---



