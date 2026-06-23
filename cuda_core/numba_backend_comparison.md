# numba-cuda vs numba-cuda-mlir — comparative analysis

Same source file (`examples/gl_interop_fluid_numba_cuda_mlir.py`, 9 kernels), run
through two compilers via the import switch at the top. Everything below holds
the workload, GPU, and CUDA toolchain constant and varies only the compiler.

## Setup / what is held constant

| | classic | mlir |
|---|---|---|
| package | numba-cuda 0.30.2 (+ numba 0.65.1, llvmlite 0.47.0) | numba-cuda-mlir 0.4.0 |
| frontend | Python→Numba IR→LLVM IR→**NVVM**→PTX | Python→**MLIR**→PTX (LTOIR) |
| opt / LTO default | NVVM **opt-3, no LTO** | **LTO, opt>0** |
| PTX→cubin | library (nvJitLink/nvptxcompiler) | ptxas binary |
| Python | 3.12.13 (primary) | 3.12.13 (primary) + 3.14 (interpreter datapoint) |
| cuda-bindings | 13.3.1 | 13.3.1 |
| ptxas | CUDA 13.3 | CUDA 13.3 |

Constant: RTX 5880 Ada (sm_89, driver 595.71.05), CUDA 13.3 ptxas, numpy 2.4.6,
workload (512² unless swept, 30 pressure iters, fixed seed). Primary comparison
uses matched Python 3.12 to remove the interpreter confound.

## 1. Equivalence — bit-exact

Per-kernel output on identical seeded inputs, diffed offline. **0.0 difference**
on all 9 kernels at every grid size (256², 512², 1024², 2048²). Same source →
same results, regardless of compiler or problem size.

## 2. PTX / SASS code generation

Registers/thread (default config) and SASS instruction count (lower = tighter):

| kernel | regs c/m | SASS c/m | note |
|---|---|---|---|
| pressure_jacobi (62% GPU) | 30 / 30 | 184 / **152** | mlir −17% |
| compute_curl | 26 / **22** | 168 / **136** | mlir −19% |
| divergence | 26 / **22** | 168 / **136** | mlir −19% |
| subtract_gradient | 26 / **22** | 160 / **144** | mlir −10% |
| apply_vorticity | 40 / **37** | 392 / **376** | mlir −4% |
| colorize | 34 / **32** | 248 / 208 | **classic** −16% |
| advect_velocity | 39 / **38** | 216 / 216 | tie |
| advect_dye | 40 / 40 | 256 / 256 | tie |
| splat | 28 / **25** | 256 / 256 | tie |

- In its default config mlir emits slightly fewer registers and tighter SASS on
  the stencil kernels (incl. the dominant `pressure_jacobi`); classic wins
  `colorize`; ties elsewhere. No register count exceeds the ~42 occupancy
  threshold at 16×16, so these deltas do **not** change occupancy.
- **Shared property — float64 promotion:** Python literals (`0.25`, `0.5`, …) are
  float64, so all arithmetic runs in f64 (`DFMA`) and converts to f32 only on
  store. Identical on both backends (hence bit-exact). Large latent perf
  opportunity: `numpy.float32` literals would cut this.
- **Shared property — transcendentals:** `exp` lowers to an f64 DFMA polynomial
  (no fast `MUFU.EX2`); `sqrt`/division use `MUFU.RSQ64H`/`RCP64H` (f64
  Newton-Raphson). Identical counts on both backends.
- `ld.global` counts match exactly per kernel (same memory access).

Despite mlir's fewer static instructions on `pressure_jacobi`, its **GPU runtime
is identical** to classic (§4) — these kernels are f64-throughput bound, not
instruction-issue bound, so the SASS delta does not convert to speed.

## 3. JIT compilation time (context pre-warmed; isolates compile)

Full 9-kernel compile, median of 5 cold processes:

| | total | pressure_jacobi |
|---|---|---|
| classic (3.12) | 1.36 s | 0.16 s |
| mlir (3.12) | **1.17 s** | **0.10 s** |
| mlir (3.14) | 1.06 s | 0.10 s |

mlir compiles ~14% faster on matched Python (faster per-kernel except the first,
which absorbs one-time backend init). No on-disk cache → paid every process.

## 4. Runtime / FPS + crossover (headless, matched Python 3.12)

| grid | classic fps | mlir fps | regime |
|---|---|---|---|
| 256² | 1579 | **2188** | launch-bound — mlir +39% |
| 512² | 1346 | **1859** | launch-bound — mlir +38% |
| 1024² | 677 | 671 | compute-bound — parity |
| 2048² | 180 | 179 | compute-bound — parity |

- **Crossover ≈ 1024².** Below it the workload is host-launch-bound (37 launches/
  frame, tiny kernels) and mlir's lighter dispatch wins ~38%; at/above it the GPU
  dominates and the backends are identical.
- Pure GPU time/frame is **equal** (nsys, 512²: 0.34 ms both). ncu on
  `pressure_jacobi` (512²): registers 30/30, achieved occupancy 76.6/77.2%,
  SM throughput 79/79%, DRAM ~3% — codegen-equal at the hardware level.
- Per-launch host dispatch (no-op kernel, isolated): **classic 8.6 µs, mlir
  5.3 µs** (mlir −38%). ×37 launches ≈ the measured wall-vs-GPU gap. This is the
  entire source of the eager-mode FPS difference.

## 5. Launch model — eager vs CUDA graphs (dedicated vector)

Both backends capture/instantiate/launch a graph via the driver API on the numba
stream (no asymmetry). Capturing the 37-launch compute pipeline and replaying as
one graph launch:

| grid | classic eager→graph | mlir eager→graph | graphed result |
|---|---|---|---|
| 256² | 1728 → 6404 (3.71×) | 2527 → 6420 (2.54×) | **parity (~6410)** |
| 512² | 1704 → 2714 (1.59×) | 2294 → 2511 (1.09×) | **~parity** |

- **CUDA graphs erase the gap.** Graphed, both converge to the same FPS (classic
  marginally ahead at 512²). classic gains *more* from graphing (3.71× vs 2.54×)
  because it had more per-launch overhead to remove. Build cost ~0.5–0.8 ms, one
  time. ⇒ mlir's eager advantage is real only on the eager path; any graph-using
  app sees parity.

## 6. LTO isolation (2×2, pressure_jacobi registers)

| | LTO off | LTO on |
|---|---|---|
| classic (NVVM) | 30 | 30 |
| mlir (MLIR) | 50 | 30 |

At **matched LTO the frontends are identical** (30=30). mlir's default-config
codegen tightness is the **LTO pipeline**, not the MLIR frontend — which alone
(LTO off) is *worse* (50). classic's NVVM reaches 30 without LTO.

## Verdict per vector

| vector | winner | magnitude |
|---|---|---|
| correctness / equivalence | tie | bit-exact, all sizes |
| GPU codegen (runtime) | tie | equal GPU time; identical occupancy/regs on dominant kernel |
| GPU codegen (static SASS) | mlir (default) | −10–19% on stencils; but = runtime; and = classic at matched LTO |
| JIT compile time | mlir | ~14% faster full set |
| eager runtime / FPS (<1024²) | mlir | +38% (per-launch dispatch 5.3 vs 8.6 µs) |
| runtime / FPS (≥1024²) | tie | compute-bound parity |
| with CUDA graphs | tie | gap erased; classic gains more |
| frontend codegen at matched LTO | tie | 30=30 |

## Bottom line

For this sim the two compilers produce **equivalent GPU code** (bit-exact output,
equal GPU runtime, identical occupancy; the static-SASS edge is an LTO effect that
classic matches at equal LTO). The only real-world difference is **eager-mode host
launch overhead**: mlir dispatches ~38% cheaper, which matters **only** in the
launch-bound regime (<~1024² here) and **disappears entirely under CUDA graphs**.
mlir also compiles ~14% faster. Net: pick based on launch model and grid size, not
codegen — they generate the same machine code.

Two backend-independent optimization opportunities surfaced: (a) the kernels run
in float64 (Python literals) — float32 literals would cut arithmetic cost on both;
(b) the workload is launch-bound below ~1024² — CUDA graphs roughly triple FPS on
both backends.
```
Reproduction: harnesses in /tmp/backend_cmp/ (equiv_sizes, codegen_dump,
extract_classic_sass, jit_time, throughput_sweep, graph_fps, lto_probe,
launch_overhead). Envs: numba-classic (3.12), numba-mlir-py312 (3.12),
numba-mlir (3.14). Run with PYTHONSAFEPATH=1 from a non-repo cwd.
```
