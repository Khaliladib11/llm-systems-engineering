# Week 5 — CUDA, Triton, and Kernel Optimisation

## Goals

- Write custom GPU kernels
- Profile and optimise real bottlenecks
- Understand what happens at the hardware level

---

## Concepts to Study

### GPU Architecture

A GPU is composed of many **Streaming Multiprocessors (SMs)**. Each SM runs multiple **warps** — groups of 32 threads that execute the same instruction in lockstep. Within each SM there are:

- **CUDA Cores**: scalar floating-point and integer operations
- **Tensor Cores**: perform matrix multiply-accumulate on small tiles (e.g. 16×16) in a single clock — critical for attention and linear layers
- **Occupancy**: the fraction of maximum possible warps that are active on an SM — limited by register and shared memory usage

### Memory Hierarchy

From slowest/largest to fastest/smallest:

| Level | Bandwidth | Capacity | Scope |
|-------|-----------|----------|-------|
| HBM (High Bandwidth Memory) | ~3 TB/s | ~80 GB (A100) | All SMs |
| L2 Cache | ~6 TB/s | ~40 MB | All SMs |
| L1 / Shared Memory | ~20 TB/s | ~192 KB per SM | Threads in a block |
| Registers | Fastest | Very limited | Individual threads |

**The key insight**: most GPU operations are *memory-bandwidth bound* — the bottleneck is moving data from HBM, not performing FLOPs.

### Roofline Model

The roofline model plots achievable performance (TFLOPS) against arithmetic intensity (FLOPS per byte of memory traffic):

- **Memory-bound region**: performance scales with memory bandwidth — moving operations left to right (reducing memory traffic) helps
- **Compute-bound region**: performance is capped by peak TFLOPS — only algorithmic improvements or more compute helps
- **Kernel fusion** moves operations to the right by eliminating intermediate writes to HBM

### CUDA Programming Model

Kernels are launched with a grid of thread blocks. Each block contains up to 1024 threads, organised along x/y/z dimensions. Threads within a block share L1 / shared memory and can synchronise with `__syncthreads()`.

- **Shared memory**: programmer-controlled L1 — use it to cache data that multiple threads in a block need, avoiding repeated HBM reads
- **Bank conflicts**: shared memory is divided into 32 banks — simultaneous accesses to the same bank serialise; avoid by padding arrays
- **Warp divergence**: when threads in a warp take different branches of an `if` statement, both branches execute serially and inactive threads are masked — minimise branching within a warp

### Triton

Triton is a Python-based DSL for writing GPU kernels that compiles to PTX:

- Programs in terms of **tiles** (blocks of data) rather than individual threads
- Handles thread indexing, shared memory management, and memory coalescing automatically
- Much easier than raw CUDA while achieving near-cuBLAS performance for many kernels
- Backed by an MLIR compiler — generates efficient PTX/SASS

**Why fusion matters**: a fused softmax kernel reads the input once and writes the output once. A naïve implementation (separate exp, sum, divide) reads and writes to HBM three times — pure memory bandwidth waste.

### Profiling Tools

| Tool | What it Shows |
|------|--------------|
| `torch.profiler` | CPU + CUDA timeline, operator-level GPU time |
| `nsys` (Nsight Systems) | System-wide GPU/CPU timeline, kernel launches, memory transfers |
| `ncu` (Nsight Compute) | Per-kernel hardware metrics: memory throughput, compute throughput, occupancy, warp efficiency |

#### Key `ncu` Metrics

| Metric | What it Means |
|--------|--------------|
| Memory Throughput | How much of peak HBM bandwidth is being used |
| Compute (SM) Throughput | How much of peak compute is being used |
| Occupancy | Fraction of warps active — low = register/shared-memory pressure |
| L1/TEX Cache Throughput | Cache hit rate for L1 |
| Warp Efficiency | Fraction of active threads in active warps — low = warp divergence |

---

## Hands-On Tasks

- [ ] Write a **CUDA matrix multiplication** kernel from scratch (no cuBLAS)
- [ ] Optimise it: add shared memory tiling, measure speedup
- [ ] Write a **Triton fused softmax kernel** — understand why fusion reduces memory bandwidth
- [ ] Write a **Triton fused LayerNorm kernel**
- [ ] Profile a real transformer forward pass with `torch.profiler` — find the top 3 bottlenecks
- [ ] Profile with `ncu` (Nsight Compute) — check memory bandwidth utilisation and occupancy
- [ ] Replace a PyTorch operation with your custom Triton kernel in a small model — measure end-to-end speedup
- [ ] Read and annotate the Flash Attention Triton implementation

---

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [GPU Mode YouTube lectures](https://www.youtube.com/@GPUMODE) — excellent free content
- *Programming Massively Parallel Processors* — Kirk & Hwu — chapters 1–6
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

## Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| FlashAttention | Dao et al. | 2022 | IO-aware attention with recomputation |
| FlashAttention-2 | Dao | 2023 | Improved parallelism and work partitioning |
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | Tillet et al. | 2019 | Triton language and compiler |

---

## Week 5 Deliverable

A GitHub repo containing:

- CUDA matmul kernel (naive + tiled versions with benchmark)
- Triton fused softmax kernel
- Triton fused LayerNorm kernel

Plus:

- Profiling report: roofline analysis of one transformer operation
- README with benchmark results (TFLOPS, memory bandwidth utilisation)
