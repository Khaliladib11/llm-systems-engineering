# Week 5 — CUDA, Triton, and Kernel Optimisation

## Goals

- Write custom GPU kernels
- Profile and optimise real bottlenecks
- Understand what happens at the hardware level

---

## Concepts to Study

### GPU Architecture

```
GPU
└── Streaming Multiprocessors (SMs)
    └── Warps (32 threads, execute in lockstep)
        └── CUDA Cores (scalar FP/INT operations)
        └── Tensor Cores (matrix multiply, FP16/BF16/FP8)
```

- **Warp**: the fundamental execution unit — 32 threads that run the same instruction simultaneously
- **Occupancy**: how many warps are active on an SM at once — limited by registers and shared memory
- **Tensor Cores**: perform 4×4 or 16×16 matrix multiply-accumulate in a single clock — critical for attention and linear layers

### Memory Hierarchy

```
HBM (High Bandwidth Memory) — ~3 TB/s, ~80 GB on A100
    │
    ▼
L2 Cache — ~40 MB, shared across all SMs
    │
    ▼
L1 Cache / Shared Memory — ~192 KB per SM, programmer-controlled
    │
    ▼
Registers — fastest, private to each thread
```

- **The key insight**: most GPU operations are *memory-bandwidth bound* — the bottleneck is reading data from HBM, not compute
- **Roofline model**: plots achievable FLOPS vs arithmetic intensity (FLOPS/byte) — tells you whether a kernel is memory or compute bound

### Roofline Analysis

```
        │              /
TFLOPS  │             / compute bound
        │            /
 peak   │───────────/──────────────── peak compute
compute │          /
        │         /
        │        /  memory bound
        │       /
        └──────────────────────────►
               arithmetic intensity (FLOPS/byte)
```

- Move operations to the right (fuse operations, reduce memory accesses) to escape memory-bound territory

### CUDA Programming Model

```
kernel<<<grid, block>>>(args)

grid  = (Bx, By, Bz)   — number of blocks
block = (Tx, Ty, Tz)   — threads per block (max 1024)

thread ID:  threadIdx.{x,y,z}
block ID:   blockIdx.{x,y,z}
block dim:  blockDim.{x,y,z}
```

- **Shared memory**: declared with `__shared__`, lives in L1, visible to all threads in a block — use to avoid repeated HBM reads
- **Bank conflicts**: shared memory is divided into 32 banks — simultaneous accesses to the same bank serialise; avoid by padding

### Warp Divergence

When threads in a warp take different branches of an `if` statement, both branches execute serially and inactive threads are masked out — called **warp divergence**. Minimise branching within a warp.

### Triton

Triton is a Python-based DSL for writing GPU kernels that compiles to PTX:

- Programs in terms of **tiles** (blocks of data) rather than individual threads
- Handles thread indexing, shared memory management, and memory coalescing automatically
- Much easier than raw CUDA while achieving near-cuBLAS performance for many kernels
- Backed by an MLIR compiler — generates efficient PTX/SASS

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

## Key Code Patterns

### Triton Fused Softmax Kernel

```python
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused softmax using a custom Triton kernel."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

### Profiling with torch.profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
) as prof:
    for _ in range(10):
        output = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### Profiling with Nsight Compute (CLI)

```bash
# Profile a single training step
ncu --set full \
    --target-processes all \
    -o profile_report \
    python train_step.py

# Open in GUI
ncu-ui profile_report.ncu-rep
```

### Key Things to Look For in ncu

| Metric | What it Means |
|--------|--------------|
| `Memory Throughput` | How much of peak HBM bandwidth is being used |
| `Compute (SM) Throughput` | How much of peak compute is being used |
| `Occupancy` | Fraction of warps that are active — low = register/shared-memory pressure |
| `L1/TEX Cache Throughput` | Cache hit rate for L1 |
| `Warp Efficiency` | Fraction of active threads in active warps — low = warp divergence |

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
