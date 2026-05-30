# Week 3 — Distributed Training at Scale

## Goals

- Run real multi-GPU training
- Understand DeepSpeed ZeRO stages and when to use each
- Build production-grade training scripts

---

## Concepts to Study

### Parallelism Strategies

| Strategy | What is Split | When to Use |
|----------|--------------|-------------|
| **Data parallelism** | Data — each GPU holds a full model copy | Default starting point |
| **Tensor parallelism** | Individual weight matrices split across GPUs | Very large layers, same node |
| **Pipeline parallelism** | Model layers split into stages across GPUs | Models too large for tensor parallelism alone |

### DeepSpeed ZeRO (Zero Redundancy Optimiser)

ZeRO eliminates redundancy in data-parallel training by partitioning model state across GPUs:

#### Stage 1 — Optimiser State Sharding
- Optimizer states (momentum, variance) are split across GPUs
- Each GPU only stores 1/N of the optimiser states
- Memory saving: ~4× on optimiser memory (dominant for Adam)

#### Stage 2 — + Gradient Sharding
- Gradients are also partitioned — each GPU accumulates only its shard
- Memory saving: ~8× combined with Stage 1

#### Stage 3 — + Parameter Sharding (Full Model Sharding)
- Model parameters are also partitioned — each GPU holds 1/N of all weights
- Parameters are gathered on-the-fly during forward/backward passes
- Memory saving: proportional to number of GPUs, but adds communication overhead
- Use when the model does not fit in aggregate GPU memory with Stage 2

### FSDP (Fully Sharded Data Parallel)

PyTorch's native alternative to DeepSpeed ZeRO Stage 3:

- Built into `torch.distributed`
- No additional dependencies beyond PyTorch
- Slightly less flexible than DeepSpeed but tightly integrated with the PyTorch ecosystem
- `ShardingStrategy.FULL_SHARD` ≈ ZeRO Stage 3

### Key Memory Techniques

#### Gradient Checkpointing
- During the forward pass, **discard** intermediate activations instead of storing them
- During the backward pass, **recompute** them on-the-fly
- Trade-off: ~33% extra compute cost for significant memory savings

#### Mixed Precision
- **fp16**: 16-bit floating point — fast but numerically unstable, requires loss scaling
- **bf16**: brain float 16 — same range as fp32, less precision than fp16 — preferred on Ampere+ GPUs (A100, H100)
- Loss scaling compensates for fp16's limited dynamic range

#### Gradient Accumulation
- Perform `N` forward/backward passes before calling `optimizer.step()`
- Effective batch size = `per_device_batch_size × gradient_accumulation_steps × num_gpus`
- Allows training with large effective batch sizes on limited GPU memory

### The `accelerate` Library

`accelerate` provides a unified interface over multi-GPU, multi-node, DeepSpeed, and FSDP:

- Single code change: wrap your model with `accelerator.prepare()`
- Configure via `accelerate config` CLI — generates a YAML file
- Supports DeepSpeed JSON configs passed via the YAML config

---

## Hands-On Tasks

- [ ] Run a training job with `accelerate` on 2+ GPUs (use cloud: Lambda Labs, Vast.ai, or RunPod)
- [ ] Configure **DeepSpeed ZeRO Stage 2** for a 7B model fine-tune
- [ ] Configure **DeepSpeed ZeRO Stage 3** — observe memory savings vs Stage 2
- [ ] Enable **gradient checkpointing** and measure memory reduction
- [ ] Run the same job with **FSDP** — compare to DeepSpeed
- [ ] Profile a training step using `torch.profiler` — identify bottlenecks
- [ ] Implement proper **checkpoint saving and resuming** from mid-training
- [ ] Benchmark: tokens/second across different ZeRO stages and GPU counts

---

## Key Code Patterns

### DeepSpeed ZeRO Stage 2 Config (`ds_config.json`)

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "bf16": { "enabled": true },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

### Accelerate Training Script Skeleton

```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)

model, optimiser, train_dataloader, scheduler = accelerator.prepare(
    model, optimiser, train_dataloader, scheduler
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()

    if accelerator.is_main_process:
        accelerator.log({"loss": loss.item()})
```

### When to Use Which Strategy

| Scenario | Recommendation |
|----------|---------------|
| Model fits on one GPU, want faster training | Data parallel (DDP) |
| Model fits per GPU with Stage 1/2 | ZeRO Stage 2 — best performance/complexity trade-off |
| Model does not fit even with Stage 2 | ZeRO Stage 3 or FSDP |
| Prefer PyTorch-native, no DeepSpeed dependency | FSDP |
| Very large models (70B+) across many nodes | Pipeline + tensor parallelism (Megatron-LM) |

---

## Resources

- [DeepSpeed docs — config reference](https://www.deepspeed.ai/docs/config-json/)
- [HuggingFace `accelerate` docs](https://huggingface.co/docs/accelerate)
- [DeepSpeed + HuggingFace integration](https://huggingface.co/docs/transformers/deepspeed)
- Cloud GPU: Vast.ai (cheapest), Lambda Labs (reliable), RunPod

---

## Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| ZeRO: Memory Optimisations Toward Training Trillion Parameter Models | Rajbhandari et al., Microsoft | 2020 | ZeRO Stage 1/2/3 |
| ZeRO-Offload / ZeRO-Infinity | DeepSpeed team | 2021 | CPU and NVMe offloading |
| PyTorch FSDP | Meta | 2023 | Native PyTorch full sharding |

---

## Week 3 Deliverable

- Multi-GPU training script (configurable for DeepSpeed or FSDP)
- Benchmark table: memory usage and throughput across ZeRO stages
- Short writeup: when to use ZeRO Stage 2 vs Stage 3 vs FSDP
