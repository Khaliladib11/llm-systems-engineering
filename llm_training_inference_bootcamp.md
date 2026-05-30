# LLM Training & Inference — 6-Week Intensive Bootcamp

> Target roles: LLM Training Expert, ML Infra Engineer, Inference Engineer, GPU Optimization Engineer
> Assumes solid Python, cloud computing, and general LLM familiarity.

---

## Structure

- **5 days/week**, ~6-8 hours/day
- Each week has: concepts, hands-on tasks, papers, and a deliverable
- By the end you will have a portfolio of real projects covering training, inference, and kernel optimization

---

## Week 1 — SFT: Supervised Fine-Tuning from the Ground Up

### Goals
- Deeply understand how instruction fine-tuning works
- Get hands-on with the full SFT pipeline
- Understand LoRA/QLoRA and when to use them

### Concepts to Study
- Causal language modeling objective (next token prediction) vs instruction tuning
- Data formatting: Alpaca format, ShareGPT/ChatML format, system prompts
- Tokenization edge cases: padding, truncation, attention masks
- LoRA: low-rank decomposition, rank `r`, alpha, which layers to target
- QLoRA: 4-bit NF4 quantization + LoRA, double quantization
- SFT trainer internals: packing, data collator, gradient accumulation

### Hands-On Tasks
- [ ] Set up a clean training environment (Python 3.11, PyTorch, CUDA)
- [ ] Fine-tune **Qwen2.5-1.5B** on the Alpaca dataset using Hugging Face `trl` SFTTrainer
- [ ] Repeat the above with **LoRA** (PEFT) — compare memory usage and output quality
- [ ] Repeat with **QLoRA** (4-bit) — run on a single consumer GPU if available
- [ ] Write a script that converts a raw Q&A CSV into ShareGPT format
- [ ] Log training metrics to **Weights & Biases** (loss curves, learning rate schedule)
- [ ] Evaluate outputs manually: write 10 test prompts, compare base vs fine-tuned

### Resources
- Hugging Face `trl` SFTTrainer docs: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face `peft` docs: https://huggingface.co/docs/peft
- `bitsandbytes` quantization: https://github.com/TimDettmers/bitsandbytes
- Blog: "Fine-tuning Llama 2 with QLoRA" — Tim Dettmers

### Papers
- **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al. (2021)
- **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al. (2023)

### Week 1 Deliverable
A GitHub repo with:
- Fine-tuned Qwen2.5-1.5B (LoRA adapters pushed to HuggingFace Hub)
- Training script with W&B logging
- README with loss curves, sample outputs, and GPU memory stats

---

## Week 2 — RLHF, DPO, and Reward Modeling

### Goals
- Understand the full RLHF pipeline conceptually and practically
- Train a reward model from a preference dataset
- Run DPO and GRPO fine-tuning

### Concepts to Study
- RLHF pipeline: SFT model → reward model → PPO optimization
- PPO for LLMs: policy model, value model, reference model, KL divergence penalty
- DPO (Direct Preference Optimization): why it replaces PPO in many cases, Bradley-Terry model
- GRPO (Group Relative Policy Optimization): used in DeepSeek-R1 and Qwen — understand the group reward advantage
- Reward hacking and how to mitigate it
- Preference datasets: what "chosen" vs "rejected" means, how to build them
- KL penalty: why we don't want the model to drift too far from the SFT model

### Hands-On Tasks
- [ ] Train a **reward model** on the `Anthropic/hh-rlhf` preference dataset using a small base model
- [ ] Run **DPO** fine-tuning using `trl` DPOTrainer on a preference dataset
- [ ] Run **GRPO** fine-tuning using `trl` GRPOTrainer — use a math reasoning task (GSM8K style)
- [ ] Compare outputs: SFT-only vs DPO vs GRPO on the same prompts
- [ ] Implement a simple rule-based reward function (e.g., reward correct format, penalize refusals)
- [ ] Log all runs to W&B, compare reward scores across training steps

### Resources
- `trl` DPOTrainer docs: https://huggingface.co/docs/trl/dpo_trainer
- `trl` GRPOTrainer docs: https://huggingface.co/docs/trl/grpo_trainer
- Anthropic HH-RLHF dataset: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Blog: "RLHF: Reinforcement Learning from Human Feedback" — HuggingFace

### Papers
- **InstructGPT: Training language models to follow instructions with human feedback** — Ouyang et al., OpenAI (2022)
- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — Rafailov et al. (2023)
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek (2025) — focus on GRPO section

### Week 2 Deliverable
- Reward model uploaded to HuggingFace Hub
- DPO and GRPO fine-tuned model adapters
- Comparison writeup (markdown): reward scores, qualitative outputs, what worked

---

## Week 3 — Distributed Training at Scale

### Goals
- Run real multi-GPU training
- Understand DeepSpeed ZeRO stages and when to use each
- Build production-grade training scripts

### Concepts to Study
- Data parallelism vs tensor parallelism vs pipeline parallelism
- DeepSpeed ZeRO:
  - Stage 1: optimizer state sharding
  - Stage 2: + gradient sharding
  - Stage 3: + parameter sharding (full model sharding)
- FSDP (Fully Sharded Data Parallel) — PyTorch native alternative to DeepSpeed
- Gradient checkpointing: trade compute for memory
- Mixed precision training: fp16 vs bf16, loss scaling
- Gradient accumulation: simulate large batch sizes on limited GPU memory
- `accelerate` library: how it abstracts multi-GPU and DeepSpeed config
- Checkpointing strategies: save every N steps, resume from checkpoint

### Hands-On Tasks
- [ ] Run a training job with `accelerate` on 2+ GPUs (use cloud: Lambda Labs, Vast.ai, or RunPod)
- [ ] Configure **DeepSpeed ZeRO Stage 2** for a 7B model fine-tune
- [ ] Configure **DeepSpeed ZeRO Stage 3** — observe memory savings vs Stage 2
- [ ] Enable **gradient checkpointing** and measure memory reduction
- [ ] Run the same job with **FSDP** — compare to DeepSpeed
- [ ] Profile a training step using `torch.profiler` — identify bottlenecks
- [ ] Implement proper **checkpoint saving and resuming** from mid-training
- [ ] Benchmark: tokens/second across different ZeRO stages and GPU counts

### Resources
- DeepSpeed docs: https://www.deepspeed.ai/docs/config-json/
- HuggingFace `accelerate` docs: https://huggingface.co/docs/accelerate
- DeepSpeed + HuggingFace integration: https://huggingface.co/docs/transformers/deepspeed
- Cloud GPU: Vast.ai (cheapest), Lambda Labs (reliable), RunPod

### Papers
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — Rajbhandari et al., Microsoft (2020)
- **ZeRO-Offload / ZeRO-Infinity** — DeepSpeed team (2021)
- **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** — Meta (2023)

### Week 3 Deliverable
- Multi-GPU training script (configurable for DeepSpeed or FSDP)
- Benchmark table: memory usage and throughput across ZeRO stages
- Short writeup: when to use ZeRO Stage 2 vs 3 vs FSDP

---

## Week 4 — Inference: vLLM, SGLang, and Serving

### Goals
- Deploy models efficiently at scale
- Understand the internals of modern inference engines
- Benchmark and optimize serving performance

### Concepts to Study
- Naive inference bottlenecks: why sequential generation is slow
- KV Cache: what it stores, why it grows with context length, memory implications
- **PagedAttention** (vLLM): virtual memory for KV cache, why it eliminates fragmentation
- **Continuous batching**: process requests of different lengths simultaneously
- **Tensor parallelism** in inference: splitting attention heads and MLP layers across GPUs
- **Speculative decoding**: use a small draft model to propose tokens, large model verifies
- **Flash Attention**: IO-aware exact attention, recomputation trick to save HBM bandwidth
- Quantization for inference: GPTQ (post-training), AWQ (activation-aware), fp8
- **RadixAttention** (SGLang): prefix caching via radix tree, huge speedup for shared prefixes

### Hands-On Tasks
- [ ] Deploy a 7B model with **vLLM** — OpenAI-compatible API server
- [ ] Benchmark vLLM: measure TTFT (time to first token) and throughput (tokens/sec) at different batch sizes
- [ ] Enable **tensor parallelism** in vLLM across 2 GPUs
- [ ] Deploy the same model with **SGLang** — benchmark vs vLLM
- [ ] Serve a model with **AWQ quantization** — compare throughput and quality vs fp16
- [ ] Implement a simple load test script (concurrent requests, measure latency percentiles)
- [ ] Enable **speculative decoding** in vLLM (use a small draft model)
- [ ] Experiment with different `max_num_seqs` and `gpu_memory_utilization` settings

### Resources
- vLLM docs: https://docs.vllm.ai
- SGLang docs: https://sgl-project.github.io
- AWQ: https://github.com/mit-han-lab/llm-awq
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- vLLM blog: https://vllm.ai/blog/2023/06/20/vllm.html

### Papers
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** — Kwon et al. (2023) — vLLM paper
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao et al. (2022)
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** — Dao (2023)
- **SGLang: Efficient Execution of Structured Language Model Programs** — Zheng et al. (2024)
- **Accelerating Large Language Model Decoding with Speculative Sampling** — Chen et al., DeepMind (2023)

### Week 4 Deliverable
- Deployed vLLM + SGLang serving setup
- Benchmark report: latency (p50, p95, p99), throughput, GPU memory across quantization levels
- Comparison table: vLLM vs SGLang on same model and hardware

---

## Week 5 — CUDA, Triton, and Kernel Optimization

### Goals
- Write custom GPU kernels
- Profile and optimize real bottlenecks
- Understand what happens at the hardware level

### Concepts to Study
- GPU architecture: SMs, warps (32 threads), CUDA cores, Tensor Cores
- Memory hierarchy: HBM (slow, large) → L2 cache → L1/shared memory (fast, small) → registers
- Memory bandwidth vs compute bound operations — roofline model
- CUDA programming model: grid → blocks → threads, thread indexing
- Shared memory usage: avoid global memory round-trips
- Bank conflicts in shared memory
- Warp divergence: why branching hurts performance
- **Triton**: Python-level kernel authoring, compiles to PTX, much easier than raw CUDA
- Profiling tools: `torch.profiler`, Nsight Systems (`nsys`), Nsight Compute (`ncu`)

### Hands-On Tasks
- [ ] Write a **CUDA matrix multiplication** kernel from scratch (no cuBLAS)
- [ ] Optimize it: add shared memory tiling, measure speedup
- [ ] Write a **Triton fused softmax kernel** — understand why fusion reduces memory bandwidth
- [ ] Write a **Triton fused LayerNorm kernel**
- [ ] Profile a real transformer forward pass with `torch.profiler` — find the top 3 bottlenecks
- [ ] Profile with `ncu` (Nsight Compute) — check memory bandwidth utilization and occupancy
- [ ] Replace a PyTorch operation with your custom Triton kernel in a small model — measure end-to-end speedup
- [ ] Read and annotate the Flash Attention Triton implementation

### Resources
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- "GPU Mode" YouTube lectures: https://www.youtube.com/@GPUMODE — excellent free content
- "Programming Massively Parallel Processors" (book) — Kirk & Hwu — chapters 1-6
- Nsight Systems: https://developer.nvidia.com/nsight-systems

### Papers
- **Flash Attention** — Dao et al. (2022) — read with Triton implementation open
- **Flash Attention 2** — Dao (2023)
- **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** — Tillet et al. (2019)

### Week 5 Deliverable
- GitHub repo with:
  - CUDA matmul kernel (naive + tiled versions)
  - Triton fused softmax kernel
  - Triton fused LayerNorm kernel
- Profiling report: roofline analysis of one transformer operation
- README with benchmark results (FLOPS, memory bandwidth utilization)

---

## Week 6 — Agentic AI, MLOps, and Capstone

### Goals
- Connect training + inference knowledge to agentic systems
- Get familiar with ML infra tooling used at scale
- Build a full end-to-end capstone project

### Part A — Agentic AI (Days 1-2)

#### Concepts to Study
- Agentic patterns: ReAct, Chain-of-Thought, reflection, tool use
- Tool-calling: how models are trained to call functions (function calling datasets)
- AgentInstruct: synthetic dataset generation for agentic fine-tuning
- Multi-step reasoning: planning, subgoal decomposition
- Common failure modes: hallucinated tool calls, infinite loops, reward hacking in agentic RL

#### Hands-On Tasks
- [ ] Build a simple ReAct agent using `smolagents` or `LangGraph`
- [ ] Fine-tune a small model on a tool-calling dataset (e.g., `glaiveai/glaive-function-calling`)
- [ ] Evaluate the agent on multi-step tasks — measure task completion rate
- [ ] Implement a simple rule-based reward function for agentic task completion

#### Resources
- smolagents: https://github.com/huggingface/smolagents
- LangGraph: https://langchain-ai.github.io/langgraph/
- Glaive function calling dataset: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

#### Papers
- **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al. (2022)
- **AgentInstruct: Toward Generative Teaching with Agentic Flows** — Microsoft (2024)

---

### Part B — ML Infra & MLOps (Days 3-4)

#### Concepts to Study
- Kubernetes basics: pods, deployments, services, namespaces
- GPU scheduling in K8s: NVIDIA device plugin, resource limits
- Slurm: job submission, multi-node jobs, `sbatch` scripts
- Experiment tracking: W&B, MLflow — runs, artifacts, model registry
- Data versioning: DVC
- Model serving infrastructure: what a production inference stack looks like
- Observability: latency tracking, error rates, GPU utilization dashboards

#### Hands-On Tasks
- [ ] Deploy vLLM on Kubernetes (use minikube or a cloud K8s cluster)
- [ ] Write a Slurm `sbatch` script for a multi-node training job
- [ ] Set up a full W&B project: training run → model artifact → evaluation table
- [ ] Write a simple Dockerfile for a training job
- [ ] Set up Prometheus + Grafana to monitor GPU utilization (optional but impressive)

#### Resources
- Kubernetes docs: https://kubernetes.io/docs/
- NVIDIA K8s device plugin: https://github.com/NVIDIA/k8s-device-plugin
- W&B docs: https://docs.wandb.ai
- "Designing Machine Learning Systems" (book) — Chip Huyen — chapters on deployment

---

### Part C — Capstone Project (Day 5)

Build and document a complete end-to-end project that demonstrates all phases:

#### Capstone: Train and Deploy a Specialized Agentic Model

1. **Dataset**: Build or curate a small tool-calling / agentic task dataset (~500-1000 samples)
2. **SFT**: Fine-tune Qwen2.5-1.5B or 3B with LoRA
3. **Preference optimization**: Run DPO or GRPO with a reward signal (format correctness, task completion)
4. **Inference**: Serve the final model with vLLM, expose an OpenAI-compatible API
5. **Evaluation**: Measure task completion rate, response quality, latency
6. **Write-up**: Full README with methodology, results, lessons learned

---

## Tools & Libraries Cheat Sheet

| Area | Tools |
|---|---|
| Fine-tuning | `transformers`, `trl`, `peft`, `bitsandbytes` |
| Distributed training | `deepspeed`, `accelerate`, `torch.distributed` |
| Inference serving | `vllm`, `sglang` |
| Quantization | `auto-gptq`, `autoawq`, `bitsandbytes` |
| Kernel development | `triton`, `CUDA C++`, `torch.utils.cpp_extension` |
| Profiling | `torch.profiler`, `nsys`, `ncu` |
| Experiment tracking | `wandb`, `mlflow` |
| Infrastructure | `docker`, `kubernetes`, `slurm` |
| Agentic frameworks | `smolagents`, `langgraph` |

---

## Essential Papers (Full Reading List)

### Training
- LoRA — Hu et al. (2021)
- QLoRA — Dettmers et al. (2023)
- InstructGPT — Ouyang et al., OpenAI (2022)
- DPO — Rafailov et al. (2023)
- DeepSeek-R1 — DeepSeek (2025)
- ZeRO — Rajbhandari et al. (2020)

### Inference
- vLLM / PagedAttention — Kwon et al. (2023)
- FlashAttention — Dao et al. (2022)
- FlashAttention-2 — Dao (2023)
- SGLang — Zheng et al. (2024)
- Speculative Decoding — Chen et al., DeepMind (2023)

### Agentic
- ReAct — Yao et al. (2022)
- AgentInstruct — Microsoft (2024)

---

## Cloud GPU Recommendations

| Provider | Best For | Notes |
|---|---|---|
| Vast.ai | Cheap spot GPUs | Good for experimentation |
| RunPod | Reliable on-demand | Good balance of cost/reliability |
| Lambda Labs | Longer training runs | Stable, good H100 availability |
| Google Colab Pro+ | Quick prototyping | A100 access, cheap |
| AWS / GCP | Production / multi-node | More expensive, full K8s support |

---

## Portfolio Checklist (by end of week 6)

- [ ] Fine-tuned model with LoRA/QLoRA — pushed to HuggingFace Hub
- [ ] Reward model + DPO/GRPO trained model
- [ ] Multi-GPU distributed training script (DeepSpeed + FSDP)
- [ ] vLLM + SGLang benchmark report
- [ ] Custom CUDA/Triton kernels repo
- [ ] Agentic model capstone — full pipeline end-to-end
- [ ] W&B project showing training runs and evaluation results
- [ ] Technical writeups for each major project

---

*Good luck. The gap between knowing LLMs and building the systems that train and serve them at scale is where these roles live — this plan is designed to close that gap.*
