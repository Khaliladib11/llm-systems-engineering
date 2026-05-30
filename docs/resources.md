# Resources — Papers, Tools & Cloud GPU Reference

## Essential Papers

### Training

| Paper | Authors | Year | Link / Notes |
|-------|---------|------|-------------|
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | Foundation for parameter-efficient fine-tuning |
| QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers et al. | 2023 | 4-bit NF4 quantisation + LoRA |
| InstructGPT: Training language models to follow instructions with human feedback | Ouyang et al., OpenAI | 2022 | First large-scale RLHF |
| Direct Preference Optimisation (DPO) | Rafailov et al. | 2023 | RLHF without a separate reward model |
| DeepSeek-R1: Incentivising Reasoning via Reinforcement Learning | DeepSeek | 2025 | GRPO for reasoning — key GRPO reference |
| ZeRO: Memory Optimisations Toward Training Trillion Parameter Models | Rajbhandari et al., Microsoft | 2020 | ZeRO Stage 1/2/3 |
| ZeRO-Offload / ZeRO-Infinity | DeepSpeed team | 2021 | CPU and NVMe offloading |
| PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | Meta | 2023 | Native PyTorch full model sharding |

### Inference

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al. | 2023 | vLLM / PagedAttention |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | 2022 | IO-aware attention kernel |
| FlashAttention-2: Faster Attention with Better Parallelism | Dao | 2023 | Improved parallelism and work partitioning |
| SGLang: Efficient Execution of Structured Language Model Programs | Zheng et al. | 2024 | RadixAttention prefix caching |
| Accelerating Large Language Model Decoding with Speculative Sampling | Chen et al., DeepMind | 2023 | Speculative decoding |

### Kernel Development

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | Tillet et al. | 2019 | Triton language and compiler |
| FlashAttention | Dao et al. | 2022 | Read with the Triton implementation open |
| FlashAttention-2 | Dao | 2023 | Improved Triton implementation |

### Agentic AI

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al. | 2022 | ReAct prompting pattern |
| AgentInstruct: Toward Generative Teaching with Agentic Flows | Microsoft | 2024 | Synthetic agentic dataset generation |

---

## Tools & Libraries Reference

### Fine-Tuning

| Library | Purpose | Docs |
|---------|---------|------|
| `transformers` | Model loading, training, tokenisation | https://huggingface.co/docs/transformers |
| `trl` | SFTTrainer, DPOTrainer, GRPOTrainer, PPOTrainer | https://huggingface.co/docs/trl |
| `peft` | LoRA, QLoRA, adapter management | https://huggingface.co/docs/peft |
| `bitsandbytes` | 4-bit / 8-bit quantisation for training | https://github.com/TimDettmers/bitsandbytes |

### Distributed Training

| Library | Purpose | Docs |
|---------|---------|------|
| `deepspeed` | ZeRO Stage 1/2/3, CPU offload | https://www.deepspeed.ai |
| `accelerate` | Unified multi-GPU / DeepSpeed / FSDP interface | https://huggingface.co/docs/accelerate |
| `torch.distributed` | PyTorch native distributed primitives | https://pytorch.org/docs/stable/distributed.html |

### Inference Serving

| Library | Purpose | Docs |
|---------|---------|------|
| `vllm` | High-throughput LLM serving with PagedAttention | https://docs.vllm.ai |
| `sglang` | Structured generation, RadixAttention | https://sgl-project.github.io |

### Quantisation

| Library | Purpose | Docs |
|---------|---------|------|
| `auto-gptq` | GPTQ post-training quantisation | https://github.com/AutoGPTQ/AutoGPTQ |
| `autoawq` | AWQ activation-aware quantisation | https://github.com/mit-han-lab/llm-awq |
| `bitsandbytes` | 4-bit / 8-bit inference quantisation | https://github.com/TimDettmers/bitsandbytes |

### Kernel Development

| Tool | Purpose | Docs |
|------|---------|------|
| `triton` | Python-level GPU kernel authoring | https://triton-lang.org |
| `CUDA C++` | Low-level GPU kernel development | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ |
| `torch.utils.cpp_extension` | Build and load custom CUDA extensions in PyTorch | https://pytorch.org/docs/stable/cpp_extension.html |

### Profiling

| Tool | Purpose | Command |
|------|---------|---------|
| `torch.profiler` | Python-level CPU + CUDA profiling | `with profile(activities=[...])` |
| `nsys` (Nsight Systems) | System-wide GPU timeline | `nsys profile python train.py` |
| `ncu` (Nsight Compute) | Per-kernel hardware metric analysis | `ncu --set full python script.py` |

### Experiment Tracking

| Tool | Purpose | Docs |
|------|---------|------|
| `wandb` | Runs, metrics, artifacts, model registry | https://docs.wandb.ai |
| `mlflow` | Open-source alternative to W&B | https://mlflow.org/docs |

### Infrastructure

| Tool | Purpose | Docs |
|------|---------|------|
| `docker` | Containerise training and serving jobs | https://docs.docker.com |
| `kubernetes` | Orchestrate containerised workloads | https://kubernetes.io/docs/ |
| `slurm` | HPC job scheduler for multi-node training | https://slurm.schedmd.com/documentation.html |

### Agentic Frameworks

| Library | Purpose | Docs |
|---------|---------|------|
| `smolagents` | Lightweight agentic framework from HuggingFace | https://github.com/huggingface/smolagents |
| `langgraph` | Graph-based agentic workflows | https://langchain-ai.github.io/langgraph/ |

---

## Cloud GPU Recommendations

| Provider | Best For | GPU Options | Notes |
|----------|----------|-------------|-------|
| [Vast.ai](https://vast.ai) | Cheap spot GPUs | RTX 3090, 4090, A100 | Best price for experimentation; spot pricing |
| [RunPod](https://runpod.io) | Reliable on-demand | RTX 4090, A100, H100 | Good balance of cost and reliability |
| [Lambda Labs](https://lambdalabs.com) | Longer training runs | A100, H100, A10 | Stable; good H100 availability |
| [Google Colab Pro+](https://colab.research.google.com) | Quick prototyping | A100, V100 | Cheapest entry; not suited for long runs |
| [AWS EC2 (p4/p5)](https://aws.amazon.com/ec2/instance-types/p4/) | Production / multi-node | A100, H100 | Full K8s support; expensive |
| [GCP (a3)](https://cloud.google.com/compute/docs/gpus) | Production / multi-node | H100 | Full K8s support; expensive |

### Cost Estimates (approximate, 2025)

| GPU | Provider | $/hour |
|-----|----------|--------|
| RTX 4090 (24 GB) | Vast.ai | ~$0.30–0.60 |
| A100 80 GB | RunPod | ~$2.00–2.50 |
| H100 80 GB | Lambda Labs | ~$2.50–3.50 |
| H100 80 GB | AWS p5.48xlarge / 8× H100 | ~$32 (on-demand) |

---

## Datasets Reference

| Dataset | Task | Link |
|---------|------|------|
| `tatsu-lab/alpaca` | SFT instruction following | HuggingFace Hub |
| `Anthropic/hh-rlhf` | Preference / reward modelling | HuggingFace Hub |
| `openai/gsm8k` | Maths reasoning (GRPO) | HuggingFace Hub |
| `glaiveai/glaive-function-calling-v2` | Tool calling / agentic SFT | HuggingFace Hub |

---

## Books

| Book | Authors | Chapters to Focus On |
|------|---------|---------------------|
| *Programming Massively Parallel Processors* | Kirk & Hwu | 1–6 (CUDA fundamentals) |
| *Designing Machine Learning Systems* | Chip Huyen | Deployment and serving chapters |
