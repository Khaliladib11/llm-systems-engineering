# LLM Systems Engineering — Training & Inference Bootcamp

> **Target roles:** LLM Training Expert · ML Infra Engineer · Inference Engineer · GPU Optimisation Engineer

A 6-week intensive self-study plan covering the full stack of building, training, and serving large language models at scale. Assumes solid Python, cloud computing, and general LLM familiarity.

---

## Structure

- **5 days/week**, ~6–8 hours/day
- Each week has: concepts, hands-on tasks, papers, and a deliverable
- By the end you will have a portfolio of real projects covering training, inference, and kernel optimisation

---

## Curriculum Overview

| Week | Topic | Key Skills |
|------|-------|------------|
| [Week 1](docs/week-1-sft.md) | Supervised Fine-Tuning (SFT) | LoRA, QLoRA, SFTTrainer, W&B |
| [Week 2](docs/week-2-rlhf-dpo.md) | RLHF, DPO & Reward Modelling | PPO, DPO, GRPO, preference datasets |
| [Week 3](docs/week-3-distributed-training.md) | Distributed Training at Scale | DeepSpeed ZeRO, FSDP, accelerate |
| [Week 4](docs/week-4-inference.md) | Inference: vLLM, SGLang & Serving | PagedAttention, quantisation, speculative decoding |
| [Week 5](docs/week-5-cuda-triton.md) | CUDA, Triton & Kernel Optimisation | Custom kernels, Triton, profiling |
| [Week 6](docs/week-6-agentic-mlops-capstone.md) | Agentic AI, MLOps & Capstone | ReAct, K8s, Slurm, end-to-end project |

---

## Portfolio Checklist

By the end of Week 6 you should have:

- [ ] Fine-tuned model with LoRA/QLoRA — pushed to HuggingFace Hub
- [ ] Reward model + DPO/GRPO trained model
- [ ] Multi-GPU distributed training script (DeepSpeed + FSDP)
- [ ] vLLM + SGLang benchmark report
- [ ] Custom CUDA/Triton kernels repo
- [ ] Agentic model capstone — full pipeline end-to-end
- [ ] W&B project showing training runs and evaluation results
- [ ] Technical writeups for each major project

---

## Tools & Libraries

| Area | Tools |
|------|-------|
| Fine-tuning | `transformers`, `trl`, `peft`, `bitsandbytes` |
| Distributed training | `deepspeed`, `accelerate`, `torch.distributed` |
| Inference serving | `vllm`, `sglang` |
| Quantisation | `auto-gptq`, `autoawq`, `bitsandbytes` |
| Kernel development | `triton`, `CUDA C++`, `torch.utils.cpp_extension` |
| Profiling | `torch.profiler`, `nsys`, `ncu` |
| Experiment tracking | `wandb`, `mlflow` |
| Infrastructure | `docker`, `kubernetes`, `slurm` |
| Agentic frameworks | `smolagents`, `langgraph` |

---

## Cloud GPU Recommendations

| Provider | Best For | Notes |
|----------|----------|-------|
| Vast.ai | Cheap spot GPUs | Good for experimentation |
| RunPod | Reliable on-demand | Good balance of cost/reliability |
| Lambda Labs | Longer training runs | Stable, good H100 availability |
| Google Colab Pro+ | Quick prototyping | A100 access, cheap |
| AWS / GCP | Production / multi-node | More expensive, full K8s support |

---

## Documentation

Detailed notes and learning materials for each week live in the [`docs/`](docs/) folder:

- [`docs/week-1-sft.md`](docs/week-1-sft.md) — Supervised Fine-Tuning
- [`docs/week-2-rlhf-dpo.md`](docs/week-2-rlhf-dpo.md) — RLHF, DPO & Reward Modelling
- [`docs/week-3-distributed-training.md`](docs/week-3-distributed-training.md) — Distributed Training
- [`docs/week-4-inference.md`](docs/week-4-inference.md) — Inference & Serving
- [`docs/week-5-cuda-triton.md`](docs/week-5-cuda-triton.md) — CUDA & Triton Kernels
- [`docs/week-6-agentic-mlops-capstone.md`](docs/week-6-agentic-mlops-capstone.md) — Agentic AI, MLOps & Capstone
- [`docs/resources.md`](docs/resources.md) — Full papers list, tools reference & cloud GPU guide

---

*The gap between knowing LLMs and building the systems that train and serve them at scale is where these roles live — this plan is designed to close that gap.*
