# Week 6 — Agentic AI, MLOps, and Capstone

## Goals

- Connect training + inference knowledge to agentic systems
- Get familiar with ML infra tooling used at scale
- Build a full end-to-end capstone project

---

## Part A — Agentic AI (Days 1–2)

### Concepts to Study

#### Agentic Patterns

| Pattern | Description |
|---------|-------------|
| **ReAct** | Interleave reasoning (`Thought:`) with actions (`Action:`) — the model thinks before acting |
| **Chain-of-Thought** | Produce step-by-step reasoning before the final answer |
| **Reflection** | The model critiques its own output and revises it |
| **Tool use** | The model outputs structured calls to external functions (search, code exec, APIs) |

#### Tool Calling

- Models are fine-tuned on datasets where the response includes structured JSON matching a function schema
- The schema (name, parameters, descriptions) is injected into the system prompt
- Common training datasets: `glaiveai/glaive-function-calling-v2`, NexusRaven, ToolBench
- Function calling is just SFT — the only difference is the output format

#### AgentInstruct

Microsoft's approach to synthetic agentic dataset generation:
- Uses a seed set of tasks and a generator model to create diverse, high-quality agentic trajectories
- Covers tool use, multi-step planning, and self-correction
- Key insight: synthetically generated agentic data can match human-annotated data in quality

#### Common Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|-----------|
| Hallucinated tool calls | Model invents functions that do not exist | Strict schema validation, retry with error feedback |
| Infinite loops | No termination condition | Max step limit, progress detection |
| Reward hacking (agentic RL) | Agent finds shortcuts that score well but don't solve the task | Process reward models, step-level verification |

### Hands-On Tasks

- [ ] Build a simple ReAct agent using `smolagents` or `LangGraph`
- [ ] Fine-tune a small model on a tool-calling dataset (e.g., `glaiveai/glaive-function-calling`)
- [ ] Evaluate the agent on multi-step tasks — measure task completion rate
- [ ] Implement a simple rule-based reward function for agentic task completion

### Key Code Pattern: ReAct Agent with smolagents

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

model = HfApiModel(model_id="Qwen/Qwen2.5-7B-Instruct")

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=10,
)

result = agent.run("Find the latest research on speculative decoding for LLMs.")
```

### Resources

- [smolagents](https://github.com/huggingface/smolagents)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Glaive function calling dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

### Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al. | 2022 | ReAct prompting framework |
| AgentInstruct: Toward Generative Teaching with Agentic Flows | Microsoft | 2024 | Synthetic agentic dataset generation |

---

## Part B — ML Infra & MLOps (Days 3–4)

### Concepts to Study

#### Kubernetes Basics for ML

| Concept | Relevance |
|---------|-----------|
| **Pods** | Smallest deployable unit — one container per GPU node |
| **Deployments** | Manage replicas of an inference server |
| **Services** | Load balance across pods — expose the inference API |
| **Namespaces** | Isolate different environments (dev, prod) |
| **NVIDIA Device Plugin** | Exposes GPUs as a schedulable resource — `nvidia.com/gpu: 1` |

#### Slurm for Multi-Node Training

```bash
#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8        # 8 GPUs per node
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  train.py
```

#### Experiment Tracking with W&B

- **Runs**: each training job is a run — logs metrics, config, system stats
- **Artifacts**: versioned datasets and model checkpoints
- **Model Registry**: production-grade model versioning and promotion workflow
- **Evaluation tables**: log predictions alongside ground truth for qualitative analysis

#### Observability Stack

- **Prometheus**: scrapes metrics from inference servers (request rate, latency histograms, GPU utilisation)
- **Grafana**: visualises Prometheus metrics in dashboards
- **vLLM metrics endpoint**: exposes `/metrics` in Prometheus format out of the box

### Hands-On Tasks

- [ ] Deploy vLLM on Kubernetes (use minikube or a cloud K8s cluster)
- [ ] Write a Slurm `sbatch` script for a multi-node training job
- [ ] Set up a full W&B project: training run → model artifact → evaluation table
- [ ] Write a simple Dockerfile for a training job
- [ ] Set up Prometheus + Grafana to monitor GPU utilisation (optional but impressive)

### Key Code Pattern: Dockerfile for Training

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "train.py"]
```

### Key Code Pattern: vLLM on Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "meta-llama/Llama-3-8B-Instruct"
            - "--gpu-memory-utilization"
            - "0.9"
          resources:
            limits:
              nvidia.com/gpu: "1"
          ports:
            - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### Resources

- [Kubernetes docs](https://kubernetes.io/docs/)
- [NVIDIA K8s device plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [W&B docs](https://docs.wandb.ai)
- *Designing Machine Learning Systems* — Chip Huyen — chapters on deployment

---

## Part C — Capstone Project (Day 5)

Build and document a complete end-to-end project demonstrating all phases of the bootcamp.

### Capstone: Train and Deploy a Specialised Agentic Model

#### Step 1 — Dataset

Build or curate a small tool-calling / agentic task dataset (~500–1000 samples):

- Source from `glaiveai/glaive-function-calling-v2` or generate synthetically
- Include: single-tool calls, multi-step tool chains, error recovery examples
- Convert to ShareGPT / ChatML format

#### Step 2 — SFT

Fine-tune **Qwen2.5-1.5B** or **3B** with LoRA:

- Use `trl` SFTTrainer with packing enabled
- Log to W&B — track loss, learning rate, gradient norm
- Push LoRA adapters to HuggingFace Hub

#### Step 3 — Preference Optimisation

Run DPO or GRPO with a verifiable reward signal:

- **Format correctness**: does the model output valid JSON matching the function schema?
- **Task completion**: does the tool call actually solve the stated task?
- Use GRPO if the reward is binary/sparse — group sampling stabilises training

#### Step 4 — Inference

Serve the final merged model with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./final-model \
  --port 8000
```

Expose an OpenAI-compatible API — test with the `openai` Python client.

#### Step 5 — Evaluation

| Metric | How to Measure |
|--------|---------------|
| Task completion rate | % of test tasks where the agent produces the correct final answer |
| Response quality | Human eval or LLM-as-judge on 50 samples |
| TTFT | Median time to first token under load |
| Throughput | Tokens/sec at 10, 50, 100 concurrent users |

#### Step 6 — Write-up

Full README covering:

- Problem statement and dataset description
- SFT training details (model, LoRA config, hyperparameters)
- Preference optimisation method and reward function
- Serving setup and benchmark results
- Lessons learned and what you would do differently

---

## Week 6 Deliverable

- ReAct agent demo (notebook or script)
- Capstone model fine-tuned and served with vLLM
- Full evaluation report
- Kubernetes deployment manifest (or Docker Compose equivalent)
- W&B project link showing the full training lineage
