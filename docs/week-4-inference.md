# Week 4 — Inference: vLLM, SGLang, and Serving

## Goals

- Deploy models efficiently at scale
- Understand the internals of modern inference engines
- Benchmark and optimise serving performance

---

## Concepts to Study

### Why Naive Inference is Slow

Autoregressive generation produces one token at a time. Each step:
1. Loads the entire model from GPU HBM
2. Computes attention over all previous tokens (the KV cache)
3. Writes one new token

This is **memory-bandwidth bound**, not compute bound — the GPU spends most of its time reading weights, not performing FLOPs.

### KV Cache

During attention, each token produces a **Key** and **Value** vector. These are cached so they don't need to be recomputed for previous tokens:

- Memory usage grows linearly with sequence length and batch size
- A 7B model with 4096 context length can use several GB of KV cache per request
- Naive allocation (one contiguous block per sequence) causes severe **fragmentation**

### PagedAttention (vLLM)

Inspired by virtual memory in operating systems:

- KV cache is divided into fixed-size **pages** (blocks)
- A **block table** maps logical positions to physical blocks
- Pages from different requests can be non-contiguous in memory
- Pages can be **shared** across requests that share a common prefix (e.g., system prompt)
- Eliminates fragmentation — near-zero wasted KV cache memory

### Continuous Batching

Traditional batching waits for all requests in a batch to finish before starting new ones — short requests waste GPU time waiting for long ones to complete.

Continuous batching inserts new requests into free slots as soon as any request finishes, keeping the GPU busy at all times.

### Tensor Parallelism in Inference

- Attention heads are split across GPUs — each GPU handles a subset of heads
- MLP layers are column/row split — outputs are all-reduced before the next layer
- All-reduce communication at every layer — best on NVLink-connected GPUs within one node

### Speculative Decoding

- A small **draft model** generates `k` tokens speculatively
- The large **verifier model** checks all `k` tokens in a single forward pass
- Accepted tokens are kept; rejected tokens trigger a rollback
- No loss in output quality — the output distribution is identical to sampling from the large model alone
- Speedup: 2–3× for short drafts on latency-sensitive workloads

### Flash Attention

The standard attention algorithm reads/writes the full N×N attention matrix to HBM (slow). Flash Attention:

1. Tiles Q, K, V into blocks that fit in SRAM
2. Computes softmax incrementally using the online normalisation trick
3. Never materialises the full attention matrix in HBM
4. **Recomputation trick**: discards intermediate results, recomputes them during the backward pass

Result: same mathematical output, much less HBM bandwidth, O(N) memory instead of O(N²).

### Quantisation for Inference

| Method | Type | Notes |
|--------|------|-------|
| GPTQ | Post-training, weight-only | Minimises quantisation error per layer |
| AWQ (Activation-Aware) | Post-training, weight-only | Scales weights by activation importance — better quality than GPTQ |
| fp8 | Hardware-native | Supported on H100 — near fp16 quality at half the memory |

### RadixAttention (SGLang)

SGLang uses a **radix tree** to cache KV entries keyed by token prefixes:

- Requests sharing the same prefix (e.g., the same system prompt) share the cached KV blocks
- Huge speedup for multi-turn conversations and RAG workloads with repeated context

### Key Metrics to Track

| Metric | What it Measures | Target |
|--------|-----------------|--------|
| TTFT (Time to First Token) | Prefill latency | < 500 ms for interactive use |
| TPOT (Time Per Output Token) | Decode latency per token | < 30 ms/token |
| Throughput | Total tokens/sec across all requests | Maximise for batch workloads |
| GPU memory utilisation | KV cache + model weights | 85–95% |

---

## Hands-On Tasks

- [ ] Deploy a 7B model with **vLLM** — OpenAI-compatible API server
- [ ] Benchmark vLLM: measure TTFT (time to first token) and throughput (tokens/sec) at different batch sizes
- [ ] Enable **tensor parallelism** in vLLM across 2 GPUs
- [ ] Deploy the same model with **SGLang** — benchmark vs vLLM
- [ ] Serve a model with **AWQ quantisation** — compare throughput and quality vs fp16
- [ ] Implement a simple load test script (concurrent requests, measure latency percentiles)
- [ ] Enable **speculative decoding** in vLLM (use a small draft model)
- [ ] Experiment with different `max_num_seqs` and `gpu_memory_utilization` settings

---

## Resources

- [vLLM docs](https://docs.vllm.ai)
- [SGLang docs](https://sgl-project.github.io)
- [AWQ — MIT HAN Lab](https://github.com/mit-han-lab/llm-awq)
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- [vLLM blog post (original)](https://vllm.ai/blog/2023/06/20/vllm.html)

---

## Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al. | 2023 | vLLM / PagedAttention |
| FlashAttention | Dao et al. | 2022 | IO-aware attention |
| FlashAttention-2 | Dao | 2023 | Better parallelism and work partitioning |
| SGLang: Efficient Execution of Structured LM Programs | Zheng et al. | 2024 | RadixAttention prefix caching |
| Accelerating LLM Decoding with Speculative Sampling | Chen et al., DeepMind | 2023 | Speculative decoding |

---

## Week 4 Deliverable

- Deployed vLLM + SGLang serving setup
- Benchmark report: latency (p50, p95, p99), throughput, GPU memory across quantisation levels
- Comparison table: vLLM vs SGLang on the same model and hardware
