# Week 1 — SFT: Supervised Fine-Tuning from the Ground Up

## Goals

- Deeply understand how instruction fine-tuning works
- Get hands-on with the full SFT pipeline
- Understand LoRA/QLoRA and when to use them

---

## Concepts to Study

### Causal Language Modelling vs Instruction Tuning

- **Causal LM objective**: next token prediction — the model learns to predict the next token given all previous tokens
- **Instruction tuning (SFT)**: the same objective but applied to `(instruction, response)` pairs, teaching the model to follow user intent

### Data Formatting

| Format | Structure | When to Use |
|--------|-----------|-------------|
| Alpaca | `instruction`, `input`, `output` fields | Simple Q&A / instruction following |
| ShareGPT / ChatML | `role: user/assistant` turns | Multi-turn conversations |
| System prompts | Prepended context before the conversation | Persona, constraints, tool definitions |

### Tokenisation Edge Cases

- **Padding**: sequences in a batch must be the same length — pad tokens are added and masked in the attention mask
- **Truncation**: sequences longer than `max_length` are cut — be careful not to truncate the response
- **Attention masks**: `1` = real token, `0` = padding — the model ignores padded positions

### LoRA (Low-Rank Adaptation)

Instead of updating all model weights, LoRA injects two small trainable matrices **A** and **B** into selected layers such that the update is `ΔW = BA` where the rank `r ≪ d`:

- **Rank `r`**: controls the expressivity of the adapter — typical values: 8, 16, 32
- **Alpha `α`**: scaling factor, often set to `2r` — controls how much the adapter contributes
- **Target modules**: typically `q_proj`, `v_proj` in attention layers; sometimes `up_proj`, `down_proj` in MLP
- **Trainable parameters**: only A and B matrices — massive memory saving vs full fine-tuning

### QLoRA

QLoRA combines 4-bit quantisation with LoRA:

1. **NF4 (NormalFloat4)**: a 4-bit data type optimised for normally distributed weights
2. **Double quantisation**: quantise the quantisation constants themselves — saves ~0.4 bits/parameter extra
3. **Paged optimisers**: use NVIDIA unified memory to handle memory spikes during gradient computation
4. **Practical result**: fine-tune a 7B model on a single 24 GB GPU

### SFTTrainer Internals

- **Packing**: concatenate multiple short examples into one sequence up to `max_seq_length` — improves GPU utilisation
- **Data collator**: handles padding and label masking (mask the prompt tokens so only the response drives the loss)
- **Gradient accumulation**: accumulate gradients over `N` steps before an optimiser update — simulates a larger batch size

---

## Hands-On Tasks

- [ ] Set up a clean training environment (Python 3.11, PyTorch, CUDA)
- [ ] Fine-tune **Qwen2.5-1.5B** on the Alpaca dataset using Hugging Face `trl` SFTTrainer
- [ ] Repeat the above with **LoRA** (PEFT) — compare memory usage and output quality
- [ ] Repeat with **QLoRA** (4-bit) — run on a single consumer GPU if available
- [ ] Write a script that converts a raw Q&A CSV into ShareGPT format
- [ ] Log training metrics to **Weights & Biases** (loss curves, learning rate schedule)
- [ ] Evaluate outputs manually: write 10 test prompts, compare base vs fine-tuned

---

## Resources

- [Hugging Face `trl` SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [Hugging Face `peft` docs](https://huggingface.co/docs/peft)
- [`bitsandbytes` quantisation](https://github.com/TimDettmers/bitsandbytes)
- Blog: "Fine-tuning Llama 2 with QLoRA" — Tim Dettmers

---

## Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | Introduced LoRA adapter training |
| QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers et al. | 2023 | 4-bit NF4 + LoRA on consumer GPUs |

---

## Week 1 Deliverable

A GitHub repo containing:

- Fine-tuned Qwen2.5-1.5B (LoRA adapters pushed to HuggingFace Hub)
- Training script with W&B logging
- README with loss curves, sample outputs, and GPU memory stats
