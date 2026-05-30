# Week 2 — RLHF, DPO, and Reward Modelling

## Goals

- Understand the full RLHF pipeline conceptually and practically
- Train a reward model from a preference dataset
- Run DPO and GRPO fine-tuning

---

## Concepts to Study

### The RLHF Pipeline

```
Pre-trained LLM
      │
      ▼
  SFT Model ──────────────────────────────────┐
      │                                        │
      ▼                                        │
 Reward Model ◄── Human preference labels      │
  (Bradley-Terry)                              │
      │                                        │
      ▼                                        ▼
  PPO Training ◄── KL divergence penalty ◄── Reference model
      │
      ▼
 RLHF-aligned Model
```

### PPO for LLMs

PPO (Proximal Policy Optimisation) requires four models simultaneously:

| Model | Role | Trainable? |
|-------|------|-----------|
| Policy model | The model being trained | Yes |
| Value model | Estimates future reward (critic) | Yes |
| Reference model | The SFT model — frozen | No |
| Reward model | Scores outputs | No |

- **KL divergence penalty**: `reward = r(x,y) - β · KL(policy ‖ reference)` — keeps the policy from straying too far from the SFT model
- **Reward hacking**: the model learns to exploit the reward model rather than actually improve — mitigated by the KL penalty and diverse evaluation

### DPO (Direct Preference Optimisation)

DPO reframes RLHF as a classification problem — no separate reward model needed:

- Given a preference pair `(prompt, chosen, rejected)`, the model is directly optimised to assign higher probability to the chosen response
- The optimal reward function is implicitly contained in the ratio of policy to reference log-probabilities
- **Why it's popular**: simpler to implement, no instability from PPO, competitive results

### GRPO (Group Relative Policy Optimisation)

Used in DeepSeek-R1 and Qwen reasoning models:

- Sample **G** outputs for the same prompt
- Compute a reward for each output
- The advantage of output `i` is its reward relative to the group mean: `A_i = (r_i - mean(r)) / std(r)`
- No value model required — the group itself provides the baseline
- Well-suited to verifiable tasks: maths, code, structured outputs

### Preference Datasets

A preference dataset contains triplets: `(prompt, chosen, rejected)`:

- **Chosen**: the response humans preferred
- **Rejected**: the response humans did not prefer
- The margin between chosen and rejected scores matters — larger margin = cleaner signal

---

## Hands-On Tasks

- [ ] Train a **reward model** on the `Anthropic/hh-rlhf` preference dataset using a small base model
- [ ] Run **DPO** fine-tuning using `trl` DPOTrainer on a preference dataset
- [ ] Run **GRPO** fine-tuning using `trl` GRPOTrainer — use a maths reasoning task (GSM8K style)
- [ ] Compare outputs: SFT-only vs DPO vs GRPO on the same prompts
- [ ] Implement a simple rule-based reward function (e.g., reward correct format, penalise refusals)
- [ ] Log all runs to W&B, compare reward scores across training steps

---

## Key Code Patterns

### DPO Training

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sft-checkpoint")
ref_model = AutoModelForCausalLM.from_pretrained("sft-checkpoint")
tokenizer = AutoTokenizer.from_pretrained("sft-checkpoint")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(
        beta=0.1,              # KL penalty coefficient
        output_dir="./dpo-output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        report_to="wandb",
    ),
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Simple Rule-Based Reward Function

```python
def format_reward(response: str) -> float:
    """Score a response based on simple format rules."""
    score = 0.0
    if response.startswith("Sure"):
        score -= 0.5   # penalise sycophantic openers
    if len(response.split()) > 10:
        score += 0.2   # reward substantive answers
    if "I cannot" in response or "I'm unable" in response:
        score -= 1.0   # penalise unnecessary refusals
    return score
```

---

## Resources

- [`trl` DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer)
- [`trl` GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer)
- [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- Blog: "RLHF: Reinforcement Learning from Human Feedback" — HuggingFace

---

## Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|-----------------|
| InstructGPT | Ouyang et al., OpenAI | 2022 | First large-scale RLHF for instruction following |
| Direct Preference Optimisation (DPO) | Rafailov et al. | 2023 | Eliminates the need for a separate reward model |
| DeepSeek-R1 | DeepSeek | 2025 | GRPO for reasoning — read the GRPO section |

---

## Week 2 Deliverable

- Reward model uploaded to HuggingFace Hub
- DPO and GRPO fine-tuned model adapters
- Comparison writeup (markdown): reward scores, qualitative outputs, what worked
