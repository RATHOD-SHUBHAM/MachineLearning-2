# RLHF

## Say this (60–90 sec)
RLHF aligns LLMs with human preferences — helpful, harmless, honest — beyond raw next-token prediction. Typical three stages: one, supervised fine-tuning on high-quality demonstrations. Two, train a reward model on human comparisons — humans pick better of two responses, reward model learns to score outputs. Three, optimize the policy LLM with RL — usually PPO — to maximize reward while staying close to the SFT model via KL penalty so it doesn't drift into gibberish. The reward model is the proxy for human judgment — imperfect and gameable. DPO and similar methods skip explicit RL by directly optimizing on preference pairs — simpler and stable. RLHF is why chat models feel better than base LMs — they refuse harmful requests, follow instructions, sound natural. Cost is human labeling and training complexity.

## Why it matters
Explains the gap between base model and ChatGPT-style assistant. Alignment and safety interviews reference RLHF, reward hacking, and alternatives.

## How it works
- **SFT**: fine-tune on curated (prompt, ideal response) pairs.
- **Reward model (RM)**: Bradley-Terry / ranking loss on chosen vs rejected responses.
- **PPO phase**: sample responses, RM scores them, policy gradient update + KL to reference model.
- **KL penalty**: prevents policy from collapsing to high-reward but low-quality text.
- **DPO**: directly optimize preferences without separate RM + PPO loop.

## Tradeoffs
- Use when: chat assistants, safety-critical products, behavior must match human preferences not just likelihood.
- Avoid when: narrow task with clear metric — supervised fine-tune alone may suffice; RLHF is expensive.

## If they dig deeper
- Reward hacking — model exploits RM weaknesses (verbose, sycophantic answers).
- Constitutional AI / RLAIF — AI-generated preferences reduce human label cost.
- Alignment tax — RLHF can slightly hurt raw capability on some benchmarks.
