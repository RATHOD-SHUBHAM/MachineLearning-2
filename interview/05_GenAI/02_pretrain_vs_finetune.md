# Pretrain vs Fine-Tune

## Say this (60–90 sec)
Pretraining teaches a model general language or vision knowledge from massive unlabeled or weakly labeled data. For LLMs, that's usually next-token prediction on internet-scale text — the model learns grammar, facts, reasoning patterns. Fine-tuning adapts that general model to a specific task or behavior with smaller labeled or curated data. Full fine-tuning updates all weights — expensive and can forget pretrain knowledge. Parameter-efficient fine-tuning — LoRA, adapters — updates a small subset. Instruction tuning is fine-tuning on prompt-response pairs so the model follows instructions. RLHF is another fine-tuning stage for alignment. Rule of thumb: pretrain once at huge cost; fine-tune cheaply per product. You rarely pretrain from scratch unless you're a foundation model lab. Most teams start from an open checkpoint and fine-tune or prompt.

## Why it matters
Core GenAI workflow distinction. Shows you understand cost, data, and when prompting beats training.

## How it works
- **Pretrain objective**: causal LM (predict next token), masked LM (BERT), or multimodal variants.
- **Fine-tune**: supervised labels — classification heads, QA, summarization, chat format.
- **Instruction tuning**: (instruction, response) pairs — SFT before RLHF in chat models.
- **Catastrophic forgetting**: aggressive full FT on narrow data hurts general ability.
- **PEFT**: LoRA/adapters — train low-rank deltas; base weights frozen.

## Tradeoffs
- Use when: fine-tune for domain-specific tone, format, or task; pretrain only with huge compute + data.
- Avoid when: fine-tuning for knowledge that changes daily — RAG is cheaper and updatable; don't fine-tune what retrieval can fix.

## If they dig deeper
- Continued pretraining — more LM on domain corpus before task fine-tune (domain adaptation).
- Multi-task fine-tune vs single-task — trade specialization vs versatility.
- Data contamination — benchmark leakage in pretrain corpus inflates eval scores.
