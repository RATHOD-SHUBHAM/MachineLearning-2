# LoRA and PEFT

## Say this (60–90 sec)
Full fine-tuning updates every weight — expensive in GPU memory and storage. PEFT — parameter-efficient fine-tuning — trains a small adapter while freezing the base model. LoRA is the most popular: instead of updating full weight matrix W, learn low-rank matrices A and B such that the update is B times A — rank r much smaller than dimension. W stays frozen; only A and B train. At inference, merge BA into W or apply as side path. Typical r is 8–64; targets attention projection layers. Benefits: one base model serves many LoRA adapters — swap adapters per task; train on consumer GPUs; less catastrophic forgetting. QLoRA quantizes base weights to 4-bit and trains LoRA on top — even cheaper. Adapters and prefix tuning are alternatives — small modules or virtual tokens inserted into the network. For most product fine-tunes, LoRA is the default choice.

## Why it matters
Industry standard for customizing LLMs. Shows you know practical deployment — multi-tenant adapters, memory math, not just full FT.

## How it works
- **LoRA**: `W' = W + BA`, W frozen, B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k).
- **Trainable params**: 2×r×d per targeted layer vs d×k full matrix.
- **Target modules**: usually `q_proj`, `v_proj`, sometimes all linear in attention/MLP.
- **QLoRA**: NF4 quantized base + LoRA in bf16/fp16 — fits large models on one GPU.
- **Merge**: `W + BA` for deployment without adapter overhead.

## Tradeoffs
- Use when: task-specific adaptation, limited GPU, many tasks on one base model, rapid iteration.
- Avoid when: domain shift is huge and task needs deep representation change — may need full FT or continued pretrain.

## If they dig deeper
- Rank selection — higher r more capacity, diminishing returns.
- LoRA vs adapters — LoRA modifies linear layer; adapters insert bottleneck FFN modules.
- Multi-LoRA serving — batch requests with different adapters efficiently.
