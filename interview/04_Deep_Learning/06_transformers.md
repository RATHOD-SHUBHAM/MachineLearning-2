# Transformers

## Say this (60–90 sec)
Transformers are encoder-decoder or decoder-only stacks built from self-attention and feedforward layers, with no recurrence. Input tokens get embedded, plus positional encoding so order is known — attention itself is permutation-invariant. Each block: multi-head self-attention, residual connection, layer norm, MLP, residual, layer norm. Pre-norm vs post-norm varies by architecture. Encoder sees full sequence bidirectionally — BERT-style masked language modeling. Decoder is causal — each token only attends to past tokens — GPT-style next-token prediction. Transformers parallelize across sequence length during training, unlike RNNs. Scale is the story: bigger models, more data, more compute. LLMs are mostly decoder-only transformers. Vision transformers patchify images and treat patches as tokens. Key hyperparameters: layers, heads, hidden dim, context length.

## Why it matters
Dominant architecture for NLP, vision, multimodal. Interviewers expect block diagram fluency and encoder vs decoder vs encoder-decoder distinctions.

## How it works
- **Embedding + position**: token embeddings + learned or sinusoidal position encodings.
- **Encoder block**: self-attention (full visibility) + FFN (two linear layers with GELU/ReLU).
- **Decoder block**: masked self-attention + cross-attention to encoder + FFN.
- **Decoder-only (GPT)**: stacked masked self-attention blocks; autoregressive generation.
- **Residual + LayerNorm**: stabilizes deep training; Pre-LN common in modern LLMs.

## Tradeoffs
- Use when: NLP, code, multimodal, long-context with efficient attention; scale to large data.
- Avoid when: tiny data, strict latency on CPU, need streaming with tiny memory — smaller RNN/CNN may suffice.

## If they dig deeper
- BERT (encoder) vs GPT (decoder) vs T5 (encoder-decoder) — pretraining objective drives use case.
- KV cache at inference — cache key/value for past tokens; only compute new token each step.
- Context length limits — quadratic memory; RoPE, ALiBi, sliding window extend context.
