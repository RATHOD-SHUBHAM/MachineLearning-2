# Attention

## Say this (60–90 sec)
Attention lets a model focus on relevant parts of the input when producing each output. Instead of compressing a whole sequence into one fixed vector, we compute a weighted sum of all input representations — weights depend on the current query. Scaled dot-product attention: take query Q, keys K, values V — scores are Q times K transpose, scaled by sqrt of dimension, softmax to get weights, multiply by V. High score means "pay attention here." In seq2seq, the decoder query attends over encoder outputs. Self-attention uses the same sequence for Q, K, V — each token attends to all others, capturing context in one layer. Multi-head attention runs several attention heads in parallel — different subspaces, then concatenate. Attention is O(n²) in sequence length but highly parallelizable on GPU — unlike RNNs. It's the core idea behind transformers.

## Why it matters
Attention replaced recurrence for NLP and beyond. Must explain Q/K/V intuition and why it's better than a single bottleneck vector.

## How it works
- **Query, Key, Value**: learned linear projections of input embeddings.
- **Scores**: `softmax(QK^T / √d_k) V` — softmax over keys (which positions matter).
- **Self-attention**: Q, K, V all from same sequence — pairwise token relationships.
- **Cross-attention**: Q from decoder, K/V from encoder — machine translation pattern.
- **Multi-head**: h parallel heads, each d_k dimensional; concat + project.

## Tradeoffs
- Use when: need long-range dependencies, parallel training, interpretable alignment (which tokens attended).
- Avoid when: very long sequences without sparse/linear attention — O(n²) memory and compute hurt.

## If they dig deeper
- Bahdanau vs Luong attention — additive vs dot-product (early seq2seq).
- Causal/masked attention — block future tokens in decoder (autoregressive LM).
- Linear attention, FlashAttention — efficiency optimizations for long context.
