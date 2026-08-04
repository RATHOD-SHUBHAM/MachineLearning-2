# Dot Product and Norms

## Say this (60–90 sec)
The dot product takes two vectors of the same length and returns a scalar: multiply corresponding entries and add them up. Geometrically, it's magnitude of A times magnitude of B times the cosine of the angle between them. So if vectors point the same way, dot product is large and positive; if they're orthogonal, it's zero; opposite directions give negative. That's why dot product measures similarity — embeddings with high dot product are "aligned." A norm measures vector length. L2 norm is the straight-line distance from origin — square root of sum of squares. L1 norm sums absolute values — sparser, used in regularization. We normalize vectors to unit length so dot product equals cosine similarity directly.

## Why it matters
Attention scores, linear layer outputs (weight row dotted with input), cosine similarity in retrieval, and gradient clipping all use dot products and norms. Understanding geometry beats memorizing syntax. If two embedding vectors dot to a large positive value, the model treats them as semantically similar.

## How it works
- **Dot product**: `a · b = Σ ai bi`. In matrix form: `a^T b` for column vectors.
- **L2 norm**: `||v|| = sqrt(Σ vi²)` — Euclidean length. Unit vector: divide by L2 norm.
- **L1 norm**: `Σ |vi|` — sum of absolute values; encourages sparsity in Lasso.
- **Cosine similarity**: `(a · b) / (||a|| ||b||)` — angle only, ignores magnitude; common for text embeddings.
- **Orthogonal vectors**: dot product zero — no shared direction; useful in initialization (Xavier/He).
- **Projection**: component of a along b is `(a · b / ||b||²) b` — how much of a lies in b's direction.

## Tradeoffs
- Use when: measuring similarity, computing linear layer pre-activations, normalizing for stable training.
- Avoid when: L2 dot product on unnormalized vectors — magnitude dominates; use cosine if direction matters more than scale.

## If they dig deeper
- Cauchy-Schwarz: `|a · b| ≤ ||a|| ||b||` — dot product bounded by product of lengths.
- Why weight decay penalizes L2 norm of weights — keeps parameters small, smoother decision boundaries.
- Frobenius norm for matrices: treat all entries as one big vector — used in matrix regularization.
- Gradient clipping caps global L2 norm of the gradient vector — prevents one bad batch from blowing up weights.
- Dot product as similarity only works well when vectors are normalized or magnitudes are comparable.
