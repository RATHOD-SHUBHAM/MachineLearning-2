# Matrix Multiply (Intuition for Neural Layers)

## Say this (60–90 sec)
Matrix multiplication is how a neural layer transforms a batch of inputs in one shot. If X is n examples by d features and W is k-by-d weights, then X times W transpose — or XW^T depending on layout — gives n-by-k outputs. Each output neuron is a dot product between one row of X and one row of W: weighted sum of inputs plus bias. So one matrix multiply replaces thousands of scalar loops. Stacking layers means chaining these transforms: each layer learns a new representation space. The key intuition: W's rows are learned templates; the layer asks "how much does this input match each template?" and passes those scores forward.

## Why it matters
Fully connected layers, attention projections, and embedding lookups all reduce to matrix multiply plus bias. GPUs are built for this. Shape literacy here separates people who debug models from people who guess. A transformer block is mostly a sequence of matrix multiplies with nonlinearities and normalization in between.

## How it works
- **Rule**: (m×n) times (n×p) gives (m×p). Inner dimensions must match.
- **One example**: output_j = Σ_i W_ji * x_i — j-th neuron combines all inputs with learned weights.
- **Batch**: each row of X is one sample; one multiply processes the whole batch — that's why batching is fast.
- **Bias**: add a vector b to each row — shifts activation thresholds.
- **Activation** (ReLU, etc.) applied element-wise after the linear transform — nonlinearity is what makes depth useful.
- **Parameter count**: k-by-d weight matrix has k×d learnable weights — easy to estimate model size.
- **Composition**: two layers `W2(W1 x)` — still linear if no activation between them; depth needs nonlinearity.

## Tradeoffs
- Use when: implementing or reasoning about linear layers, projections in transformers, efficient batch inference.
- Avoid when: dimensions don't align — never assume (n×d)(n×d) works; verify inner dimension match every time.

## If they dig deeper
- Associativity: (AB)C = A(BC) — lets us fuse ops; also (AB)^T = B^T A^T.
- Low-rank factorization: approximate big W as product of two smaller matrices — fewer params, used in LoRA/adapters.
- Why depth helps: each layer composes linear maps; with nonlinearities, the stack can represent complex functions a single layer cannot.
- GEMM (general matrix multiply) is the core BLAS operation — frameworks call highly optimized CUDA kernels for this.
- Batched matmul in attention: (batch, heads, seq, dim) @ (batch, heads, dim, seq) — same rule, higher rank tensors.
