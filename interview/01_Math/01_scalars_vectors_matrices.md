# Scalars, Vectors, Matrices

## Say this (60–90 sec)
A scalar is just a single number — like a temperature or a loss value. A vector is an ordered list of numbers, often a point in space or a feature vector for one example. A matrix is a grid of numbers — rows and columns — and in ML it usually means a batch of vectors stacked together, or a weight matrix that transforms inputs. Shape matters: a vector of length d, a matrix of size m-by-n has m rows and n columns. When I read "X is n-by-d," I think n examples, d features per example. That vocabulary shows up everywhere — embeddings, weights, activations — so getting shapes right is half the battle in debugging models.

## Why it matters
Neural networks are mostly tensor operations. Misreading shapes causes silent bugs — wrong broadcasts, wrong layer sizes, gradients that don't align. Interviewers use this to check whether you think in terms of data layout, not just formulas. When someone says "the model is 768-dimensional," they mean vectors of length 768 — not a 768×768 matrix.

## How it works
- **Scalar**: one number; rank 0.
- **Vector**: `[x1, x2, ..., xd]` — one sample or one direction in d-dimensional space. Column vs row vector matters for multiplication order.
- **Matrix**: each row can be one example, each column one feature (convention varies — always clarify). Weight matrix W maps input dimension to output dimension: if input is d and output is k, W is typically k-by-d.
- **Broadcasting**: NumPy/PyTorch stretch smaller tensors to match shapes for element-wise ops — convenient but easy to misuse.
- **Transpose**: swap rows and columns — flips which dimension is "examples" vs "features" if you're not careful.
- **Identity matrix**: diagonal ones, rest zeros — multiplying by I leaves a vector unchanged; useful sanity check.

## Tradeoffs
- Use when: describing data batches, layer dimensions, or geometric intuition (vector = direction + magnitude in feature space).
- Avoid when: treating matrices as abstract symbols without tracking row/column meaning — leads to wrong implementations.

## If they dig deeper
- Row-major vs column-major storage — rarely asked, but shows you know memory layout affects cache performance.
- Tensors as generalization: scalar (0D), vector (1D), matrix (2D), batch of images (4D: N, C, H, W).
- Why we use column vectors in math texts but row vectors in some ML code — convention, not physics.
- `X @ W.T` vs `X @ W` — always trace which side holds examples and which holds output neurons.
- Sparse vs dense matrices — embeddings and one-hot encodings are sparse; most weight matrices are dense.
