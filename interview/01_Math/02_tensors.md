# Tensor

## Say this (60–90 sec)
A tensor is a multi-dimensional array of numbers — the general form of scalar, vector, and matrix. Rank (or order) is how many axes it has: a scalar is rank 0, a vector rank 1, a matrix rank 2. An image batch is often rank 4: batch, channels, height, width — written N×C×H×W. Shape is the size along each axis; if someone says “shape is (32, 768),” that’s 32 vectors of length 768. In deep learning almost everything is a tensor — inputs, weights, activations, losses. The math idea is the same whether you use NumPy or PyTorch; frameworks just add GPU and gradients. When I debug, I always print shape first — most model bugs are shape mismatches, not fancy math.

## Why it matters
Interviewers expect you to speak tensors fluently: rank, shape, and what each axis means. Saying “a 3D tensor of embeddings” without knowing axes is a red flag.

## How it works
- **Rank 0**: scalar — one loss value.
- **Rank 1**: vector — one feature vector or one embedding.
- **Rank 2**: matrix — a batch of feature vectors, or a weight matrix.
- **Rank 3+**: sequences (batch × time × features), images (N×C×H×W), video, etc.
- **Shape**: tuple of sizes per axis — must match for matmul and most ops.
- **Dtype / device** (in frameworks): float32 vs int64, CPU vs GPU — same concept, different hardware.

## Tradeoffs
- Use when: describing any multi-dimensional data or model state in ML/DL.
- Avoid when: calling every array a “tensor” without stating shape and what each dimension means — vague answers get follow-ups.

## If they dig deeper
- Tensor vs NumPy ndarray: same mental model; PyTorch tensors can live on GPU and track gradients.
- Broadcasting: smaller shapes expand to match — powerful, easy to misuse.
- Contiguous memory / view vs reshape — framework detail; see [`06_PyTorch_Essentials/01_tensor.md`](../06_PyTorch_Essentials/01_tensor.md).
- Hands-on: [`PyTorch/chap_1_tensors.py`](../../PyTorch/chap_1_tensors.py).
