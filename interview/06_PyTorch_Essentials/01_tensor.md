# Tensor (PyTorch)

Concept first: [`01_Math/02_tensors.md`](../01_Math/02_tensors.md). This note is the **framework** version.

## Say this (60–90 sec)
A PyTorch tensor is the core data structure — a multi-dimensional array like NumPy but with GPU support and autograd tracking. Three things I check on every tensor: shape, dtype, and device. Shape tells dimensions — a batch of 32 images might be 32×3×224×224 for NCHW. Dtype matters — float32 for training, float16 or bfloat16 for speed, int64 for indices. Device is CPU or cuda — operations must be on the same device or you get errors. Create tensors with torch.tensor, torch.zeros, torch.randn. Reshape with view or reshape; watch contiguity. Broadcasting applies smaller tensors to larger ones element-wise. For ML, most bugs are shape mismatches — so I print x.shape constantly. Hands-on reference: [`PyTorch/chap_1_tensors.py`](../../PyTorch/chap_1_tensors.py).

## Why it matters
Every PyTorch interview and every debugging session starts with tensors. Shape/dtype/device fluency is non-negotiable.

## How it works
- **Creation**: `torch.randn(2, 3)`, `torch.zeros_like(x)`, from NumPy `torch.from_numpy(arr)`.
- **Shape ops**: `x.view`, `x.reshape`, `x.unsqueeze`, `x.permute`, `x.squeeze`.
- **Dtype**: `.float()`, `.long()`, `.to(dtype=torch.float16)`.
- **Device**: `x.to("cuda")`, `tensor.device`, create on GPU directly when possible.
- **Indexing**: same as NumPy — slices, fancy indexing; mind shared storage with views.

## Tradeoffs
- Use when: all PyTorch computation — data, weights, activations, losses.
- Avoid when: mixing NumPy and torch without `.detach().cpu().numpy()` — breaks graph and device assumptions.

## If they dig deeper
- Contiguous memory — view requires contiguous tensor; permute may need `.contiguous()`.
- torch.compile / pinned memory — performance optimizations for DataLoader and GPU transfer.
- Named tensors (limited adoption) — explicit dimension names reduce permute bugs.
