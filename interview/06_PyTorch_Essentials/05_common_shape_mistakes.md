# Common Shape Mistakes

## Say this (60–90 sec)
Most PyTorch bugs are shape errors. Classic ones: forgetting to flatten before Linear — conv output (N, C, H, W) needs view(N, -1) or adaptive avg pool. Wrong loss input shapes — CrossEntropyLoss expects logits (N, C) and class indices (N,) long dtype, not one-hot. Mixing NCHW vs NHWC — PyTorch conv expects channels first. Batch dimension missing — single sample (3, 224, 224) should be (1, 3, 224, 224). Transpose confusion — (N, seq, dim) vs (seq, N, dim) for RNNs; transformers usually (N, seq, dim). Broadcasting surprises — (N, 1) + (N, C) works but (N,) + (N, C) may not. Matrix multiply order — (N, d) @ (d, k) not (d, N) @ (d, k). My debug ritual: print shapes at layer boundaries, draw dimensions on paper, use assert x.shape == expected early in forward.

## Why it matters
Shape fluency separates people who copy tutorials from people who ship models. Interviewers often give a broken snippet and ask you to spot the error.

## How it works
- **Linear after conv**: `x = x.view(x.size(0), -1)` or `nn.AdaptiveAvgPool2d(1)` then flatten.
- **CrossEntropyLoss**: logits `(N, num_classes)`, target `(N,)` dtype long, values 0..C-1.
- **BCEWithLogitsLoss**: logits and targets same shape, targets float 0/1.
- **Permute for images**: if data is NHWC, `x = x.permute(0, 3, 1, 2)`.
- **Unsqueeze for batch**: `x.unsqueeze(0)` adds batch dim.

## Tradeoffs
- Use when: debugging any dimension mismatch — always trace batch and feature axes explicitly.
- Avoid when: blindly adding `.squeeze()` — removes all size-1 dims and may drop batch accidentally.

## If they dig deeper
- einops.rearrange — readable reshape/permute reduces bugs.
- torch.Size vs tuple — negative indexing in view (-1 infer dim).
- ONNX/export shape errors — dynamic axes for variable batch/seq length.
