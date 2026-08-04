# Convolutional Neural Networks (CNN)

## Say this (60–90 sec)
CNNs are built for grid data like images. Convolution slides a small filter over the input, computing local dot products — same weights everywhere, so it's translation-equivariant: a cat shifted left still gets detected. Each filter learns a pattern — edge, color blob, texture. Stacking conv layers builds hierarchical features. Pooling downsamples — max pool takes the max in a window, reducing spatial size and adding local invariance to small shifts. Receptive field is the region of the input one neuron "sees" — it grows with depth and kernel size. Early layers see pixels; deep layers see large image patches. Parameter sharing keeps CNNs efficient vs fully connected layers on raw pixels. Modern stacks: conv blocks, batch norm, ReLU, maybe residual connections, global average pool, linear classifier. CNNs dominated vision before ViT; still strong baselines and backbones for detection/segmentation.

## Why it matters
CNNs explain inductive bias — locality and translation invariance. Core for vision interviews and understanding why transformers added patch embeddings.

## How it works
- **Conv layer**: output[i,j] = sum over kernel of input patch × filter weights + bias. Output channels = number of filters.
- **Stride**: step size of the filter — stride 2 halves spatial dimensions.
- **Padding**: pad borders so output size is preserved (same padding).
- **Pooling**: max or average over windows — downsamples H×W, keeps channels.
- **Receptive field**: RF grows with layer depth, kernel size, stride — deep units integrate global context.
- **Typical shape**: input (N, C, H, W) — batch, channels, height, width.

## Tradeoffs
- Use when: images, video frames, spectrograms, any spatially local structure.
- Avoid when: pure tabular or unordered token sequences — 1D conv possible but transformers often win on text.

## If they dig deeper
- 1×1 conv — channel mixing without spatial blur (Network-in-Network, ResNet bottlenecks).
- Depthwise separable conv — MobileNet efficiency trick.
- ViT vs CNN — ViT needs more data; CNN inductive bias helps small/medium datasets.
