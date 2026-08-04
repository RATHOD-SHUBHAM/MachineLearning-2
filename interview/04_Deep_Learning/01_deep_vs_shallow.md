# Deep vs Shallow Networks

## Say this (60–90 sec)
Shallow networks have one hidden layer — maybe two. Deep networks stack many layers. The difference isn't just parameter count; depth lets the network build hierarchical features. Early layers learn simple patterns, deeper layers compose them into complex concepts — edges to shapes to objects in vision. Some functions need depth: you can prove certain problems need exponentially many neurons in a shallow net but only polynomial in a deep one. In practice, depth only helps if you can train it — that's why ReLU, batch norm, residuals, and good init matter. Shallow nets can work fine on small tabular tasks or when data is limited. Deep nets shine with large datasets and structured inputs like images and text. The tradeoff is capacity vs overfitting and training difficulty.

## Why it matters
This is the "why deep learning?" question. Interviewers want hierarchical representation learning and honest limits — depth isn't free lunch without data and training tricks.

## How it works
- **Shallow**: input → one hidden layer → output. Universal approximator with enough width.
- **Deep**: repeated composition of nonlinear transforms — f = fL ∘ ... ∘ f1.
- **Hierarchical features**: layer k builds on layer k-1 (CNN: edges → textures → parts).
- **Depth efficiency**: some functions need shallow width exponential in input size but deep size polynomial.
- **Training challenge**: deeper = harder optimization without skip connections, normalization, good LR.

## Tradeoffs
- Use when: large datasets, structured high-dimensional inputs (images, speech, text), need compositional features.
- Avoid when: tiny data, simple tabular problems — shallow or classical ML may win with less tuning.

## If they dig deeper
- ResNet insight — depth with skip connections trains easier than plain deep stacks.
- Lottery ticket hypothesis — sparse subnetworks inside random init can train alone.
- Double descent — more parameters than data can still generalize in some regimes.
