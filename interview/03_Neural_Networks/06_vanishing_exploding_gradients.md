# Vanishing / Exploding Gradients

## Say this (60–90 sec)
In deep networks, gradients multiply as they flow backward through layers. If each layer shrinks the gradient — say sigmoid derivatives near zero — the signal vanishes in early layers and they stop learning. That's vanishing gradients. If layers amplify it, gradients explode — weights become NaN, training diverges. Sigmoid and tanh in deep stacks were notorious for vanishing. ReLU helps on the positive side because gradient is one. LSTM/GRU gates were designed to preserve gradient flow over long sequences. Residual connections add a skip path so gradients have a highway. Batch norm stabilizes scale. Good initialization keeps activations in a reasonable range. In practice I watch gradient norms, clip if needed, and use architectures designed for depth — ResNet, transformers with layer norm. Exploding is easier to spot; vanishing shows up as early layers not changing.

## Why it matters
This explains why depth was hard before ReLU, batch norm, and residual nets — and why RNNs needed LSTM. Interviewers want to know you diagnose training failures, not just blame "bad data."

## How it works
- **Chain rule product**: gradient through L layers ≈ product of L Jacobian terms — repeated shrink → vanish, repeated grow → explode.
- **Sigmoid/tanh**: saturate at |z| large → derivative ≈ 0 → vanishing in deep hidden layers.
- **ReLU**: gradient 0 for z < 0 (dead neuron), 1 for z > 0 — no shrink on active path.
- **RNNs**: same weight matrix applied repeatedly — eigenvalues > 1 explode, < 1 vanish over time steps.
- **Fixes**: ReLU/GELU, residual/skip connections, layer norm, LSTM/GRU, gradient clipping, careful init.

## Tradeoffs
- Use when: explaining why deep sigmoid MLPs fail, why ResNets train deeper, or why plain RNNs forget long context.
- Avoid when: blaming vanishing gradients for every slow train — could be LR, bad init, or data issues; diagnose first.

## If they dig deeper
- Gradient clipping: cap global norm (e.g., 1.0) — standard in RNN/transformer training.
- Highway and dense connections — generalization of residuals.
- Spectral normalization — controls Lipschitz constant of layers to stabilize GANs.
