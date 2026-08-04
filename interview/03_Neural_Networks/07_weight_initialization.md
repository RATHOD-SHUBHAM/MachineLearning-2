# Weight Initialization

## Say this (60–90 sec)
Weight initialization sets the starting point before training. Bad init can kill learning before epoch one — all sigmoids saturated, all ReLUs dead, or activations exploding layer by layer. The goal is to keep variance stable across layers so gradients and activations stay in a useful range. Xavier or Glorot init is for sigmoid/tanh: scale weights by one over sqrt of fan-in plus fan-out. He or Kaiming init is for ReLU: scale by sqrt of two over fan-in because ReLU zeros half the activations. Biases usually start at zero. Don't initialize all weights to zero — neurons would stay identical forever, symmetry never breaks. In PyTorch, `nn.Linear` defaults are often Kaiming uniform. For deep nets, init plus batch norm or layer norm is the combo that makes depth trainable from random start.

## Why it matters
Init is a one-line change that can make or break convergence. Shows you understand signal propagation through depth, not just "random weights are fine."

## How it works
- **Zero init (weights)**: bad — symmetric neurons, no learning diversity.
- **Too large**: activations saturate or explode; gradients unstable.
- **Too small**: activations shrink to zero; vanishing signal.
- **Xavier/Glorot**: `Var(W) ≈ 2 / (fan_in + fan_out)` — balances forward and backward variance for tanh/sigmoid.
- **He/Kaiming**: `Var(W) ≈ 2 / fan_in` — accounts for ReLU killing half the units.
- **Fan-in/fan-out**: input and output dimension of the layer.

## Tradeoffs
- Use when: starting any new architecture — match init to activation (He for ReLU, Xavier for tanh).
- Avoid when: re-init mid-training unless deliberate (e.g., lottery ticket research); transfer learning uses pretrained weights instead.

## If they dig deeper
- Orthogonal init for RNNs — preserves norm across time steps better than i.i.d. Gaussian.
- Fixup initialization / zero-init last layer — train deep nets without normalization in some setups.
- Pretrained weights as "init" — transfer learning skips random init entirely.
