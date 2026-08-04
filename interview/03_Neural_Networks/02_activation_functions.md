# Activation Functions

## Say this (60–90 sec)
Activation functions sit after the linear part of a neuron and decide how much signal passes through. Without them, stacking layers would just collapse into one big linear transform — depth would be useless. ReLU is the default for hidden layers: max of zero and z. It's fast, sparse, and avoids vanishing gradients on the positive side. Sigmoid squashes to zero-one — great for binary probability outputs but saturates in hidden layers. Tanh is sigmoid shifted to minus-one to plus-one, zero-centered, still saturates. Softmax turns a vector of logits into a probability distribution — sum to one — used at the output for multi-class classification. Rule of thumb: ReLU in hidden layers, sigmoid or softmax at the output depending on the task, cross-entropy loss paired with logits when possible.

## Why it matters
Wrong activation choice causes dead neurons, vanishing gradients, or mismatched loss functions. This is a quick signal that you know practical training, not just the forward-pass formula.

## How it works
- **ReLU**: `max(0, z)`. Dead ReLU when z stays negative forever — gradient is zero.
- **Sigmoid**: `1 / (1 + e^(-z))`. Output in (0, 1). Gradient peaks at 0.25 — vanishes for large |z|.
- **Tanh**: `(e^z - e^(-z)) / (e^z + e^(-z))`. Zero-centered, range (-1, 1).
- **Softmax**: `e^z_i / Σ e^z_j`. Multi-class probabilities; numerically stable with log-sum-exp trick.
- **Linear (identity)**: used in regression output heads.

## Tradeoffs
- Use when: ReLU for hidden layers in MLPs/CNNs; sigmoid for binary output; softmax for multi-class; tanh sometimes in RNN gates.
- Avoid when: sigmoid/tanh in deep hidden layers without good init — saturation kills learning; softmax in hidden layers — no benefit over ReLU.

## If they dig deeper
- Leaky ReLU, GELU, Swish — smooth or leaky variants reduce dead neurons; GELU common in transformers.
- Why pair `CrossEntropyLoss` with raw logits — softmax baked into loss for numerical stability.
- Softmax temperature: divide logits by T; T > 1 flattens distribution, T < 1 sharpens it.
