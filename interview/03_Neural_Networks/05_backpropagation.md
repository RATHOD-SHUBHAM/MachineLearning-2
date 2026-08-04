# Backpropagation

## Say this (60–90 sec)
Backpropagation is how we compute gradients in neural networks efficiently. Forward pass: data flows input to output, we compute the loss. Backward pass: we apply the chain rule layer by layer, starting from the loss and flowing backward. Each layer needs two things — the local gradient of its operation and the upstream gradient from the layer above. Multiply them to get gradients for that layer's weights and pass the gradient down to the layer below. The key insight is we reuse intermediate values from the forward pass and share computation across the graph — that's why it's O(parameters) instead of naive finite differences. In PyTorch, autograd builds this graph automatically. When I debug training, I check: are gradients flowing? Are they zero, exploding, or NaN? Backprop is the engine; the optimizer just uses the gradients it produces.

## Why it matters
Every training loop depends on correct gradients. Understanding backprop separates "I call loss.backward()" from "I know why my LSTM gate isn't learning."

## How it works
- **Forward**: compute activations, cache values needed for derivatives (inputs, pre-activations).
- **Backward**: ∂L/∂W = (∂L/∂z) · (∂z/∂W); chain rule propagates ∂L/∂h backward.
- **Computational graph**: each op has forward and backward rules; autodiff composes them.
- **Vector-Jacobian product**: for matrix layers, gradients match tensor shapes via broadcasting rules.
- **One backward per forward** (standard): call `loss.backward()`, then `optimizer.step()`.

## Tradeoffs
- Use when: any differentiable model trained with gradient descent — MLPs, CNNs, transformers.
- Avoid when: discrete/non-differentiable ops without relaxation — need REINFORCE, straight-through estimators, or surrogate gradients.

## If they dig deeper
- Reverse-mode vs forward-mode autodiff — reverse mode wins when params << input dim (typical in NNs).
- Vanishing gradient in deep chains — why skip connections and ReLU help.
- Gradient checkpointing — recompute activations during backward to save memory in large models.
