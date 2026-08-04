# Chain Rule (Prep for Backprop)

## Say this (60–90 sec)
The chain rule handles nested functions. If y depends on u, and u depends on x, then the rate of change of y with respect to x is the rate of y w.r.t. u times the rate of u w.r.t. x. In neural nets, the loss depends on the output, which depends on hidden layers, which depend on weights — a long chain. Backpropagation is just the chain rule applied systematically from loss backward through the graph, reusing intermediate results. Each layer passes "how much did I contribute to the error?" to the layer below. That's why we cache activations during the forward pass — we need them for the backward pass. No magic: chain rule plus efficient bookkeeping.

## Why it matters
Backprop is the engine of deep learning. If you can explain chain rule on a tiny two-layer net, you can explain why vanishing gradients happen in deep sigmoid stacks and why ReLU and skip connections help. Autograd libraries automate the chain rule — but you still need the mental model to debug.

## How it works
- **Single variable**: if `y = f(u)` and `u = g(x)`, then `dy/dx = (dy/du)(du/dx)`.
- **Many variables**: `∂L/∂x = Σ (∂L/∂u_i)(∂u_i/∂x)` — sum over all paths (multivariate chain rule).
- **Computational graph**: each node is an op; backward pass multiplies local Jacobians along edges from loss to parameters.
- **Example**: linear layer `z = Wx`, ReLU `a = max(0,z)`, loss L — gradient w.r.t. W flows through ReLU mask and input x.
- **Forward cache**: store x and z during forward pass — backward pass needs them for local derivatives.
- **Shared weights**: same W used twice — gradients from both paths add (multivariate chain rule).

## Tradeoffs
- Use when: deriving gradients by hand for simple layers, debugging autograd, explaining backprop in interviews.
- Avoid when: doing it manually for entire ResNet — use autograd; but know the principle.

## If they dig deeper
- Vanishing gradient: many sigmoid/tanh layers multiply small derivatives (<1) — signal dies in early layers.
- Exploding gradient: products of large values — fix with gradient clipping, careful init, normalization.
- ReLU chain rule: gradient passes through where activation > 0, blocked where ≤ 0 — sparse gradient flow.
- Residual connections: gradient can flow directly through skip path — mitigates vanishing in very deep nets.
- Manual backprop on `L = (y - wx)²`: ∂L/∂w = 2(y - wx)(-x) — good whiteboard exercise.
