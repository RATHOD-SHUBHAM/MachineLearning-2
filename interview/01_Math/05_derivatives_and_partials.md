# Derivatives and Partial Derivatives

## Say this (60–90 sec)
A derivative tells you how fast a function changes when you nudge its input a tiny bit. On a curve, it's the slope of the tangent line at a point. In ML, we have loss as a function of many parameters — not one input. A partial derivative asks: if I change just this one weight and hold everything else fixed, does loss go up or down, and how fast? That's the sensitivity of loss to that single parameter. We compute partials for every weight, bias, and hyperparameter we optimize. Without derivatives, gradient descent has no direction. Smooth, differentiable losses let us use calculus; non-differentiable spots — like ReLU at zero — we pick a convention or use subgradients.

## Why it matters
Training is optimization: minimize loss by moving parameters opposite to the direction loss increases. Partials are the building blocks of the gradient vector. Interviewers want to see you connect calculus to "why backprop works." If ∂L/∂w is positive, increasing w increases loss — so we decrease w.

## How it works
- **Derivative**: `f'(x) ≈ (f(x+h) - f(x)) / h` as h goes to zero — instantaneous rate of change.
- **Partial derivative**: `∂L/∂w` — change in loss L w.r.t. one weight w, others held constant.
- **Gradient** (preview): stack all partials into one vector — points uphill on the loss surface.
- **Common rules**: power rule, sum rule, product rule — plus chain rule for composed functions (next topic).
- **Sigmoid derivative**: σ(1−σ) — largest at 0.5, shrinks toward 0 at extremes; contributes to vanishing gradients.
- **MSE loss**: derivative w.r.t. prediction is proportional to (prediction − target) — intuitive error signal.

## Tradeoffs
- Use when: reasoning about sensitivity, deriving update rules, debugging vanishing/exploding gradients.
- Avoid when: trying to optimize discrete choices (which architecture node to pick) with plain gradients — need different methods (REINFORCE, evolutionary search).

## If they dig deeper
- ReLU derivative: 1 for x > 0, 0 for x < 0, undefined at 0 — frameworks use 0 or 1 at zero by convention.
- Numerical gradient check: compare analytic partial to finite-difference estimate — sanity check for backprop bugs.
- Convex vs non-convex: linear regression loss is convex (one bowl); deep nets are non-convex (many local valleys, but SGD still works in practice).
- Jacobian: matrix of all partial derivatives for vector-valued functions — generalizes gradient to multi-output layers.
- Subgradient: valid direction for non-differentiable points — lets us optimize ReLU networks rigorously.
