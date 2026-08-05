# Gradient Descent (Batch / SGD / Mini-Batch)

## Say this (60–90 sec)
Gradient descent is how most ML models learn parameters. You define a loss — how wrong predictions are — then repeatedly move weights in the direction that decreases loss fastest. The gradient tells you that direction: partial derivatives of the loss with respect to each weight. Take a small step opposite the gradient, scaled by the learning rate, and repeat until loss stops improving. For linear regression on house prices, each step nudges weights so predicted prices move closer to actual sale prices. Three variants differ by how much data each step uses: batch uses the full dataset, stochastic uses one sample, mini-batch uses a small chunk — usually the best balance in practice.

## Why it matters
Nearly every trainable model — linear regression, logistic regression, neural networks — relies on some form of gradient-based optimization. Interviewers want you to explain the update rule and why batch size and learning rate matter.

## How it works
- **Update rule**: w ← w − η ∇L(w) — η is learning rate, ∇L is the gradient of loss.
- **Batch GD**: gradient computed over all training samples — smooth, slow per epoch, memory-heavy.
- **SGD**: one random sample per update — noisy but fast, escapes shallow local minima, needs learning-rate scheduling.
- **Mini-batch**: B samples per update (e.g. 32, 64) — GPU-friendly, stable enough, industry default.
- **Convergence**: stop when loss plateaus or after fixed epochs; monitor validation loss for overfitting.

## Tradeoffs
- Use when: loss is differentiable and you have too many parameters for a closed-form solution.
- Avoid when: loss is non-differentiable everywhere (use subgradients or different optimizers), or the landscape is extremely noisy with tiny data and no regularization.

## If they dig deeper
- Learning rate too high → divergence or oscillation; too low → slow convergence.
- Momentum and Adam smooth noisy SGD updates — Adam is the default for deep learning.
- Epoch vs iteration — one epoch = one full pass through training data; iterations depend on batch size.
- Local minima are less of a problem in high-dimensional non-convex loss (saddle points dominate).
