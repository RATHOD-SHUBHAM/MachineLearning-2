# Gradients (Steepest Ascent; We Go Opposite for GD)

## Say this (60–90 sec)
The gradient of a scalar function is a vector of all partial derivatives — one entry per parameter. It points in the direction of steepest ascent: if you take a tiny step along the gradient, the function increases fastest. For loss minimization, we go the opposite way — negative gradient — that's gradient descent. Learning rate controls step size: too big, you overshoot and diverge; too small, training crawls. In high dimensions, the gradient is still meaningful: each component tells you which parameter to nudge and in which direction. Batch gradient descent uses the full dataset gradient; stochastic uses one or a mini-batch — noisier but cheaper per step, and noise can help escape shallow local minima.

## Why it matters
Every optimizer — SGD, Adam, AdamW — is built on gradients. Understanding ascent vs descent, step size, and batch noise is core ML literacy, not optional math trivia. The loss surface in deep learning is high-dimensional — we can't visualize it, but local gradient direction still guides useful updates.

## How it works
- **Definition**: `∇L = [∂L/∂w1, ∂L/∂w2, ..., ∂L/∂wn]^T` for parameters w1...wn.
- **Steepest ascent**: direction of max increase; **gradient descent**: `w ← w - η ∇L` where η is learning rate.
- **Mini-batch**: average gradient over B samples — tradeoff between accuracy of direction and compute.
- **Zero gradient**: critical point — could be minimum, maximum, or saddle; in deep nets, saddles are common.
- **Learning rate schedule**: start larger, decay over time — big steps early, fine-tuning later.
- **Weight decay**: often added as L2 penalty — gradient includes extra term pushing weights toward zero.

## Tradeoffs
- Use when: explaining how training updates weights, choosing learning rate schedules, diagnosing plateauing loss.
- Avoid when: loss landscape is flat (vanishing gradient) or chaotic (exploding) — need architecture fixes, normalization, clipping, or better init — not just a bigger learning rate.

## If they dig deeper
- Why negative gradient minimizes: first-order Taylor approximation — moving opposite to gradient decreases function locally.
- Momentum: accumulate past gradients — smooths updates, speeds convergence in consistent directions.
- Adam adapts per-parameter step sizes using running averages of gradient and squared gradient — default choice for many tasks.
- Second-order methods (Newton) use Hessian — faster convergence but expensive for millions of parameters.
- Plateauing loss often means gradient magnitude is small — try higher LR, different init, or architecture change.
