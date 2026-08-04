# Optimizers

## Say this (60–90 sec)
Optimizers decide how to update weights given gradients. Plain SGD: w equals w minus learning rate times gradient. Simple and generalizes well with tuning, but noisy and slow on ill-conditioned landscapes. Momentum adds a velocity term — it accumulates past gradients like a ball rolling downhill, smoothing oscillations and speeding convergence in consistent directions. Adam adapts the learning rate per parameter using running estimates of first and second moments of the gradient — fast out of the box, good default for many problems. AdamW decouples weight decay from the adaptive step, which matters for regularization in transformers. In practice: Adam or AdamW for quick experiments and transformers; SGD with momentum for CNNs when you want best final accuracy and can tune. Always pair with a sensible learning rate and schedule.

## Why it matters
Same architecture, different optimizer, very different training curves. Interviewers check if you know when defaults work and when SGD still wins.

## How it works
- **SGD**: `w ← w - η ∇L`. Optional weight decay: `w ← w - η(∇L + λw)`.
- **Momentum**: `v ← βv + ∇L`, `w ← w - ηv`. β ≈ 0.9 typical.
- **Adam**: maintains `m` (mean grad) and `v` (mean squared grad); `w ← w - η · m / (√v + ε)`.
- **AdamW**: weight decay applied directly on weights, not mixed into gradient.
- **All require**: `optimizer.zero_grad()` before backward, `loss.backward()`, `optimizer.step()`.

## Tradeoffs
- Use when: Adam/AdamW for default DL training, NLP, small datasets; SGD+momentum for large vision with careful LR tuning.
- Avoid when: Adam with too high LR — can diverge early; switching optimizers mid-run without retuning LR.

## If they dig deeper
- Why Adam generalizes worse than SGD on some vision tasks — sharp vs flat minima debate (not settled).
- Learning rate warmup with Adam — small LR early prevents bad moment estimates.
- RMSprop, Adagrad — older adaptive methods; Adam largely superseded them for general use.
