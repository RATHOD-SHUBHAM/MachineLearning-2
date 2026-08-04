# Softmax + Cross-Entropy Intuition

## Say this (60–90 sec)
Softmax turns a vector of raw scores — logits — into a valid probability distribution. It exponentiates each score, then divides by the sum so all outputs are positive and add to one. Bigger logits get disproportionately more probability — that's the exponential. Cross-entropy loss measures how wrong your predicted distribution is compared to the true label. For classification, the true label is usually one-hot: probability 1 on the correct class, 0 elsewhere. Cross-entropy punishes confident wrong predictions heavily — log of a small probability is a big negative number. Together, softmax plus cross-entropy is the standard classification head: softmax at inference for probabilities, cross-entropy during training for a smooth, well-behaved gradient. The gradient has a clean form: predicted probability minus true label — simple and numerically stable when implemented as one fused op.

## Why it matters
This pair appears in virtually every classifier — image models, NLP, tabular. Interviewers expect you to explain why we don't use MSE on probabilities and why logits are preferred over applying softmax before the loss. The output is interpretable: class with highest softmax score is the model's prediction.

## How it works
- **Softmax**: `p_i = exp(z_i) / Σ exp(z_j)` — converts logits z to probabilities p.
- **Cross-entropy** (one-hot target): `L = -log(p_correct)` — only the true class term matters.
- **Combined gradient**: `∂L/∂z_i = p_i - y_i` — prediction minus target; drives logits up for true class, down for others.
- **Logits + log-softmax + NLL**: frameworks fuse this for stability — avoids computing exp twice or log(0).
- **Multi-class**: one softmax over all classes — probabilities compete and sum to 1.
- **Numerical trick**: subtract max logit before exp — prevents overflow, doesn't change softmax result.

## Tradeoffs
- Use when: multi-class classification, comparing predicted vs true distributions, training with class labels.
- Avoid when: regression (use MSE/MAE), multi-label with independent sigmoids (not softmax — classes aren't mutually exclusive), or when calibrated ranking matters more than normalized probabilities.

## If they dig deeper
- Why not MSE on one-hot: cross-entropy gradient doesn't vanish when prediction is wrong but confident — better learning signal.
- Label smoothing: replace hard 0/1 with soft targets — reduces overconfidence, improves generalization.
- Temperature scaling: divide logits by T before softmax — T > 1 flattens distribution (more uncertain), T < 1 sharpens it; used in distillation and calibration.
- Class imbalance: cross-entropy still works but you may weight rare classes or use focal loss for hard examples.
- Top-k accuracy vs cross-entropy: metric cares about rank; loss cares about probability mass on true class.
