# Log Loss (Cross-Entropy as a Metric)

## Say this (60–90 sec)
Log loss — binary or multiclass cross-entropy — measures how good your **predicted probabilities** are, not just hard labels. For spam, if the true label is spam and I output 0.99, log loss is small; if I output 0.01, I’m confidently wrong and log loss explodes. Formula for binary: −[y log p + (1−y) log(1−p)], averaged over examples. Unlike accuracy, it rewards calibrated confidence. You can use the same formula as a **training loss** and as an **evaluation metric** on a holdout set. If two models have similar AUC but one has much worse log loss, that one is overconfident or poorly calibrated.

## Why it matters
Interviews often ask: accuracy vs log loss. Shows you understand probabilistic predictions.

## How it works
- Needs predicted probabilities p ∈ (0, 1), not just 0/1 labels.
- Clip p away from 0/1 for numerical stability (e.g. 1e−15).
- Multiclass: −Σ y_k log p_k (one-hot or true class only).
- Lower is better; perfect = 0.

## Tradeoffs
- Use when: you care about probability quality (risk scores, ranking, calibration).
- Avoid when: you only have hard predictions; or business only cares about one threshold — then precision/recall/F1 at that threshold matter more.

## If they dig deeper
- Same family as training loss — see [`21_loss_vs_cost.md`](./21_loss_vs_cost.md) and NN losses.
- Brier score — another probabilistic metric (squared error on probabilities).
- Calibration curves / expected calibration error — companion to log loss.
