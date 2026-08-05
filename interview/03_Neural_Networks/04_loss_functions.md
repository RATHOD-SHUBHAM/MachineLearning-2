# Loss Functions

## Say this (60–90 sec)
The loss function tells the network how wrong its predictions are — it's what we minimize during training. For regression, Mean Squared Error is standard: average of squared differences between prediction and target. It penalizes large errors heavily because of the square. For classification, cross-entropy is the go-to: negative log of predicted probability for the true class. It heavily punishes confident wrong answers. Binary cross-entropy for two classes; categorical cross-entropy for multi-class one-hot labels. In practice I pass raw logits to the loss — PyTorch's CrossEntropyLoss applies softmax internally for stability. MSE assumes Gaussian noise; cross-entropy comes from maximum likelihood for classification. Pick the loss to match your output activation and task — mismatch means wrong gradients.

## Why it matters
Loss defines the optimization target. Interviewers check whether you connect output layer design, probability interpretation, and gradient behavior — not just name the formulas.

## How it works
- **MSE**: `(1/n) Σ (y - ŷ)²`. Regression default. Gradient linear in error.
- **Binary cross-entropy**: `- [y log(p) + (1-y) log(1-p)]`. y ∈ {0, 1}, p = predicted probability.
- **Categorical cross-entropy**: `- Σ y_i log(p_i)` over classes. y is one-hot or class index with log-softmax.
- **Relationship**: cross-entropy = negative log-likelihood when p comes from softmax/sigmoid.
- **Logits**: pre-softmax scores — numerically safer to compute loss in log-space.

## Tradeoffs
- Use when: MSE for continuous targets; BCE for binary; cross-entropy for multi-class classification.
- Avoid when: MSE on classification probabilities — wrong gradient shape; BCE with wrong label encoding (need 0/1 not -1/1 unless modified).

## If they dig deeper
- Loss (per example) vs cost (average over set) — [`02_Machine_Learning/19_loss_vs_cost.md`](../02_Machine_Learning/19_loss_vs_cost.md).
- MAE (L1) vs MSE — MAE more robust to outliers, MSE smooth for gradient descent.
- Focal loss — down-weights easy examples for imbalanced detection.
- Label smoothing — softens one-hot targets to reduce overconfidence.
