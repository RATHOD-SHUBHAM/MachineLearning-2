# Regularization (L1 / L2)

## Say this (60–90 sec)
Regularization fights overfitting by penalizing large weights during training. Instead of minimizing loss alone, you minimize loss plus a penalty term controlled by λ — the regularization strength. L2, or Ridge, adds the sum of squared weights — it shrinks all coefficients toward zero but rarely to exactly zero. L1, or Lasso, adds the sum of absolute weights — it can zero out features entirely, giving you built-in feature selection. For house price prediction with 200 correlated features, L2 keeps all features but dampens noise; L1 might keep only sqft, bedrooms, and year built. You tune λ on validation — too small and you overfit, too large and you underfit. Regularization is one of the simplest, most effective tools in the ML toolkit.

## Why it matters
Overfitting comes up in almost every interview. Regularization shows you understand the bias–variance tradeoff and how to control model complexity without just collecting more data.

## How it works
- **Objective**: minimize L(w) + λ · R(w) — L is task loss (MSE, cross-entropy), R is the penalty.
- **L2 (Ridge)**: R = Σ wⱼ² — smooth shrinkage; closed-form solution exists for linear regression.
- **L1 (Lasso)**: R = Σ |wⱼ| — sparse solutions; non-differentiable at 0, solved with subgradient or coordinate descent.
- **Elastic Net**: L1 + L2 — sparsity plus stability with correlated features.
- **Effect**: higher λ → simpler model, higher bias, lower variance.

## Tradeoffs
- Use when: many features, multicollinearity, limited data, or weights are growing large on training but validation error rises.
- Avoid when: model is already underfitting (regularization makes it worse), or you need all features retained for interpretability without selection.

## If they dig deeper
- λ is a hyperparameter — tune via cross-validation, not on the test set.
- L1 prefers one feature from a correlated group; L2 spreads weight across them.
- In neural nets, weight decay is L2 on weights; dropout is a different regularization mechanism.
- Bayesian view: L2 ≈ Gaussian prior on weights, L1 ≈ Laplace prior.
