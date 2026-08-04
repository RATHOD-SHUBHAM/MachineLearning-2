# Underfitting vs Overfitting

## Say this (60–90 sec)
**Underfitting** means the model is too simple to capture the signal in the data — high error on both training and validation. It’s not learning enough; think linear model on a highly nonlinear pattern. **Overfitting** means the model memorizes training noise — low training error but poor validation/test performance. It’s too flexible for the amount of data or noise level. The sweet spot is a model complex enough to capture real patterns but constrained enough to generalize. I diagnose with learning curves: if train and val error are both high, underfit; if train error is low and val error is high, overfit. Fixes for underfitting: richer features, more complex model, train longer. Fixes for overfitting: more data, regularization, simpler model, dropout, early stopping, better features. Regularization explicitly penalizes complexity so the model prefers simpler explanations.

## Why it matters
This is the central tension in ML. Interviewers expect you to recognize symptoms and prescribe fixes without hand-waving.

## How it works
- **Underfitting**: high bias — model assumptions too rigid. Train loss plateaus high; val loss similar.
- **Overfitting**: high variance — model fits idiosyncrasies. Train loss very low; val loss much higher.
- **Model capacity**: parameters, depth, polynomial degree — higher capacity eases underfitting but risks overfitting.
- **Learning curves**: plot error vs training set size — overfit gap may shrink with more data.
- **Regularization**: L1/L2 shrink weights; dropout; data augmentation; ensembling reduces variance.

## Tradeoffs
- Use simpler models when: data is scarce, noise is high, interpretability matters.
- Use complex models when: large clean data, rich signal, and you have validation to tune regularization.
- Avoid when: chasing zero training error as the goal — that often means overfitting.

## If they dig deeper
- Double descent — very over-parameterized models can generalize well (modern deep learning).
- Bias–variance decomposition — under/overfitting as bias vs variance extremes.
- Early stopping as implicit regularization — stop before memorization completes.
