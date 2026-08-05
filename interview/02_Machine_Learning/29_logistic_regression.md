# Logistic Regression

## Say this (60–90 sec)
Logistic regression is the go-to binary classifier despite the name. It models the probability that an example belongs to the positive class — spam or not spam — using a linear score passed through a sigmoid. The output is between 0 and 1, so you get calibrated-ish probabilities, not just a hard label. Training minimizes log loss, also called cross-entropy, which heavily penalizes confident wrong predictions. For spam detection, features might be word counts or sender reputation; the model learns weights that push spam emails toward probability near 1 and ham toward 0. Decision boundary is linear in feature space — a hyperplane — which makes it fast, interpretable, and a strong baseline before trying trees or neural nets.

## Why it matters
It bridges regression math and classification. Interviewers test whether you know why sigmoid plus cross-entropy beat MSE for classification, and how thresholds affect precision and recall.

## How it works
- **Model**: P(y=1|x) = σ(wᵀx + b) where σ(z) = 1 / (1 + e⁻ᶻ).
- **Decision**: predict positive if P(y=1|x) ≥ threshold (default 0.5); threshold is tunable.
- **Loss**: binary cross-entropy — −[y log p + (1−y) log(1−p)] — convex in w for logistic regression.
- **Training**: gradient descent on log loss; no closed-form like linear regression.
- **Interpretation**: log-odds are linear in features; exp(wⱼ) is the odds ratio per unit of feature j.

## Tradeoffs
- Use when: binary (or multiclass one-vs-rest) classification, you need probabilities or interpretable coefficients, or as a fast baseline.
- Avoid when: decision boundary is highly nonlinear and you have enough data for richer models, or classes are extremely imbalanced without resampling or class weights.

## If they dig deeper
- Why not MSE for classification? — sigmoid saturates, gradients vanish; cross-entropy aligns with maximum likelihood.
- Multiclass extension: softmax regression (one-vs-rest or multinomial).
- Regularization (L1/L2) prevents overfitting with many sparse text features.
- Platt scaling or isotonic regression can improve probability calibration post-hoc.
