# Multi-Layer Perceptron (MLP)

## Say this (60–90 sec)
An MLP is a feedforward network: input layer, one or more hidden layers, output layer. Each layer is a matrix multiply plus bias plus activation. Hidden layers learn intermediate representations — early layers might detect edges or simple patterns, later layers combine them into higher-level features. MLPs are universal function approximators: with enough width and depth, they can approximate any continuous function. In practice they're great for tabular data and fixed-size vectors. They struggle with raw images or sequences because they ignore spatial and temporal structure — that's where CNNs and transformers help. Training is standard: forward pass, compute loss, backprop, update weights. The design choices are depth, width, activation, regularization, and how many parameters you can afford to fit without overfitting.

## Why it matters
MLPs are the simplest deep architecture. Understanding them shows you grasp how depth composes nonlinear transforms and why representation learning beats hand-crafted features on complex tasks.

## How it works
- **Layer**: `h = σ(Wx + b)` where W is (hidden_dim × input_dim).
- **Stacking**: `h1 = σ(W1 x + b1)`, `h2 = σ(W2 h1 + b2)`, ..., `y = W_out hL + b_out`.
- **Output head**: linear + MSE for regression; linear logits + cross-entropy for classification.
- **Capacity**: more layers/width = more expressivity, also more overfitting risk.
- **Tabular sweet spot**: often beats trees with enough data, tuning, and feature scaling.

## Tradeoffs
- Use when: tabular classification/regression, embeddings as fixed-size vectors, baseline before fancier models.
- Avoid when: grid data with local structure (images, audio spectrograms) or variable-length sequences — use CNN/RNN/transformer instead.

## If they dig deeper
- Universal approximation theorem — existence proof, not guarantee of learnability or generalization.
- Why depth beats width for some functions — compositional structure (e.g., parity) needs depth.
- Parameter count: two layers of size d → O(d²) weights; practical limit driven by data and regularization.
