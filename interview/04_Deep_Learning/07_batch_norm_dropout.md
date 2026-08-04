# Batch Normalization and Dropout

## Say this (60–90 sec)
Batch normalization normalizes activations within a mini-batch — subtract mean, divide by std, then scale and shift with learnable gamma and beta. It stabilizes training, allows higher learning rates, and acts as mild regularization. Works on conv and linear layers; placed before or after activation depending on convention. At inference, uses running averages of mean and var accumulated during training — that's why model.eval() matters. Dropout randomly zeros neurons during training with probability p — forces redundant representations, reduces co-adaptation. At inference, dropout is off and outputs are scaled accordingly. Batch norm fights internal covariate shift; dropout fights overfitting. Layer norm normalizes across features per token — preferred in transformers where batch size or sequence length varies. Use batch norm in CNNs, layer norm in transformers; dropout in both when overfitting.

## Why it matters
Standard regularization and stabilization tools. Explaining train vs eval behavior for both is a common gotcha question.

## How it works
- **BatchNorm**: normalize `(x - μ_batch) / σ_batch`, then `γx + β`. Track EMA of μ, σ for inference.
- **Dropout**: mask ~ Bernoulli(1-p); training only; inference uses full network (weights implicitly scaled).
- **LayerNorm**: normalize each sample's features — no batch statistics; stable for variable seq length.
- **Placement**: ConvNet often Conv → BN → ReLU; transformer uses LN inside residual blocks.

## Tradeoffs
- Use when: BN for CNNs/MLPs with decent batch size; LN for transformers/RNNs; dropout when overfitting.
- Avoid when: BN with batch size 1 or very small — noisy stats; dropout in bottleneck layers without tuning (may hurt).

## If they dig deeper
- Why BatchNorm helps — stabilization, implicit regularization, smoother loss landscape (debated mechanisms).
- Dropout vs weight decay — complementary regularizers.
- Stochastic depth, DropPath — structured dropout for residual blocks.
