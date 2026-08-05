# Loss Function vs Cost Function

## Say this (60–90 sec)
A **loss function** measures error on **one** training example — how wrong is this single prediction? For a house priced at 300k that I predicted as 280k, squared loss is (20k)². A **cost function** is usually the **average (or sum) of losses over the whole training set** — or over a batch. So loss is per example; cost is the aggregate we actually minimize with gradient descent. People say “loss” and “cost” interchangeably in industry and PyTorch — `loss = criterion(pred, y)` is often already a batch mean — but in interviews, this distinction shows you read the theory carefully. Closely related: the **objective** is what you optimize end-to-end — often cost plus regularization (L1/L2). Metrics like accuracy or MAE on a test set can look like losses, but metrics are for **evaluation**; loss/cost drive **training**.

## Why it matters
Classic “define your terms” interview question. Also clarifies what `loss.backward()` is differentiating — usually a batch-averaged cost.

## How it works
- **Loss** L(ŷᵢ, yᵢ): error for example i — e.g. (ŷᵢ − yᵢ)², or −log pᵢ for classification.
- **Cost** J(w) ≈ (1/n) Σ L(ŷᵢ, yᵢ): average loss over n examples (sometimes sum, not mean — same minimizer up to scale).
- **Batch cost**: average loss over the mini-batch (what SGD steps on).
- **Objective**: J(w) + λ R(w) — cost plus penalty.
- Training: compute cost → gradients w.r.t. w → update w.

## Tradeoffs
- Use when: explaining training math, deriving gradients, or when the interviewer asks loss vs cost specifically.
- Avoid when: being pedantic in casual code review — saying “loss” for batch mean is normal; just be ready to unbundle it if asked.

## If they dig deeper
- Empirical risk = average loss on the training data ≈ cost.
- Risk (true) = expected loss over the data distribution — what we wish we could minimize.
- Same formula as a metric (e.g. MSE) can be a loss in training and a metric in reporting — role differs.
- NN catalog of losses: [`03_Neural_Networks/04_loss_functions.md`](../03_Neural_Networks/04_loss_functions.md).
