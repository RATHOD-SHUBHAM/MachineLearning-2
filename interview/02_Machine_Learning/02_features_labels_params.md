# Features, Labels, Samples, Parameters vs Hyperparameters

## Say this (60–90 sec)
In supervised learning, each **sample** is one example — one email, one user, one transaction. **Features** are the measurable inputs we feed the model — word counts, amount spent, time of day. The **label** is what we want to predict — spam or not spam, price, churn yes/no. The model learns **parameters** — the internal weights and biases it adjusts during training, like the coefficients in linear regression. **Hyperparameters** are choices we make before training: learning rate, tree depth, number of neighbors in k-NN. The model does not learn those from gradient descent; we tune them with validation or cross-validation. A clean mental model: features and labels are data; parameters are what training optimizes; hyperparameters are how we configure the learning process itself.

## Why it matters
Mixing up these terms signals shallow understanding. Interviewers probe here before diving into algorithms — especially the params vs hyperparams distinction, which drives train/val/test and grid search.

## How it works
- **Sample / instance / example**: one row in your dataset.
- **Feature vector**: numeric (or encoded) description of one sample; dimension = number of features.
- **Label / target**: ground truth for supervised learning.
- **Parameters (θ)**: learned by minimizing loss on training data — weights in linear/logistic regression, split rules in trees.
- **Hyperparameters**: set externally — regularization strength λ, k in k-NN, epochs, batch size.
- **Training**: find parameters that fit data; **tuning**: pick hyperparameters that generalize best on validation.

## Tradeoffs
- Use when: framing any supervised problem — always name what is X (features) and y (label).
- Avoid when: using “parameter” loosely for everything — be precise in interviews.

## If they dig deeper
- Categorical features need encoding — one-hot, embeddings, target encoding.
- High-dimensional sparse features (text) vs dense tabular — different model choices.
- More hyperparameters → more search cost and overfitting risk on small validation sets.
