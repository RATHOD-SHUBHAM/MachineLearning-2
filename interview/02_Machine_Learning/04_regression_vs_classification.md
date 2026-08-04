# Regression vs Classification

## Say this (60–90 sec)
Both are supervised learning, but the output type differs. **Regression** predicts a continuous number — house price, temperature, revenue. The model outputs a real value; we measure error with losses like MSE or MAE. **Classification** predicts a discrete category — spam or not, digit 0–9, disease present or absent. Binary classification has two classes; multiclass has more. For classification we often output class probabilities — via sigmoid for binary or softmax for multiclass — then pick the class with highest probability or tune a threshold. Same pipeline either way: features in, train on labeled data, evaluate on held-out data. The evaluation metrics change though: RMSE for regression; accuracy, precision, recall, AUC for classification. Choosing the wrong framing — treating a category as a number — is a classic mistake, like encoding {low, medium, high} as 1, 2, 3 and running linear regression.

## Why it matters
Problem type drives loss function, metrics, and model output layer. Interviewers check you match the task to the right toolbox.

## How it works
- **Regression**: y ∈ ℝ (or bounded interval). Loss: MSE, MAE, Huber. Output: linear activation, one neuron.
- **Binary classification**: y ∈ {0, 1}. Loss: log loss / binary cross-entropy. Output: sigmoid → probability.
- **Multiclass**: y ∈ {1, …, K}. Loss: categorical cross-entropy. Output: softmax over K classes.
- **Ordinal vs nominal**: ordered categories (ratings) may need special treatment — not plain multiclass.
- **Multi-label**: each sample can have multiple true labels — different from multiclass (one label per sample).

## Tradeoffs
- Use regression when: the target is truly continuous and ordering/magnitude of errors matters.
- Use classification when: decisions are discrete buckets or yes/no actions.
- Avoid regression on arbitrary category codes — use classification or proper ordinal models.

## If they dig deeper
- Logistic regression is classification despite the name — linear model + sigmoid.
- Imbalanced classification — accuracy misleads; precision/recall matter.
- Regression with outliers — MAE or Huber more robust than MSE.
