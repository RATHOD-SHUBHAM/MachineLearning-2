# Interview Prep Notes

Spoken-answer cheat sheets for ML / DL / GenAI interviews.

Read folders **in order**. Files inside each folder are numbered — don’t shuffle.

Study schedule lives in [`docs/pytorch-ml-study-roadmap.md`](../docs/pytorch-ml-study-roadmap.md). This folder is the **speak-it-out-loud** layer.

## How to use

1. Open a topic.
2. Read **Say this** out loud once (60–90 sec).
3. Skim Why / How / Tradeoffs.
4. Practice the **If they dig deeper** follow-ups without looking.

Template every file follows:

- **Say this** — full spoken answer
- **Why it matters**
- **How it works**
- **Tradeoffs** (Use when / Avoid when)
- **If they dig deeper**

## Reading order

### 01 Math
1. Scalars, vectors, matrices
2. Tensors (rank, shape — concept)
3. Dot product and norms
4. Matrix multiply
5. Derivatives and partials
6. Gradients
7. Chain rule
8. Probability basics
9. Softmax and cross-entropy

### 02 Machine Learning
**Vocabulary** → **Classification metrics** → **Regression metrics** → **Algorithms** → **Anomaly detection** → **Ranking / LM metrics**

1. What is ML
2. Features, labels, params vs hyperparameters
3. Supervised / unsupervised / RL
4. Regression vs classification
5. Train / val / test
6. Underfitting vs overfitting
7. Bias–variance
8. Feature scaling
9. Cross-validation
10. Confusion matrix
11. TP / TN / FP / FN
12. Accuracy
13. Precision
14. Recall
15. Specificity
16. F1
17. Precision–recall tradeoff + threshold
18. ROC and AUC
19. PR-AUC
20. Log loss (cross-entropy metric)
21. Loss vs cost function
22. MAE
23. MSE
24. RMSE
25. R² / Adjusted R²
26. MAPE
27. Linear regression
28. Gradient descent
29. Logistic regression
30. Regularization (L1/L2)
31. Decision trees
32. Random forest
33. SVM
34. K-means
35. PCA
36. Anomaly detection overview
37. Isolation Forest
38. DBSCAN
39. LOF (Local Outlier Factor)
40. One-Class SVM
41. NDCG
42. Hit Rate
43. Perplexity

### 03 Neural Networks
1. Perceptron / neuron
2. Activations
3. MLP
4. Loss functions
5. Backpropagation
6. Vanishing / exploding gradients
7. Weight init
8. Optimizers
9. Learning rate

### 04 Deep Learning
1. Deep vs shallow
2. CNN
3. Transfer learning
4. RNN / LSTM / GRU
5. Attention
6. Transformers
7. Batch norm + dropout
8. Autoencoders

### 05 GenAI
1. Tokens and embeddings
2. Pretrain vs fine-tune
3. Decoder LLMs
4. Prompting
5. LoRA / PEFT
6. RAG
7. RLHF
8. Hallucinations and eval

### 06 PyTorch Essentials
1. Tensor in PyTorch (dtype, device, view) — concept first in Math `02_tensors`
2. Autograd
3. nn.Module + training step
4. Train vs eval / no_grad
5. Common shape mistakes

## Running example

For classification metrics, we reuse **spam detection** (positive = spam) so TP/FP/FN stick.
