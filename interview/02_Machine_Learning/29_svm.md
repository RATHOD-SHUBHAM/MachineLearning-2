# SVM (Intuition)

## Say this (60–90 sec)
Support Vector Machines find the decision boundary that separates classes with the maximum margin — the widest gap between the boundary and the nearest points from each class, called support vectors. For spam, imagine emails plotted in feature space; SVM draws the hyperplane with the largest buffer zone between spam and ham. Only support vectors matter for the boundary; other points could be removed without changing the model. Linear SVM works when classes are roughly separable. For nonlinear boundaries, the kernel trick maps features into a higher-dimensional space where a linear separator exists — without computing the mapping explicitly. Common kernels: polynomial, RBF. SVMs were dominant before deep learning on small-to-medium structured datasets, especially with clear margin structure.

## Why it matters
SVMs teach max-margin classification, support vectors, and the kernel trick — classic interview topics that connect geometry, optimization, and the bias–variance tradeoff.

## How it works
- **Linear SVM**: minimize ||w|| subject to correct classification with margin — or soft margin with slack variables for noise.
- **Support vectors**: training points on or inside the margin; the solution depends only on these.
- **Kernel trick**: K(x, x′) = φ(x)ᵀφ(x′) — compute inner products in implicit high-D space (RBF: exp(−γ||x−x′||²)).
- **C hyperparameter**: trades margin width vs misclassification — high C fits training data tighter (risk overfit).
- **Regression variant**: SVR — tube around predictions with ε-insensitive loss.

## Tradeoffs
- Use when: binary classification with moderate feature count, clear margin structure, or high-dimensional text (linear kernel) with strong regularization.
- Avoid when: very large datasets (training is O(n²) to O(n³) for naive solvers), you need probability outputs natively (Platt scaling needed), or deep nonlinear patterns in images/audio where CNNs dominate.

## If they dig deeper
- Hinge loss vs log loss — SVM penalizes points inside the margin but ignores well-classified points beyond it.
- RBF kernel γ controls locality — high γ = tight, wiggly boundary; low γ = smoother, more global.
- Linear SVM ≈ regularized logistic regression with different loss — often similar accuracy, different sparsity in support vectors.
- One-vs-rest or one-vs-one for multiclass.
