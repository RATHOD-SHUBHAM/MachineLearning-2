# PCA

## Say this (60–90 sec)
Principal Component Analysis reduces dimensionality by finding orthogonal directions of maximum variance in the data. The first principal component is the line along which data varies the most; the second is orthogonal to it and captures the next most variance, and so on. You keep the top K components and project data onto them — compressing 100 features down to 10 while retaining most signal. For house price datasets with correlated features like sqft, lot size, and room count, PCA removes redundancy and speeds up downstream models. It is unsupervised — it ignores labels — so components are not guaranteed to align with what helps classification. Still, it is invaluable for visualization, noise reduction, and fighting the curse of dimensionality.

## Why it matters
PCA is the standard dimensionality reduction technique. Interviewers test eigendecomposition vs SVD, variance explained, and when PCA helps vs hurts supervised tasks.

## How it works
- **Center data**: subtract mean from each feature (required for meaningful components).
- **Compute covariance matrix** (or apply SVD directly on centered X) — eigenvectors are principal directions.
- **Project**: X_reduced = X_centered · W_K — W_K columns are top K eigenvectors.
- **Choose K**: cumulative explained variance ratio (e.g. keep 95% of total variance).
- **Output**: uncorrelated components ordered by variance — PC1 has highest variance.

## Tradeoffs
- Use when: high multicollinearity, visualization (2D/3D), noise reduction, speeding up training, or as preprocessing before clustering.
- Avoid when: interpretability of original features is required (PCA components are linear mixes), features are already sparse/high-dimensional text (truncated SVD / LSA is more common), or label-relevant variance is low-variance in input space.

## If they dig deeper
- Scale features before PCA — otherwise units with larger range dominate components.
- PCA is linear — kernel PCA or autoencoders handle nonlinear structure.
- SVD on X/n is numerically preferred over explicit covariance for large n, p.
- PCA on training set only — fit on train, transform train and test to avoid leakage.
