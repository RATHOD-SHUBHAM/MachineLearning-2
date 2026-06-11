"""
Isolation Forest for anomaly detection (study script).

Uses scikit-learn on the credit card fraud dataset. Labels are only for
evaluation — the model is trained without them, like real unsupervised use.

Dataset: https://drive.google.com/file/d/1shGMkjBDTFkOM5JJO_VlzXCNCpnoA9TA/view?usp=drive_link
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to the CSV next to this script (Kaggle credit card fraud data).
DATA_PATH = Path(__file__).resolve().parent / "creditcard.csv"

# Cap rows so training stays quick on a laptop; set to None for the full file.
MAX_ROWS = 50_000
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Class: 0 = normal, 1 = fraud. Time is dropped; V1–V28 are PCA features from
# the original dataset, plus Amount.
df = pd.read_csv(DATA_PATH, nrows=MAX_ROWS)

X = df.drop(columns=["Class", "Time"])
y = df["Class"].astype(int)

print(f"Loaded {len(df):,} rows, {X.shape[1]} features")
print(f"Fraud rate: {y.mean():.4%} ({y.sum():,} fraud / {len(y):,} total)")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
# Stratify keeps the rare fraud class in both splits. Labels are not used for
# fitting IsolationForest — only for metrics later.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
# Scale each feature to mean 0 and variance 1. Fit on train only so test data
# does not leak statistics (same idea as supervised pipelines).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------
# Isolation Forest: random trees isolate points; anomalies need fewer splits.
# contamination ≈ expected outlier fraction; sklearn uses it for the -1/1 threshold.
# predict: 1 = inlier, -1 = outlier. score_samples: higher = more normal.
model = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.02,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Unsupervised fit — no y_train passed in.
model.fit(X_train_scaled)

# ---------------------------------------------------------------------------
# 5. Predictions and anomaly scores
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
anomaly_scores = model.score_samples(X_test_scaled)

# Map sklearn labels to 0/1 for metrics: -1 (outlier) -> 1 (predicted fraud).
y_pred_binary = (y_pred == -1).astype(int)

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
# With heavy class imbalance, accuracy is misleading; precision/recall matter more.
print("\n--- Classification report (1 = predicted fraud) ---")
print(classification_report(y_test, y_pred_binary, digits=4))

cm = confusion_matrix(y_test, y_pred_binary)
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(cm)

# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------
# PCA to 2D for a scatter plot (V1–V28 are already PCA components on Amount/Time).
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_test_2d = pca.fit_transform(X_test_scaled)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Left: true labels
axes[0].scatter(
    X_test_2d[y_test == 0, 0],
    X_test_2d[y_test == 0, 1],
    c="steelblue",
    s=8,
    alpha=0.35,
    label="Normal (true)",
)
axes[0].scatter(
    X_test_2d[y_test == 1, 0],
    X_test_2d[y_test == 1, 1],
    c="crimson",
    s=25,
    alpha=0.8,
    label="Fraud (true)",
)
axes[0].set_title("Ground truth")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend(loc="best", fontsize=8)

# Middle: model outliers
axes[1].scatter(
    X_test_2d[y_pred == 1, 0],
    X_test_2d[y_pred == 1, 1],
    c="steelblue",
    s=8,
    alpha=0.35,
    label="Inlier (pred)",
)
axes[1].scatter(
    X_test_2d[y_pred == -1, 0],
    X_test_2d[y_pred == -1, 1],
    c="darkorange",
    s=25,
    alpha=0.8,
    label="Outlier (pred)",
)
axes[1].set_title("Isolation Forest predictions")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend(loc="best", fontsize=8)

# Right: anomaly score distribution by true class
axes[2].hist(
    anomaly_scores[y_test == 0],
    bins=40,
    alpha=0.6,
    label="Normal (true)",
    color="steelblue",
)
axes[2].hist(
    anomaly_scores[y_test == 1],
    bins=40,
    alpha=0.7,
    label="Fraud (true)",
    color="crimson",
)
axes[2].set_title("Anomaly score (higher = more normal)")
axes[2].set_xlabel("score_samples")
axes[2].set_ylabel("Count")
axes[2].legend(loc="best", fontsize=8)

plt.tight_layout()
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.show()
