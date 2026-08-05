# One-Class SVM

## Say this (60–90 sec)
One-Class SVM is a semi-supervised anomaly method: you train mostly on **normal** data and learn a boundary around it. Using a kernel — usually RBF — it maps points to a feature space and finds a hyperplane that separates the data from the origin with maximum margin, or equivalently encloses normal points in a small region. At inference, points outside that region are anomalies. The parameter **nu** roughly upper-bounds the fraction of outliers and lower-bounds the fraction of support vectors. It’s useful when you can collect lots of healthy examples — normal machine telemetry, normal login behavior — but labeled failures are rare.

## Why it matters
Classic “train on normal only” story. Interviewers contrast it with Isolation Forest (fully unsupervised, no explicit normal-only assumption as strict) and with binary SVM.

## How it works
- Fit on normal (or mostly normal) samples.
- Kernel (RBF) makes a flexible decision surface in input space.
- Decision function: signed distance / score — negative or below threshold → anomaly.
- Key knobs: **nu**, **gamma** (RBF width), scaling of features.

## Tradeoffs
- Use when: you have clean normal data, moderate size, need a boundary around normality.
- Avoid when: normal data is dirty with many unlabeled anomalies, datasets are very large (scaling), or high-dim sparse features without careful kernels — Isolation Forest is often easier.

## If they dig deeper
- nu vs C in binary SVM — nu is more interpretable for outlier fraction.
- vs autoencoder: both can be “normal-only”; SVM is shallower/kernel; AE learns nonlinear compression.
- Always scale features before RBF kernels.
