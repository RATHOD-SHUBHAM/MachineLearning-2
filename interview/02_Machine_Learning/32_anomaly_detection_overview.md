# Anomaly Detection (Overview)

## Say this (60–90 sec)
Anomaly detection finds rare points that don’t look like normal behavior — fraud, equipment failure, network intrusion, bad sensor readings. Unlike classification, you often have few or no labeled anomalies, so you lean on unsupervised or semi-supervised methods. The idea is: learn what “normal” looks like, then flag points that are far from that pattern — by distance, density, isolation, or reconstruction error. Evaluation is tricky: anomalies are rare, so accuracy is useless; you care about precision/recall on the anomaly class, and the cost of false alarms versus misses. In production I always ask: what is normal for this system, how rare are anomalies, and can we get any labels for validation?

## Why it matters
This is a common real-world ML job — especially if you work in reliability, security, or ops. Interviewers test whether you know unsupervised options and metric pitfalls, not just supervised classifiers.

## How it works
- **Unsupervised**: Isolation Forest, DBSCAN (noise points), LOF, autoencoder reconstruction error.
- **Semi-supervised**: train only on normal data (One-Class SVM, some autoencoders).
- **Supervised**: if you have enough labeled anomalies — treat as imbalanced classification.
- **Output**: anomaly score + threshold, or binary flag.

## Tradeoffs
- Use when: labels are scarce, anomalies are diverse, “normal” is easier to define than every failure mode.
- Avoid when: you have balanced labeled classes and a clear decision boundary — plain classification may be simpler and stronger.

## If they dig deeper
- FP vs FN: false alarms fatigue operators; misses can be costly — threshold is a business decision.
- Concept drift: “normal” changes over time — models need retraining or adaptive baselines.
- Next topics: Isolation Forest, DBSCAN, LOF — then deep approaches like autoencoders.
