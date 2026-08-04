# Recall

## Say this (60–90 sec)
**Recall** — also called **sensitivity** or true positive rate — answers: among all actual spam, how much did we catch? Formula: TP / (TP + FN). High recall means we find most spam; low recall means junk and phishing slip into the inbox. Recall cares about the **actual positive** row — it ignores true negatives. In security-sensitive filtering, missing spam (FN) can mean malware links or phishing — so teams often prioritize recall, accepting more false positives. You cannot maximize precision and recall simultaneously in general; raising the spam threshold catches more spam (higher recall) but may flag more ham (lower precision). For imbalanced spam data, recall is often more informative than accuracy.

## Why it matters
Recall vs precision is the core classifier tradeoff. Interviewers test whether you know which metric matches the business goal.

## How it works
- **Formula**: Recall = TP / (TP + FN) = TP / (all actual spam). Also **Sensitivity**, **TPR**.
- **Spam example**: 200 actual spam emails; model catches TP=180, misses FN=20 → recall = 180/200 = **90%**.
- **Not affected by FP directly** — false alarms don’t change recall.
- **Perfect recall (1.0)**: FN=0 — flag every spam; can flag everything as spam (TP=all spam, but FP huge).
- **Baseline recall**: flag all as spam gives recall=1 but precision collapses.

## Tradeoffs
- Use when: **false negatives are costly** — missed fraud, undetected disease, spam/phishing in inbox.
- Optimize recall when: you must catch nearly all positives; human review can filter false alarms downstream.
- Avoid recall alone when: false positives overwhelm users — report precision or F1 alongside.

## If they dig deeper
- Recall vs detection rate in anomaly detection — same idea for rare events.
- Multiclass recall — per-class recall, then macro/micro average.
- Connection to FN rate: FN rate = FN/(TP+FN) = 1 − recall.
