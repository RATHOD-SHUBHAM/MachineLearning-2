# Accuracy

## Say this (60–90 sec)
**Accuracy** is the fraction of all predictions that are correct: (TP + TN) / (TP + TN + FP + FN). In spam detection, that’s all emails correctly classified as spam or not spam, divided by total emails. It’s intuitive when classes are balanced — roughly half spam, half not. It **lies with imbalance**. If only 2% of email is spam, a model that always predicts “not spam” gets **98% accuracy** while catching zero spam — useless for the business goal. Accuracy treats FP and FN equally, but blocking one good email vs missing one phishing email may have very different costs. I report accuracy alongside the confusion matrix, but for skewed problems I lead with precision, recall, F1, or ROC-AUC depending on whether false alarms or misses hurt more.

## Why it matters
“What's your model accuracy?” is a common question — interviewers expect you to push back on imbalance and cost asymmetry.

## How it works
- **Formula**: Accuracy = (TP + TN) / total = correct / all.
- **Spam example**: 1000 emails, 20 spam, 980 ham. Model: TP=15, FN=5, FP=10, TN=970. Accuracy = (15+970)/1000 = **98.5%** — sounds great, but recall = 15/20 = 75% (missed 5 spam).
- **Baseline accuracy**: majority-class classifier — always predict the common class.
- **When honest**: classes roughly balanced; FP and FN costs similar; all errors equally bad.

## Tradeoffs
- Use when: balanced multiclass or binary; quick sanity check vs majority baseline.
- Avoid when: heavy class imbalance (spam, fraud, disease screening); asymmetric error costs.
- Avoid as sole metric in production — pair with precision/recall or PR-AUC.

## If they dig deeper
- Balanced accuracy: average of recall per class — (sensitivity + specificity)/2 for binary.
- MCC (Matthews correlation) — single metric that handles imbalance better than raw accuracy.
- Why Kaggle uses accuracy sometimes — balanced or multiclass tasks where it’s appropriate.
