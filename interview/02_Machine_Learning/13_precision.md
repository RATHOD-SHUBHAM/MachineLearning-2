# Precision

## Say this (60–90 sec)
**Precision** answers: among everything we flagged as spam, how much was actually spam? Formula: TP / (TP + FP). High precision means when the model says “spam,” it’s usually right — few false alarms. Low precision means lots of legitimate email wrongly quarantined. Precision cares about the **predicted positive** column — it does not count missed spam (FN). In spam filtering, low precision frustrates users who keep finding good mail in the spam folder. I use precision when **false positives are costly** — blocking valid transactions, flagging innocent users, crying wolf on alerts. It pairs naturally with recall: you can often raise precision by being stricter (fewer spam flags), but you may miss more spam. Always define positive class = spam when stating the formula.

## Why it matters
Precision vs recall is a classic interview tradeoff. Precision shows you understand the cost of false alarms.

## How it works
- **Formula**: Precision = TP / (TP + FP). Denominator = all predicted spam.
- **Spam example**: TP=80, FP=20 → precision = 80/100 = **80%** — 20% of spam flags were false alarms.
- **Not affected by FN directly** — missing spam doesn’t enter the formula.
- **Perfect precision (1.0)**: FP=0 — never flag ham as spam; may mean very conservative threshold (high FN).
- Related: **Positive Predictive Value (PPV)** — same as precision.

## Tradeoffs
- Use when: false positives hurt — user trust, manual review load, wrongly blocked payments.
- Optimize precision when: you prefer silence over false alarms (high-stakes alerts with expensive investigation).
- Avoid relying on precision alone when: missing positives is dangerous — check recall too (phishing, fraud).

## If they dig deeper
- Precision@k in retrieval — top k results, not full classifier threshold.
- Multiclass precision — macro (average per class) vs micro (pool all TP/FP).
- Why threshold tuning moves precision — lower threshold → more spam flags → often lower precision, higher recall.
