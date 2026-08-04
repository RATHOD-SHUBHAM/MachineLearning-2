# True Positive, True Negative, False Positive, False Negative

## Say this (60–90 sec)
These four terms come from comparing predictions to ground truth for a **binary** classifier. Fix the **positive class** first — in spam detection, **positive = spam**. **True Positive (TP)**: we predicted spam, and it really was spam — a correct detection. **True Negative (TN)**: we predicted not spam, and it really was not spam — correct inbox delivery. **False Positive (FP)**: we predicted spam, but it was not spam — a false alarm; legitimate mail blocked. **False Negative (FN)**: we predicted not spam, but it was spam — a miss; junk slipped through. “True/False” refers to whether the **prediction** was correct; “Positive/Negative” refers to which **class we predicted**. Memory trick: the second word (Positive/Negative) is always what the **model said**; True means the model was right. Every classification metric is built from TP, TN, FP, FN counts.

## Why it matters
If you fumble TP/FP, every downstream metric is wrong in the interview. This is the most tested vocabulary in ML interviews.

## How it works
- **Positive class**: spam (our running example).
- **TP**: predicted **spam** (+), actual spam (+) → correct.
- **TN**: predicted **not spam** (−), actual not spam (−) → correct.
- **FP**: predicted **spam** (+), actual not spam (−) → wrong; Type I error in some fields.
- **FN**: predicted **not spam** (−), actual spam (+) → wrong; Type II error.
- Example: 100 emails — 10 spam, 90 ham. Model flags 8 as spam; 7 truly spam, 1 ham wrongly flagged. TP=7, FP=1, FN=3 (3 spam missed), TN=89.

## Tradeoffs
- Use when: defining any metric; always state positive class explicitly.
- Avoid when: mixing up “positive” with “good outcome” — positive means the class label we care about detecting, not morally “good.”
- In spam: FP hurts user trust (blocked good mail); FN hurts security and UX (spam in inbox).

## If they dig deeper
- Sensitivity = recall = TP/(TP+FN) — catch rate among actual spam.
- Specificity = TN/(TN+FP) — correct rejection among actual ham.
- One person’s FP is another’s FN if you flip the positive class — always define it upfront.
