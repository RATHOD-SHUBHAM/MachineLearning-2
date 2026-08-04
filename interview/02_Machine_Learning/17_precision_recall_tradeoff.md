# Precision–Recall Tradeoff and Threshold

## Say this (60–90 sec)
Most classifiers output a **probability** or score, not a hard label. We convert score to spam/not-spam with a **threshold** — default often 0.5, but that’s arbitrary. **Lower threshold** → flag more email as spam → **higher recall**, usually **lower precision** (more false alarms). **Higher threshold** → flag only confident spam → **higher precision**, usually **lower recall** (more missed spam). That’s the precision–recall tradeoff: moving the threshold slides along a curve; you rarely maximize both at once. Choose threshold from business cost — if missing phishing is unacceptable, accept lower precision and set threshold low. If users hate blocked newsletters, raise threshold. On imbalanced spam data, PR curves are often more informative than ROC. Tune threshold on validation set, not test set.

## Why it matters
Production ML is threshold selection, not just model training. Interviewers want you to connect scores to decisions and costs.

## How it works
- **Score → label**: if P(spam) ≥ t, predict spam; else not spam.
- **t ↓**: more positives predicted → TP and FP rise, FN falls → recall up, precision often down.
- **t ↑**: fewer positives → FP and TP fall, FN rises → precision up, recall often down.
- **PR curve**: plot precision vs recall as t sweeps from 0 to 1.
- **F1-max threshold**: pick t that maximizes F1 on validation — one automatic choice, not always optimal for business.
- **Cost-sensitive threshold**: minimize expected cost = C_FP×FP + C_FN×FN over t.

## Tradeoffs
- Use lower threshold when: missing spam/fraud is expensive (security, compliance).
- Use higher threshold when: false alarms annoy users or trigger costly manual review.
- Avoid fixed 0.5 when: classes imbalanced or scores miscalibrated — always tune on val.
- Avoid tuning threshold on test set — overfits your reported metrics.

## If they dig deeper
- Platt scaling / isotonic regression — calibrate scores so 0.7 means ~70% spam rate.
- Precision@fixed recall — “we need 95% recall, what precision do we get?”
- Why default sklearn 0.5 assumes balanced classes and calibrated scores — often wrong for spam.
