# Learning Rate and Schedules

## Say this (60–90 sec)
Learning rate is the step size for weight updates — the single most important hyperparameter. Too high and loss oscillates or diverges; too low and training crawls or gets stuck in bad minima. I usually start with a reasonable default — like 1e-3 for Adam, higher for SGD with momentum — and watch the loss curve. Learning rate schedules reduce LR over time: step decay drops it every N epochs; cosine annealing smoothly decreases to near zero; warmup starts small and ramps up, common in transformers to stabilize early Adam steps. Reduce-on-plateau drops LR when validation loss stalls. The idea is big steps early for speed, small steps late for fine convergence. LR finder sweeps rates and plots loss to pick a good starting point. If training is unstable, LR is the first knob I turn.

## Why it matters
Bad LR looks like a broken model. Schedules are standard in SOTA recipes — mentioning warmup and cosine shows you've trained real models, not just toy examples.

## How it works
- **Fixed LR**: constant η throughout — simple, may need manual decay.
- **Step decay**: multiply η by factor (e.g., 0.1) every fixed epochs.
- **Cosine annealing**: η follows cosine curve from max to min over total steps.
- **Warmup**: linear increase from 0 to peak LR over first W steps — stabilizes Adam/transformer training.
- **ReduceLROnPlateau**: monitor val metric, reduce when no improvement.
- **One-cycle**: LR up then down in one cycle — fast training trick (super convergence).

## Tradeoffs
- Use when: any gradient-based training; schedules for long runs and transformers; warmup with Adam and large batch.
- Avoid when: tiny datasets where full schedule never completes — may overcomplicate; grid search LR first before exotic schedules.

## If they dig deeper
- Linear scaling rule: LR scales with batch size (linear for SGD, sqrt for Adam — rules vary).
- Weight decay interaction — higher LR sometimes pairs with stronger decay.
- Cyclical LR — periodic increases to escape sharp minima.
