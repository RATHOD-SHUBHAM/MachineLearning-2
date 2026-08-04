# Train vs Eval / no_grad

## Say this (60–90 sec)
PyTorch models behave differently in train vs eval mode. model.train() enables training behavior — dropout randomly drops units, batch norm uses batch statistics. model.eval() disables dropout and makes batch norm use running averages accumulated during training. Always call eval before validation or inference, and train() before training loop. Even in eval mode, autograd still builds graphs by default — so wrap inference in torch.no_grad() or torch.inference_mode() to skip gradient storage and save memory. Pattern: training loop with train + backward; validation with eval + no_grad. Forgetting eval() is a classic bug — validation loss looks wrong because dropout is still active. Forgetting no_grad wastes GPU memory on val set. inference_mode is faster than no_grad for pure inference.

## Why it matters
Silent correctness bug in almost every codebase. Quick interview question with real production impact.

## How it works
- **model.train()**: dropout ON, batch norm uses batch stats, some custom layers train-only.
- **model.eval()**: dropout OFF, batch norm uses running_mean/var.
- **torch.no_grad()**: no graph, no grad storage — validation, metrics, generation.
- **torch.inference_mode()**: stronger disable — inference-only, slightly faster.
- **Still need eval() with no_grad()** — they solve different problems.

## Tradeoffs
- Use when: eval+no_grad for val/test/inference; train() for training epoch.
- Avoid when: leaving model in eval during training — BN and dropout broken; or running inference without no_grad on large batches.

## If they dig deeper
- model.eval() doesn't stop gradients — only affects module behavior; no_grad stops autograd.
- BatchNorm with batch size 1 in eval — uses running stats, OK; in train — unstable.
- torch.set_grad_enabled(False) — global switch alternative to context manager.
