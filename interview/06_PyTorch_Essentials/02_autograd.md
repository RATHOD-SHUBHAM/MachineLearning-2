# Autograd (PyTorch)

## Say this (60–90 sec)
Autograd is PyTorch's automatic differentiation engine. When you create a tensor with requires_grad=True, operations on it build a computational graph. Calling loss.backward() computes gradients via reverse-mode autodiff and stores them in tensor.grad. Leaf tensors — usually model parameters — accumulate .grad; you read gradients and pass them to the optimizer. Non-leaf tensors' grads are freed unless you retain_grad. Inference and eval don't need graphs — wrap code in torch.no_grad() to save memory and speed. Detach breaks the graph: x.detach() gives same values, no gradient. Common pattern: forward → loss → zero_grad → backward → step. If grad is None, you forgot to set requires_grad or detached too early. Hands-on reference: [`PyTorch/chap_2_autograd.py`](../../PyTorch/chap_2_autograd.py).

## Why it matters
Training is autograd plus optimizer. Interviewers probe whether you understand when graphs are built, when they're disabled, and common gradient bugs.

## How it works
- **requires_grad**: enables tracking on a tensor (Parameters default to True).
- **backward()**: computes ∂loss/∂x for all leaves; scalar loss required (or pass gradient arg).
- **zero_grad()**: optimizer.zero_grad() clears old grads — they accumulate by default.
- **no_grad()**: context manager disables graph — inference, metric computation.
- **detach()**: tensor shares data, no history — stop backprop through this path.

## Tradeoffs
- Use when: training anything differentiable — always backward after forward+loss.
- Avoid when: keeping graph during long inference loops — memory leak; use no_grad or inference_mode.

## If they dig deeper
- inference_mode vs no_grad — inference_mode is stricter/faster; no_grad for training-side eval blocks.
- create_graph=True — needed for higher-order derivatives.
- Gradient checkpointing — trade compute for memory in backward.
