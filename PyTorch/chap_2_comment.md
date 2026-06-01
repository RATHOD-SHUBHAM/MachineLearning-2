# Chapter 2 — Autograd: Gradients of the Loss

Notes for `chap_2_autograd.py`. See also [pytorch-ml-study-roadmap.md](../docs/pytorch-ml-study-roadmap.md).

---

## What does “compute the gradient of the loss” mean?

You have a **scalar loss** $L$ (one number, e.g. MSE). You have **parameters** $\theta$ (weights, biases — many numbers).

The **gradient** is the set of **partial derivatives** of $L$ with respect to **each** parameter:

$$
\frac{\partial L}{\partial w_1},\ \frac{\partial L}{\partial w_2},\ \ldots,\ \frac{\partial L}{\partial b}
$$

**Intuition:** For each weight, the gradient answers:

> “If I increase this weight a tiny bit, does loss go **up** or **down**, and how **fast**?”

| Sign of ∂L/∂w | Meaning | Update direction |
|---------------|---------|------------------|
| **Positive** | Increasing $w$ increases loss | **Decrease** $w$ |
| **Negative** | Increasing $w$ decreases loss | **Increase** $w$ |
| **Large magnitude** | Loss is very sensitive to that parameter | Take a smaller or more careful step |

So “compute the gradient” means: **find ∂L/∂θ for every trainable parameter θ** — not a separate magic step, but the calculus that tells you how to improve each weight and bias.

### Tiny example (one weight)

- Model: $\hat{y} = w x$
- Loss (MSE-style): $L = \frac{1}{2}(\hat{y} - y)^2$

**Gradient descent update:**

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

$\eta$ = learning rate. The **minus** moves $w$ **downhill** on the loss surface (opposite to the gradient).

Same idea for **bias** $b$: compute ∂L/∂b, then update $b$.

---

## Where this fits in a full training step

| Step | What happens | PyTorch (typical) |
|------|----------------|-------------------|
| 1. Forward | Use current weights/bias to predict | `y_pred = model(x)` |
| 2. Loss | Compare prediction to target → one scalar | `loss = criterion(y_pred, y)` |
| 3. **Compute gradients** | ∂L/∂(each parameter) | `loss.backward()` |
| 4. Update | Move parameters to reduce loss | `optimizer.step()` |
| 5. Clear old grads | Avoid mixing gradients across steps | `optimizer.zero_grad()` (start of next step) |

```python
optimizer.zero_grad()   # clear gradients from previous step
loss = criterion(model(x), y)
loss.backward()         # compute ∂loss/∂w, ∂loss/∂b, ... → param.grad
optimizer.step()        # w = w - lr * grad_w  (SGD-style)
```

- **“Compute gradients”** in code ≈ **`loss.backward()`**
- **“Adjust weights / bias”** ≈ **`optimizer.step()`**

**Gradient descent** is the algorithm (move opposite to the gradient). The **optimizer** (SGD, Adam, …) is the object that applies the update rule using those gradients.

### Training loop (high level)

1. Forward pass → compute **loss**
2. Backward pass → **compute gradients** of loss w.r.t. each parameter
3. **Update** parameters (optimizer)
4. **Repeat** for many steps / epochs

---

## What actually does the gradient math?

| Approach | Role |
|----------|------|
| **By hand** | Chain rule on the formula for $L$ |
| **Deep networks** | **Backpropagation** — efficient chain rule through the full computational graph |
| **PyTorch** | **`torch.autograd`** — builds the graph on the forward pass; `.backward()` runs backprop and fills `tensor.grad` for tensors with `requires_grad=True` |

- **Loss function** defines *what* to minimize.
- **Gradients** tell *how* to change each parameter.
- **Optimizer** applies the actual parameter change.

From PyTorch docs (paraphrased): parameters are adjusted according to the **gradient of the loss with respect to each parameter**. Set `requires_grad=True` on parameters you want to optimize so autograd tracks them.

---

## `requires_grad` and `.backward()`

**Common misconception:** `requires_grad=True` does **not** store gradients during the forward pass. Forward pass builds the graph; `.backward()` computes and writes gradients.

### Forward vs backward

| Phase | What `requires_grad=True` does | What gets stored |
|--------|--------------------------------|------------------|
| **Forward pass** | Tells autograd to **track** this tensor in the computation graph (DAG) | **Not** the gradient yet — autograd records **operations** (`grad_fn` on outputs) |
| **Backward pass** (`loss.backward()`) | Autograd **computes** ∂loss/∂(each tracked leaf) via chain rule | Gradients go into **`.grad`** on those leaf tensors |

- **`requires_grad=True`** → “Include me in the graph; when `.backward()` runs, compute my gradient.”
- **`.backward()`** → Runs differentiation using the graph built during forward and **fills** `.grad`.

Gradients are **not** stored during forward — only the graph is built so backward can run later.

### What happens on forward

During forward, autograd:

1. Runs the operation (e.g. `y = w * x + b`)
2. Links each op into a **DAG** (directed acyclic graph)
3. Attaches **`grad_fn`** to result tensors (e.g. `MulBackward0`) — how to differentiate that op on the way back

**`.grad` on parameters is still `None` before backward** — forward only records the recipe for backprop.

### What happens on `.backward()`

Calling `.backward()` on the **loss** (root of the graph):

1. Starts at the loss and walks the DAG **backward**
2. Uses each op’s backward rule (chain rule)
3. **Accumulates** gradients into **`.grad`** on **leaf** tensors with `requires_grad=True`

`.backward()` **computes** derivatives using the graph from forward — it does not read pre-stored gradients from `requires_grad`.

### Interview one-liner (`requires_grad` / `.backward()`)

> “`requires_grad=True` marks a tensor so autograd tracks it in the computation graph during the forward pass. Forward records operations, not parameter gradients. Calling `.backward()` on the loss applies the chain rule backward through the graph and stores ∂loss/∂θ in each leaf tensor’s `.grad` attribute.”

---

## Interview one-liner

> “We minimize a scalar loss over model parameters. Computing the gradient means taking the partial derivative of the loss with respect to each weight and bias. Those derivatives tell us the direction and steepness of change. Gradient descent updates parameters in the opposite direction of the gradient; in PyTorch, `backward()` computes gradients via autograd and `optimizer.step()` applies the update.”

---

## Common follow-ups

1. **Why `optimizer.zero_grad()`?** Gradients **accumulate** by default; clear them each step or old and new grads mix.
2. **Loss vs cost?** Often used interchangeably — the scalar objective you minimize.
3. **SGD vs Adam?** Same gradient computation (`backward()`); different update rules in `optimizer.step()`.

### Practice question

For $L = (w x - y)^2$, what does ∂L/∂w represent, and why do we **subtract** it (times learning rate) from $w$?

---

