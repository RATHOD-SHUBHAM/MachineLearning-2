# Perceptron / Neuron

## Say this (60–90 sec)
A single neuron is the basic building block of neural networks. It takes inputs, multiplies each by a weight, adds them up with a bias, and passes the result through an activation function. Mathematically: output equals activation of w dot x plus b. The weights learn which inputs matter; the bias shifts the decision boundary. A classic perceptron uses a step function and can only solve linearly separable problems — like AND but not XOR. Modern neurons swap the step for smooth activations like ReLU so we can stack layers and train with gradient descent. When I draw a neuron, I think: linear transform first, then nonlinearity — that's what lets deep networks approximate complex functions.

## Why it matters
Every layer in an MLP, CNN, or transformer block is built from this same pattern: weighted sum plus activation. Interviewers start here to see if you understand what "learning" actually changes — the weights and biases — and why nonlinearity is non-negotiable for depth.

## How it works
- **Inputs** `x = [x1, ..., xd]`, **weights** `w`, **bias** `b`.
- **Pre-activation (logit)**: `z = w·x + b` — affine transform.
- **Activation**: `a = σ(z)` — introduces nonlinearity (ReLU, sigmoid, etc.).
- **Perceptron (Rosenblatt)**: binary classifier with step/threshold activation; updates weights with a simple rule when misclassified.
- **Single neuron = linear classifier** with sigmoid/softmax; **multi-layer** stacks neurons to learn nonlinear boundaries.

## Tradeoffs
- Use when: explaining the atomic unit of NNs, or why XOR needs hidden layers.
- Avoid when: treating a perceptron as equivalent to a deep network — one neuron has limited capacity.

## If they dig deeper
- Perceptron convergence theorem: guaranteed to find a separating hyperplane if one exists (linearly separable data).
- Biological analogy vs reality — real neurons are spiking, leaky, and temporal; ML neurons are a useful abstraction.
- Bias as a weight on a constant input of 1 — same math, easier to implement in matrix form.
