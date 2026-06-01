"""
https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

When training neural networks, the most frequently used algorithm is back propagation. 
In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called torch.autograd. 
It supports automatic computation of gradient for any computational graph.

Consider the simplest one-layer neural network, with input x, parameters w and b, and some loss function. 
w and b are parameters, which we need to optimize. Thus, we need to be able to compute the gradients of loss function with respect to those variables. In order to do that, we set the requires_grad property of those tensors.
"""

"""
Gradient Descent Algorithm:
1. Define the Cost Function: Create a function that calculates how far off a model's predictions are from the actual values.
2. Initialize Parameters: Start the model with random weights and biases.
3. Calculate the Gradient: Compute the partial derivatives (the gradient) of the cost function with respect to every weight and bias. 
This tells the model not only which direction to move, but how steep the slope is.
4. Update the Parameters: Shift the parameters in the exact opposite direction of the gradient to minimize the cost.
5. Iterate: Repeat steps 3 and 4 until the changes in parameters no longer significantly reduce the loss (known as convergence)
"""

"""
By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation.

Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of Function objects. 
In this DAG, leaves are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

In a forward pass, autograd does two things simultaneously:

* run the requested operation to compute a resulting tensor

* maintain the operation’s gradient function in the DAG.

The backward pass kicks off when .backward() is called on the DAG root. autograd then:

* computes the gradients from each .grad_fn,

* accumulates them in the respective tensor’s .grad attribute

* using the chain rule, propagates all the way to the leaf tensors.
"""

"""
A function that we apply to tensors to construct computational graph is in fact an object of class Function. 
This object knows how to compute the function in the forward direction, and also how to compute its derivative during the backward propagation step. 
A reference to the backward propagation function is stored in grad_fn property of a tensor.
"""
import torch

x = torch.randn(5, requires_grad=True) # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

y = x + 2

z = y * y * 3

print(z) # Vector output of len 5
print(f"Gradient function for z = {z.grad_fn}") 


"""
“.backward() needs a seed gradient at the root. For a scalar loss, PyTorch uses 1 by default. 
For vector or matrix outputs, you must either reduce to a scalar (e.g. loss.backward()) or pass a gradient argument with the same shape as the output.”
"""
z.backward(torch.ones_like(z)) # The argument is a vector of shape similar to z
print(x.grad)

"""
You print x.grad because that is where PyTorch puts the answer to the question you care about for training:

“How does the final objective change if each element of x changes?”

In a tutorial, you print(x.grad) to: Confirm backward ran and reached x
"""


"""
Clear Gradients:
You clear gradients so each training step uses only that step’s batch, not a running total from earlier batches.
“backward() accumulates gradients into .grad by default. We call optimizer.zero_grad() at the start of each step so parameters get only the current batch’s gradients, not a sum over previous batches.”

Layman analogy: walking directions
Each batch is a friend telling you which way to walk right now.

Correct (with zero_grad):

Friend 1: “Go north 2 steps.” → you walk 2 north.
Erase the note.
Friend 2: “Go east 5 steps.” → you walk 5 east only.

Without zero_grad:
Friend 1: “Go north 2 steps.” → you write “north 2”.
Friend 2: “Go east 5 steps.” → you add and write “north 2 and east 5”.
You try to follow both at once in one step — that’s not what either friend meant for this step alone.
Training wants: one step = one batch’s direction, not north 2 + east 5 mashed together.
"""
x.grad.zero_()




# =================================================================================================================
# All Together Now — one training step without an optimizer (PyTorch autograd tutorial style)
# =================================================================================================================

# Learnable parameters: 3 input features → 5 outputs (like a tiny linear layer)
# Shape (3, 5): each column is one output neuron's weights
weights = torch.randn(3, 5, requires_grad=True)
bias = torch.randn(5, requires_grad=True)

print(f"weights = {weights}")
print(f"bias = {bias}")

# Input (3 features). requires_grad=True only if you also want ∂loss/∂x (optional in real training)
x = torch.randn(3)
x.requires_grad_(True)

print(f"x = {x}")

# Forward: z = x @ weights + bias  →  shape (5,)
z = torch.matmul(x, weights) + bias
print(f"z = {z}")

# Scalar loss (BCE with logits); target must match z.shape
target = torch.randn(5)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
print(f"loss = {loss}")

# Backward: compute ∂loss/∂weights, ∂loss/∂bias, ∂loss/∂x → stored in .grad
loss.backward()
print(f"Gradient for weights = {weights.grad}")
print(f"Gradient for bias = {bias.grad}")

# Manual SGD step (lr=0.01): update parameter values without building a new autograd graph
# .data avoids tracking the update itself as part of the graph
weights.data = weights.data - 0.01 * weights.grad
bias.data = bias.data - 0.01 * bias.grad

print(f"weights = {weights}")
print(f"bias = {bias}")

# Clear .grad so the next backward() does not add on top of these values (see zero_grad notes above)
weights.grad.zero_()
bias.grad.zero_()
if x.grad is not None:
    x.grad.zero_()


"""
Step 1: Input Layer

Your input has 3 features

x = [x1, x2, x3]

Visually:

Input Layer
------------

x1
x2
x3


Step 2: Weights Matrix

You created:
weights.shape = (3, 5)

Meaning:
3 input neurons
5 output neurons

Every input connects to every output.

          Output Neurons
         y1  y2  y3  y4  y5
        ---------------------
x1 --->  w11 w12 w13 w14 w15
x2 --->  w21 w22 w23 w24 w25
x3 --->  w31 w32 w33 w34 w35

Matrix form:

      y1    y2    y3    y4    y5
x1   w11   w12   w13   w14   w15
x2   w21   w22   w23   w24   w25
x3   w31   w32   w33   w34   w35

Total weights:
3 × 5 = 15 weights


Step 3: Bias

You also have

bias.shape = (5,)

One bias per output neuron.

Output Neuron  Bias

y1             b1
y2             b2
y3             b3
y4             b4
y5             b5

Total:
5 biases


Full Network
                Output Layer

             y1   y2   y3   y4   y5
             ●    ●    ●    ●    ●
            /|\  /|\  /|\  /|\  /|\
           / | \/ | \/ | \/ | \/ | \
          /  | /\ | /\ | /\ | /\ |  \

        ●      ●      ●
       x1     x2     x3

         Input Layer

Every input neuron connects to every output neuron.


What does matmul do?
When you do
z = x @ weights + bias

PyTorch computes:

For neuron 1:
z1 = x1*w11 + x2*w21 + x3*w31 + b1
and so on for each neuron.

Result:
z = [z1, z2, z3, z4, z5]
Shape:
(5,)

This code is equivalent to:
layer = nn.Linear(
    in_features=3,
    out_features=5
)

PyTorch internally stores:
weights -> (3,5)
bias    -> (5,)

and performs:
output = x @ weights + bias

followed by backpropagation to compute gradients for all 15 weights + 5 biases = 20 learnable parameters.
"""


