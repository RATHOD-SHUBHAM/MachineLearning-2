import torch

"""
Tensors are a specialized data structure that are very similar to arrays and matrices. 
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators. 
In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data. 
Tensors are also optimized for automatic differentiation. 

"What is a tensor, and how is it different from a NumPy array?"
1. Data structure — multi-dimensional array (0D=scalar, 1D=vector, 2D=matrix, nD=tensor)
2. Hardware — PyTorch tensors can live on CPU/GPU/MPS; NumPy is CPU-only
3. Autograd — tensors track computation for backprop (requires_grad=True)
4. Interop — zero-copy sharing via .numpy() / torch.from_numpy() when memory layout allows
"""

x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)

""" shape is a tuple of tensor dimensions """
shape = (2,3) # 2 rows, 3 columns → 2×3 = 6 elements
rand_tensor = torch.rand(shape)
print(rand_tensor)

ones_tensor = torch.ones(shape)
print(ones_tensor)

zeros_tensor = torch.zeros(shape)
print(zeros_tensor)



"""
Attributes of a Tensor
Tensor attributes describe their shape, datatype, and the device on which they are stored.
"""
x = (3,4)
tensor = torch.rand(x)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


#  =================================================================================================================
# Rank
#  =================================================================================================================
"""
Rank is the number of dimensions (axes) a tensor has. It tells you how many indices you need to point to one element.

PyTorch exposes this as .ndim (same idea as NumPy's .ndim).

## Rank vs shape (easy to mix up)
Rank = how many dimensions → len(tensor.shape) or tensor.ndim
Shape = how big each dimension is → e.g. (2, 3) means 2 along axis 0, 3 along axis 1

Intuition: count the nesting levels
Tensor                  Example	                        Rank	        Why
Scalar              torch.tensor(5)                      0      One number, no axis
Vector              torch.tensor([1, 2, 3])              1      One list → one axis
Matrix              torch.rand(2, 3)                     2      Rows and columns → two axes
3D batch of images  torch.rand(32, 3, 224, 224)          4      batch, channels, height, width


Example:
t = torch.rand(2, 3, 4)
t.ndim   # 3  (rank)
t.shape  # torch.Size([2, 3, 4])  (size along each axis)


"Rank is the number of dimensions in a tensor. A vector has rank 1, a matrix rank 2. In PyTorch, rank equals tensor.ndim, which is the length of the shape tuple."
→ 0. torch.tensor(5).shape is torch.Size([]) — empty shape, zero dimensions.
Follow-up: "What's the rank of a batch of 32 RGB images of size 224×224?"
→ 4: (batch, channels, height, width).
"""

#  =================================================================================================================
# Arthematic Operations
#  =================================================================================================================
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

z1 = x + y
print(z1)

z_1 = torch.add(x, y)
print(z_1)

z2 = x * y
print(z2)

z3 = x / y
print(z3)


# inplace operations
x.add_(y) # _ will Modify x in place
print(x)


# Item() method
x = torch.rand(5, 3)
print("x: ", x)
print(x[1,1])
y = x[1,1].item() # you can only use .item() to get a scalar value out of a tensor. ie only one element, when there are multiple elements, you will get an error.
print(y)


# Reshape a tensor
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(2, 8)
print(z)
w = x.view(2, 2, 4) #  one tensor with 2 blocks, each 2×4 (2 rows, 4 columns)
print(w)
u = x.view(-1,8) # -1 means "as many as needed", pytorch will figure out the number of rows needed to make the tensor have 8 columns.
print(u)

x = torch.arange(12)
print(x)
y = x.view(3, 4)
print(y)
z = x.view(3, 2, 2)
print(z)
# w = x.view(2,5) # error
# w = x.view(-1,5) # error

"""
view vs reshape

view — requires contiguous memory; fails otherwise
reshape — may copy if not contiguous
After x.transpose(0, 1), use reshape, not view.

oth change shape, not data. The difference is how they handle memory layout.

                     view	                     reshape
Goal:               New shape                   New shape
Memory:             Must be contiguous          Works either way
If not contiguous:  Error                       Copies data (safe)
If contiguous:      Returns a view (no copy)    Returns a view (no copy)

So: same result shape, different rules and guarantees.

view is safer because it requires contiguous memory, so you can't change the shape if the memory is not contiguous.
reshape is more flexible because it can work with non-contiguous memory, but it may copy the data if the memory is not contiguous.


When they diverge — the classic example
Transpose changes how you read the data without moving it in memory → tensor becomes non-contiguous

What "contiguous" means (intuition)
Memory is stored as a flat 1D array. A tensor is contiguous if reading it row-by-row matches that flat order.
- view says: "I'll just reinterpret this flat memory." → fails if layout doesn't match.
- reshape says: "I'll give you the right shape; I'll copy if I have to."

"view and reshape both change tensor shape without changing element count. view only works on contiguous tensors and never copies. reshape is safer: it returns a view when possible, otherwise it copies. After ops like transpose or permute, use reshape or call .contiguous() before view."
"""

# Reshape a tensor in a way that preserves the number of elements
x = torch.rand(4, 4)
t = x.transpose(0, 1)   # or x.T

print(t.is_contiguous())  # False, A tensor is contiguous if reading it row-by-row matches that flat order.

# y = t.view(16)      # ❌ RuntimeError: view size is not compatible...
z = t.reshape(16)   # ✅ Works — PyTorch copies data into contiguous layout
print(z)


#  =================================================================================================================
# Default CPU case,
#  =================================================================================================================

"""
Numpy and Tensor will point to same memory on cpu, so if we change Tensor it will change numpy and vice versa.
But this is not always the case,  there are some important caveats to keep in mind.
Important caveats (interview gold)
1. Only on CPU
GPU tensors don't share memory with NumPy.
2. Dtype must match
NumPy defaults to float64; PyTorch often uses float32

On CPU, torch.from_numpy() and .numpy() can share the same underlying buffer, so in-place changes in one reflect in the other. That doesn't apply to GPU tensors, and torch.tensor() / np.array() always copy. You also need compatible dtypes and to avoid ops that create a new buffer like .clone().


In PyTorch, .to(device) is the standard method used to move data or models between the host CPU and accelerator hardware like a GPU

"""

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
x = torch.rand(4, 4)
x = x.to(device)
print(x)




