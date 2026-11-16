"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    num = 1
    for i in shape:
        num *= i
    return num

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    return sorted(range(len(values)), key=lambda x: values[x])

# Step 3 - make_op_enums
def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    UnaryOps = enum.Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
    BinaryOps = enum.Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    ReduceOps = enum.Enum('ReduceOps', ['SUM', 'MAX'])
    MovementOps = enum.Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # TODO: wrap np_array as an ndarray and expose shape and dtype
        self._np = np.asarray(np_array)
        self.shape = tuple(int(i) for i in np.shape(np_array))
        self.dtype = self._np.dtype

# Step 5 - lazybuffer_const
def const(value, shape):
    # TODO: Create a new LazyBuffer of the given shape filled with a constant value.
    arr = np.full(shape, value, np.float32)
    return LazyBuffer(arr)
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
import numpy as np
def rand(shape, seed=None):
    # TODO: return a LazyBuffer of uniform random floats in [0, 1) with given shape
    rng = np.random.default_rng(seed)
    out = rng.random(shape)  
    out = out.astype(np.float32)
    return LazyBuffer(out)

# Step 7 - lazybuffer_unary_e
import numpy as np
import enum

def e(self, op):
    # TODO: apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    name = op.name
    x = self._np
    if name == "RELU":
        out = np.maximum(x, 0)
    elif name == "NEG":
        out = -x
    elif name == "EXP":
        out = np.exp(x)
    elif name == "SQRT":
        out = np.sqrt(x)
    elif name == "SIGMOID":
        out = 1.0/(1.0+np.exp(-x))
    elif name == "LOG":
        out = np.log(x)
    else:
        raise ValueError()
    return LazyBuffer(out)
LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
import enum
import numpy as np
def lazybuffer_binary_e(self, op, other):
    x = self._np
    y = other._np

    if op.name == "ADD":
        out = x + y
    elif op.name == "SUB":
        out = x - y
    elif op.name == "MUL":
        out = x * y
    elif op.name == "DIV":
        out = x / y
    elif op.name == "CMPLT":
        out = (x < y).astype(a.dtype)
    elif op.name == "MAX":
        out = np.maximum(x, y).astype(a.dtype)
    else:
        raise ValueError(f"Unsupported binary operation: {op.name}")

    return LazyBuffer(out)

LazyBuffer.lazybuffer_binary_e = lazybuffer_binary_e

# Step 9 - lazybuffer_r
import numpy as np
def r(self, op, axis):
    # TODO: reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    x = self._np
    if op.name == "SUM":
        out = x.sum(axis, keepdims=True)
    elif op.name == "MAX":
        out = x.max(axis, keepdims=True)
    else:
        raise ValueError()
    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
import numpy as np
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    x = self._np
    x = x.reshape(new_shape)
    return LazyBuffer(x)

# Step 11 - lazybuffer_expand
import numpy as np
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    x = self._np
    buffer_arr = np.zeros(new_shape, dtype=int)
    x = x + buffer_arr 
    return LazyBuffer(x)

# Step 12 - lazybuffer_permute
import numpy as np
def permute(self, order):
    # TODO: return a new LazyBuffer with axes reordered according to order
    x = self._np
    return LazyBuffer(x.transpose(order))

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # TODO: record needs_input_grad, requires_grad, and parents for backprop
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = False
        for t in tensors:
            if t.requires_grad == True:
                self.requires_grad = True
                self.parents = tensors
            elif t.requires_grad == None:
                self.requires_grad = None

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # TODO: attach forward and backward stubs to Function that raise NotImplementedError
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")
    Function.forward = forward
    Function.backward = backward

    return Function

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # TODO: build the Function, run forward on the input buffers, wrap in a
    # Tensor, and link out._ctx when a gradient is needed.
    ctx = cls(*tensors)
    buffer_list = [t.lazydata for t in tensors]
    out=ctx.forward(*buffer_list, **kwargs)
    result = Tensor(out,requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        result._ctx = ctx
    return result

# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

