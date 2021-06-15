import torch
import torch.nn as nn
from typing import Tuple


class Unsqueeze(nn.Module):
    """
    Custom class for pytorch Unsqueeze layer. It conforms to our representation and ONNX.

    Attributes
    ----------
    axes : Tuple
        List of indices at which to insert the singleton dimension.
    """

    def __init__(self, axes: Tuple):

        super().__init__()
        self.axes = axes

    def forward(self, x: torch.Tensor):

        for ax in self.axes:
            x = torch.unsqueeze(x, ax)
        return x


class Reshape(nn.Module):
    """
    Custom class for pytorch Reshape layer. It conforms to our representation and ONNX.
    Torch reshape function does not support zeros in the shape, therefore it cannot support the allow_zero attribute
    of our representation.

    Attributes
    ----------
    shape : Tuple
        Tuple which specifies the output shape
    """

    def __init__(self, shape: Tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        x = torch.reshape(x, self.shape)
        return x


class Flatten(nn.Module):
    """
    Custom class for pytorch Flatten layer. It conforms to our representation and ONNX.

    Attributes
    ----------
    axis : int
        Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output.
        The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value
        means counting dimensions from the back. When axis = 0, the shape of the output tensor is
        (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n).
        N.B: it works assuming the initial batch dimension. (default: 0)
    """

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor):
        # Given our representation we exclude the batch dimension from the operation.
        x = torch.flatten(x, 1, self.axis - 1)
        x = torch.flatten(x, self.axis, -1)
        return x
