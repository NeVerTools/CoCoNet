import numpy as np
from PyQt5.QtCore import QRectF, QLineF, QPointF
from never2.core.controller.pynevertemp.tensor import Tensor


def truncate(f: float, n: int) -> str:
    """
    Truncates/pads a float f to n decimal places without rounding

    """

    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def get_sides_of(rect: QRectF):
    """
    This method returns the sides of a rect as a dictionary.
    Parameters
    ----------
    rect : QRectF
        The rect to inspect.
    Returns
    ----------
    dict
        A container of the sides represented as QLineF
        objects and their position as a key.

    """

    return {"top": QLineF(rect.topLeft(), rect.topRight()),
            "right": QLineF(rect.topRight(), rect.bottomRight()),
            "bottom": QLineF(rect.bottomLeft(), rect.bottomRight()),
            "left": QLineF(rect.bottomLeft(), rect.topLeft())}


def get_midpoint(label: str, side: QLineF) -> (QPointF, QPointF):
    """
    Procedure to compute the midpoint of a rectangle side.

    Parameters
    ----------
    label : str
        The side label ("top", "right", "bottom", "left")
    side : QLineF
        The line representing the side.

    Returns
    ----------
    tuple
        The two coordinates of the mid-point.

    """

    mid_x = (side.x1() + side.x2()) / 2
    mid_y = (side.y1() + side.y2()) / 2

    # Add a margin depending on the side
    if label == "top":
        mid_y += 4
    elif label == "right":
        mid_x -= 4
    elif label == "bottom":
        mid_y -= 4
    elif label == "left":
        mid_x += 4

    return mid_x, mid_y


def text_to_tensor(text: str) -> Tensor:
    """
    This method takes a string in format "n,m,l" with n, m, l
    are integers and converts it into a Tensor with the
    given dimensions.

    Parameters
    ----------
    text : str
        Input string to convert.
    Returns
    ----------
    Tensor
        The converted tensor.

    """

    dims = ()
    num = 0

    for c in text:
        if '0' <= c <= '9':
            num = (num * 10) + int(c)
        else:
            dims += (num,)
            num = 0
    dims += (num,)

    return Tensor(shape=dims, buffer=np.random.normal(size=dims))


def text_to_tensor_set(text: str) -> tuple:
    """
    This method takes a string in format "(n,m,l), (n,m,l), ..."
    where n, m, l are integers and converts it into a tuple containing
    the set of Tensors with the given dimensions.

    Parameters
    ----------
    text: str
        Input string to convert.
    Returns
    ----------
    tuple
        The converted tensor set.

    """

    tensors = tuple()
    temp = tuple()

    for token in text.split(", "):
        num = int(token.replace("(", "").replace(")", ""))
        temp += (num,)
        if ")" in token:
            tensors += (Tensor(shape=temp, buffer=np.random.normal(size=temp)),)

    return tensors