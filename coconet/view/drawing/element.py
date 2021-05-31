import math
import sys

import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QLineF, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsPolygonItem, QLabel, \
    QVBoxLayout, QWidget, QGridLayout, QGraphicsProxyWidget

import coconet.view.styles as style
import coconet.view.util.utility as u
from coconet.core.controller.pynevertemp.tensor import Tensor
# Set maximum length of labels in NodeBlock
from coconet.core.model.network import NetworkProperty, PolyhedralNetworkProperty

MAX_FLOAT_LABEL_LENGTH = 5


class Line(QGraphicsLineItem):
    """
    This class is a graphic line with an arrow which connects two blocks of the
    scene.

    Attributes
    ----------
    origin : QGraphicsRectItem
        Origin rect of the line.
    destination : QGraphicsRectItem
        Destination rect of the line.
    scene : QGraphicsScene
        Current drawing scene.
    brush : QBrush
        Brush to draw the arrow.
    pen : QPen
        Pen to draw the arrow.
    arrow_head : QGraphicsPolygonItem
        Final arrow of the line.
    arrow_size : int
        Size of the head of the arrow.
    dim_label : QGraphicsTextItem
        Text showing the dimensions of the edge.
    is_valid : bool
        Flag monitoring whether the connection is consistent.

    Methods
    ----------
    gen_endpoints(QRectF, QRectF)
        Returns the shortest connection between the two rects.
    draw_arrow()
        Draws the polygon for the arrow.
    set_valid(bool)
        Assign validity for this line.
    update_dims(tuple)
        Update the line dimensions.
    update(QRectF)
        Update the line position given the new rect position.
    change_origin(QRectF)
        The origin of the line changes.
    change_destination(QRectF)
        The destination of the line changes.
    remove_self()
        Delete this line.

    """

    def __init__(self, origin: QGraphicsRectItem, destination: QGraphicsRectItem, scene):
        super(Line, self).__init__()
        self.origin = origin
        self.destination = destination
        self.scene = scene

        # This flag confirms a legal connection
        self.is_valid = True

        # Get the four sides of the rects
        destination_lines = u.get_sides_of(self.destination.sceneBoundingRect())
        origin_lines = u.get_sides_of(self.origin.sceneBoundingRect())

        # Get the shortest edge between the two blocks
        self.setLine(self.gen_endpoints(origin_lines, destination_lines))

        self.brush = QBrush(QColor(style.GREY_0))
        self.pen = QPen(QColor(style.GREY_0))
        self.pen.setWidth(4)
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.setPen(self.pen)

        # Dimensions labels
        self.dim_label = QGraphicsTextItem()
        self.dim_label.setZValue(6)
        self.scene.addItem(self.dim_label)

        # Arrow head
        self.arrow_head = QGraphicsPolygonItem()
        self.arrow_head.setPen(self.pen)
        self.arrow_head.setBrush(self.brush)
        self.arrow_size = 15.0

        self.draw_arrow()

    @staticmethod
    def gen_endpoints(origin_sides: dict, destination_sides: dict) -> QLineF:
        """
        This method finds the shortest path between two rectangles.

        Parameters
        ----------
        origin_sides : dict
            The dictionary {side_label: side_size} of the starting rect.
        destination_sides : dict
            The dictionary {side_label: side_size} of the ending rect.

        Returns
        ----------
        QLineF
            The shortest line.

        """

        # Init the line with the maximum possible value
        shortest_line = QLineF(-sys.maxsize / 2, -sys.maxsize / 2,
                               sys.maxsize / 2, sys.maxsize / 2)
        for o_side, origin_side in origin_sides.items():
            o_mid_x, o_mid_y = u.get_midpoint(o_side, origin_side)

            for d_side, destination_side in destination_sides.items():
                d_mid_x, d_mid_y = u.get_midpoint(d_side, destination_side)

                # Update line
                line = QLineF(o_mid_x, o_mid_y, d_mid_x, d_mid_y)
                if line.length() < shortest_line.length():
                    shortest_line = line

        return shortest_line

    def draw_arrow(self) -> None:
        """
        This method draws an arrow at the end of the line.

        """

        polygon_arrow_head = QPolygonF()

        # Compute the arrow angle
        angle = math.acos(self.line().dx() / self.line().length())
        angle = ((math.pi * 2) - angle)

        # Compute the direction where the arrow points (1 up, -1 down)
        arrow_direction = 1
        if math.asin(self.line().dy() / self.line().length()) < 0:
            arrow_direction = -1

        # First point of the arrow tail
        arrow_p1 = self.line().p2() - arrow_direction * QPointF(
            arrow_direction * math.sin(angle + math.pi / 2.5) * self.arrow_size,
            math.cos(angle + math.pi / 2.5) * self.arrow_size)
        # Second point of the arrow tail
        arrow_p2 = self.line().p2() - arrow_direction * QPointF(
            arrow_direction * math.sin(angle + math.pi - math.pi / 2.5) * self.arrow_size,
            math.cos(angle + math.pi - math.pi / 2.5) * self.arrow_size)

        # Third point is the line end
        polygon_arrow_head.append(self.line().p2())
        polygon_arrow_head.append(arrow_p2)
        polygon_arrow_head.append(arrow_p1)

        # Add the arrow to the scene
        self.arrow_head.setZValue(1)
        self.arrow_head.setParentItem(self)
        self.arrow_head.setPolygon(polygon_arrow_head)

    def set_valid(self, valid: bool) -> None:
        """
        This method changes the arrow style: if the connection 
        is not valid the arrow becomes red, otherwise it 
        remains grey with dimensions displayed. 
        
        Parameters
        ----------
        valid: bool
            New value for the legality flag.

        """

        if valid:
            self.is_valid = True
            self.pen.setColor(QColor(style.GREY_0))
            self.brush.setColor(QColor(style.GREY_0))
            self.dim_label.setVisible(False)
        else:
            self.is_valid = False
            self.pen.setColor(QColor(style.RED_2))
            self.brush.setColor(QColor(style.RED_2))

            if self.scene.is_dim_visible:
                self.dim_label.setVisible(True)

    def update_dims(self, dims: tuple) -> None:
        """
        This method updates the input & output dimensions.
        
        Parameters
        ----------
        dims: tuple
            The new dimensions to update.

        """

        self.dim_label.setHtml("<div style = 'background-color: " + style.RED_2 +
                               "; color: white; font-family: consolas;'>" +
                               str(dims) + "</div>")
        self.dim_label.setPos(self.line().center())

    def remove_self(self) -> None:
        """
        The line is removed from the scene along with origin and destination
        pointers.

        """

        self.scene.removeItem(self)
        self.scene.lines.remove(self)
        self.scene.removeItem(self.dim_label)
        self.origin = None
        self.destination = None


class NodeBlock(QtWidgets.QWidget):
    """
    This class is a widget for drawing the network nodes as
    blocks in the Canvas scene.

    Attributes
    ----------
    block_id : str
        String identifier of the block.
    node : NetworkNode
        Node associated to the block.
    block_data : dict
        Dictionary holding the node parameters, with string
        keys and numerical values.
    rect : QGraphicsRectItem
        Graphic rect associated to the block.
    proxy_control : QGraphicsProxyWidget
        Proxy holding the widget inside the scene.
    dim_labels : dict
        Dictionary with parameter dimension labels.
    is_head : bool
        Tells if the node is the head of the network.
    edits : dict
        Dictionary storing the parameters to update.

    Methods
    ----------
    init_layout(QVBoxLayout)
        Procedure to build the node layout.
    text_to_tuple(str)
        Procedure to convert a string in a tuple.
    update_labels()
        Procedure to update the labels after a change.
    resizeEvent(QtGui.QResizeEvent)
        Handles the resizing of the widget after updating
        the parameters dimensions.
    set_context_menu(dict)
        Procedure to build a context menu.
    contains(QPoint)
        Procedure to check if a point is inside the block.
    x()
        Top-left x coordinate.
    y()
        Top-left y coordinate.

    """

    edited = pyqtSignal()

    def __init__(self, block_id, node):
        super(NodeBlock, self).__init__()
        self.block_id = block_id
        self.in_dim = (1,)
        self.node = node
        self.block_data = dict()

        self.is_head = True
        self.rect = None
        self.proxy_control = None
        self.edits = None

        # Setting style and transparent background for the rounded corners
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(style.GRAPHIC_BLOCK_STYLE)

        # NodeBlock title label
        self.type_label = QLabel(node.name)
        self.type_label.setStyleSheet(style.BLOCK_TITLE_STYLE)

        self.dim_labels = dict()

        # Main vertical layout: it contains the label title and grid
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.type_label)
        self.setLayout(layout)
        if self.node.param:
            self.init_layout(layout)
        else:
            self.type_label.setStyleSheet(style.ZERO_PARS_BLOCK_TITLE)

    def init_layout(self, layout: QVBoxLayout) -> None:
        """
        This method sets up the the block layout with
        attributes and values.

        Parameters
        ----------
        layout: QVBoxLayout
            The layout to build.

        """

        # Create the grid for parameters
        grid = QWidget()
        grid_layout = QGridLayout()
        grid.setLayout(grid_layout)
        layout.addWidget(grid)

        # Iterate and display parameters, count rows
        par_labels = dict()
        count = 1
        for name, param in self.node.param.items():
            # Set the label
            par_labels[name] = QLabel(name)
            par_labels[name].setAlignment(Qt.AlignLeft)
            par_labels[name].setStyleSheet(style.PAR_BLOCK_STYLE)

            self.dim_labels[name] = QLabel()
            self.dim_labels[name].setAlignment(Qt.AlignCenter)
            self.dim_labels[name].setStyleSheet(style.DIM_BLOCK_STYLE)

            grid_layout.addWidget(par_labels[name], count, 0)
            grid_layout.addWidget(self.dim_labels[name], count, 1)
            count += 1

            # Init block data with default values
            if "default" in param.keys() and param["default"] == "None":
                self.block_data[name] = None
            elif param["type"] == "Tensor":
                if "shape" in param.keys():
                    shape = self.text_to_tuple(param["shape"])
                    self.block_data[name] = Tensor(shape=shape, buffer=np.random.normal(size=shape))
                else:
                    self.block_data[name] = Tensor(shape=(1, 1), buffer=np.random.normal(size=(1, 1)))
            elif param["type"] == "int":
                if "default" in param.keys():
                    if param["default"] == "rand":
                        self.block_data[name] = 1
                    else:
                        self.block_data[name] = int(param["default"])
                else:
                    self.block_data[name] = 1
            elif param["type"] == "list of ints":
                if "default" in param.keys():
                    self.block_data[name] = tuple(map(int, param["default"].split(', ')))
                else:
                    self.block_data[name] = (1, 1)
            elif param["type"] == "float":
                if "default" in param.keys():
                    if param["default"] == "rand":
                        self.block_data[name] = 1
                    else:
                        self.block_data[name] = float(param["default"])
                else:
                    self.block_data[name] = 0.1
            elif param["type"] == "boolean":
                if "default" in param.keys():
                    self.block_data[name] = bool(param["default"])
                else:
                    self.block_data[name] = False

        self.update_labels()
        self.edited.connect(lambda: self.update_labels())

    def text_to_tuple(self, text: str) -> tuple:
        """
        This method creates a tuple from a string, provided
        the string represents a tuple of dimensions.

        Parameters
        ----------
        text: str
            The dimension description string.

        Returns
        ----------
        tuple
            The tuple containing the dimensions.

        """

        str_shape = tuple(map(str, text.split(', ')))
        shape = tuple()
        for dim in str_shape:
            if isinstance(self.block_data[dim], tuple):
                shape += self.block_data[dim]
            else:
                shape += (self.block_data[dim],)
        return shape

    def update_labels(self) -> None:
        """
        This method updates the labels displaying the
        parameters dimension.

        """

        for name, param in self.node.param.items():
            value = self.block_data[name]

            # If the parameter is a Tensor write Shape
            if isinstance(value, Tensor) or isinstance(value, np.ndarray):  # TODO Necessary?
                self.dim_labels[name].setText("<" + 'x'.join(map(str, value.shape)) + ">")

            # If the floating-point digits are too many truncate it
            elif isinstance(value, float):
                if len(str(value)) <= MAX_FLOAT_LABEL_LENGTH + 3:  # The "+ 3" is used to take into account the "..."
                    self.dim_labels[name].setText("<" + str(value) + ">")
                else:
                    string = u.truncate(value, MAX_FLOAT_LABEL_LENGTH)
                    self.dim_labels[name].setText("<" + string + "...>")

            elif isinstance(value, int) or isinstance(value, bool):
                self.dim_labels[name].setText("<" + str(value) + ">")

            elif isinstance(value, tuple):
                self.dim_labels[name].setText("<" + 'x'.join(map(str, value)) + ">")

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        """
        This method resizes the selection rectangle when a
        QResizeEvent happens.

        Parameters
        ----------
        a0: event

        """

        if self.proxy_control is not None:
            # The new dimensions of the rect are given by the new dimensions
            # of the proxy widget
            x = self.proxy_control.geometry().x() + 10
            y = self.proxy_control.geometry().y() + 10
            width = self.proxy_control.geometry().width() - 20
            height = self.proxy_control.geometry().height() - 20

            self.rect.setRect(QRectF(x, y, width, height))

    def set_context_menu(self, actions) -> None:
        """
        This method builds a context menu with the given actions.

        Parameters
        ----------
        actions: dict
            The menu actions to display.

        """

        # Context Menu
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        for action in actions.values():
            self.addAction(action)

    def x(self) -> float:
        """
        Returns the x coordinate top-left corner of the rect

        Returns
        ----------
        float
            The absolute coordinate

        """

        return self.rect.x()

    def y(self) -> float:
        """
        Returns the y coordinate top-left corner of the rect

        Returns
        ----------
        float
            The absolute coordinate

        """

        return self.rect.y()


class PropertyBlock(QWidget):
    """
    This class is a widget for drawing network properties
    as blocks in the canvas scene.

    Attributes
    ----------
    property: NetworkProperty
        The property element associated to the block.

    """

    def __init__(self, property: NetworkProperty):
        super(PropertyBlock, self).__init__()
        self.property = property


class PolyhedralPropertyBlock(PropertyBlock):
    """
    This class represents the widget associated to a
    polyhedral property in NeVer.

    Attributes
    ----------
    property: PolyhedralNetworkProperty
        The concrete property element for this block.

    """

    def __init__(self, property: PolyhedralNetworkProperty):
        super().__init__(property)
