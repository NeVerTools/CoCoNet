"""
Module graphics_block.py

This module contains the graphics elements of Block objects for representing the layers,
the IO and the properties

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

from functools import partial

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor, QFont, QBrush, QPainterPath
from PyQt6.QtWidgets import QGraphicsItem, QWidget, QGraphicsProxyWidget, QGraphicsTextItem, QVBoxLayout, QGridLayout, \
    QHBoxLayout

import coconet.resources.styling.dimension as dim
import coconet.resources.styling.palette as palette
from coconet import get_classname, RES_DIR
from coconet.resources.styling.custom import CustomTextBox, CustomLabel, CustomComboBox, CustomButton
from coconet.utils.repr import ArithmeticValidator
from coconet.view.ui.dialog import ConfirmDialog


class GraphicsBlock(QGraphicsItem):
    def __init__(self, block: 'Block', parent=None):
        super().__init__(parent)
        # Reference to the block
        self.block_ref = block

        # Hover flag
        self.hover = False

        # Content widget
        self.content = None
        self.width = dim.BLOCK_BASE_WIDTH
        self.height = dim.BLOCK_BASE_HEIGHT

        # Init graphics content
        self.title_item = QGraphicsTextItem(self)
        self.graphics_content = QGraphicsProxyWidget(self)
        self.init_title()
        self.init_graphics_content()

        # Style parameters
        self.color_scheme = []
        self.init_colors()
        self._pen_default = QPen(QColor(self.color_scheme[0]))
        self._pen_default.setWidth(2)
        self._pen_hovered = QPen(QColor(self.color_scheme[1]))
        self._pen_hovered.setWidth(2)
        self._pen_selected = QPen(QColor(self.color_scheme[2]))
        self._pen_selected.setWidth(3)
        self._pen_selected.setStyle(Qt.PenStyle.DotLine)
        self._brush_title = QBrush(QColor(self.color_scheme[3]))
        self._brush_background = QBrush(QColor(self.color_scheme[4]))

        self.init_flags()

    def init_title(self):
        """
        This method sets up the title widget

        """

        self.title_item.setDefaultTextColor(QColor(palette.WHITE))
        self.title_item.setFont(QFont('Arial', 10))
        self.title_item.setPos(dim.TITLE_PAD, 0)
        self.title_item.setPlainText(self.block_ref.title)
        self.title_item.setTextWidth(self.width - 2 * dim.TITLE_PAD)

    def init_colors(self):
        """
        This method sets up the color scheme of the block
        depending on the block type

        """

        if get_classname(self.block_ref) == 'LayerBlock':
            self.color_scheme = [palette.DARK_BLUE, palette.BLUE, palette.BLUE,
                                 palette.DARK_BLUE, palette.DARK_GREY]
        elif get_classname(self.block_ref) == 'PropertyBlock':
            self.color_scheme = [palette.DARK_ORANGE, palette.ORANGE, palette.ORANGE,
                                 palette.DARK_ORANGE, palette.DARK_GREY]
        elif get_classname(self.block_ref) == 'FunctionalBlock':
            self.color_scheme = [palette.GREY, palette.LIGHT_GREY, palette.LIGHT_GREY,
                                 palette.GREY, palette.DARK_GREY]

    def init_graphics_content(self):
        """
        This method sets up the graphics properties of the block
        depending on the content

        """

        if self.block_ref.has_parameters():
            self.width = dim.BLOCK_PARAM_WIDTH
        elif get_classname(self.block_ref) == 'PropertyBlock':
            self.width = dim.BLOCK_PROPERTY_WIDTH

        if self.content is not None:
            self.content.setGeometry(dim.EDGE_ROUNDNESS, dim.TITLE_HEIGHT + dim.EDGE_ROUNDNESS,
                                     self.width - 2 * dim.EDGE_ROUNDNESS, 0)
            self.graphics_content.setWidget(self.content)

            self.width = self.content.size().width() + 2 * dim.EDGE_ROUNDNESS
            self.height = self.content.size.height() + 2 * dim.EDGE_ROUNDNESS + dim.TITLE_HEIGHT

    def init_flags(self):
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)

    def open_dock_params(self):
        self.block_ref.scene_ref.editor_widget_ref.show_inspector(self.block_ref)

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        self.hover = False
        self.update()

    def mouseMoveEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        super().mouseMoveEvent(event)

        for block in self.block_ref.scene_ref.blocks.values():
            if block.graphics_block.isSelected():
                block.update_edges()

        self.block_ref.update_edges()

    def mouseDoubleClickEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        self.open_dock_params()

    def paint(self, painter: 'QPainter', option: 'QStyleOptionGraphicsItem', widget=None) -> None:
        """
        This method draws the graphicsBlock item. It is a rounded rectangle divided in 3 sections:

        Outline section: draw the contours of the block
        Title section: a darker rectangle in which lays the title
        Content section: container for the block parameters

        """

        # Title section
        path_title = QPainterPath()
        path_title.setFillRule(Qt.FillRule.WindingFill)
        path_title.addRoundedRect(0, 0, self.width, dim.TITLE_HEIGHT, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)

        # Remove bottom rounded corners for title box
        path_title.addRect(0, dim.TITLE_HEIGHT - dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)
        path_title.addRect(self.width - dim.EDGE_ROUNDNESS, dim.TITLE_HEIGHT - dim.EDGE_ROUNDNESS,
                           dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._brush_title)
        painter.drawPath(path_title.simplified())

        # Content
        path_content = QPainterPath()
        path_content.setFillRule(Qt.FillRule.WindingFill)
        path_content.addRoundedRect(0, dim.TITLE_HEIGHT, self.width,
                                    self.height - dim.TITLE_HEIGHT, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)

        # Remove top rounded corners for content box
        # left
        path_content.addRect(0, dim.TITLE_HEIGHT, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS / 2)
        # right
        path_content.addRect(self.width - dim.EDGE_ROUNDNESS, dim.TITLE_HEIGHT,
                             dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS / 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._brush_background)
        painter.drawPath(path_content.simplified())

        # Outline
        path_outline = QPainterPath()
        path_outline.addRoundedRect(0, 0, self.width, self.height, dim.EDGE_ROUNDNESS, dim.EDGE_ROUNDNESS)
        painter.setPen(self._pen_default if not self.isSelected() else self._pen_selected)

        if self.hover and not self.isSelected():
            painter.setPen(self._pen_hovered)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path_outline.simplified())

    def boundingRect(self) -> QtCore.QRectF:
        """
        Defines the Qt bounding rectangle

        Returns
        ----------
        QRectF
            The area in which the click triggers the item

        """

        return QtCore.QRectF(0, 0, self.width, self.height).normalized()


class BlockContentWidget(QWidget):
    """
    This class provides a QGridLayout structure for the display
    of the Block object parameters inside the GraphicsBlock

    Attributes
    ----------
    block_ref : Block
        Reference to block object

    params_ref : dict
        Reference to block parameters

    wdg_param_dict : dict
        Dictionary of concrete block parameters and widgets

    """

    def __init__(self, block: 'Block', build_dict: dict = None, parent=None):
        super().__init__(parent)

        # Block and parameters reference
        self.block_ref = block
        self.params_ref = self.block_ref.attr_dict['parameters']

        # Concrete parameters
        self.wdg_param_dict = dict()

        # Set style
        with open(RES_DIR + '/styling/qss/blocks.qss') as qss_file:
            self.setStyleSheet(qss_file.read())

        # Layouts
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)

        if build_dict is None:
            self.init_content()
        else:
            self.load_content(build_dict)

        self.init_buttons()

        # Build
        self.layout.addLayout(self.grid_layout)
        self.layout.addLayout(self.buttons_layout)
        self.setLayout(self.layout)

        self.init_validators()

    def init_content(self):
        """
        This method reads the parameters and creates the corresponding objects

        """

        row_grid_count = 0

        for param_name in self.params_ref:
            # Set label
            self.grid_layout.addWidget(CustomLabel(param_name + ':'), row_grid_count, 0)

            # Switch components
            def_val = ''
            if 'default' in self.params_ref[param_name]:
                def_val = self.params_ref[param_name]['default']

            type_val = self.params_ref[param_name]['type']
            param_obj = self.params_ref[param_name]['object']

            if param_obj == 'QLabel':
                label = CustomLabel(def_val)

                self.grid_layout.addWidget(label, row_grid_count, 1)
                self.wdg_param_dict[param_name] = [label, def_val, type_val]

            elif param_obj == 'QLineEdit':
                line_edit = CustomTextBox(def_val, context=get_classname(self.block_ref))

                if self.params_ref[param_name]['editable'] == 'False':
                    line_edit.setEnabled(False)

                self.grid_layout.addWidget(line_edit, row_grid_count, 1)
                self.wdg_param_dict[param_name] = [line_edit, def_val, type_val]

            elif param_obj == 'QComboBox':
                if get_classname(self.block_ref) == 'FunctionalBlock':
                    if not self.block_ref.has_input():
                        self.setObjectName('OutputBlockContent')

                    combo = CustomComboBox(context=get_classname(self.block_ref))
                    cb_fill = self.block_ref.scene_ref.editor_widget_ref.property_data.keys()
                    combo.addItems(cb_fill)

                    combo.view().setMinimumWidth(140)
                    combo.setCurrentText(def_val)
                else:
                    combo = CustomComboBox()
                    combo.addItems(['True', 'False'])
                    if def_val == '':
                        combo.setCurrentText('True')
                    else:
                        combo.setCurrentText(def_val)

                self.grid_layout.addWidget(combo, row_grid_count, 1)
                self.wdg_param_dict[param_name] = [combo, combo.currentText(), type_val]

            # Increment for next parameter
            row_grid_count += 1

    def load_content(self, build_dict: dict):
        pass

    def init_buttons(self):
        """
        This method adds to the bottom of the block buttons for saving the updates or to
        add some property

        """

        if get_classname(self.block_ref) == 'LayerBlock':
            if self.block_ref.has_parameters():
                # Create save and discard buttons
                discard = CustomButton('Restore defaults', context='LayerBlock')
                discard.clicked.connect(self.on_button_click)
                save = CustomButton('Save', primary=True, context='LayerBlock')
                discard.clicked.connect(self.on_button_click)

                self.buttons_layout.addWidget(discard)
                self.buttons_layout.addWidget(save)

        elif get_classname(self.block_ref) == 'FunctionalBlock':
            # Create property and update buttons
            add_prop = CustomButton('Add property', context='FunctionalBlock')
            add_prop.clicked.connect(self.add_property)
            update = CustomButton('Update', context='FunctionalBlock')
            update.clicked.connect(self.on_button_click)

            self.buttons_layout.addWidget(add_prop)
            self.buttons_layout.addWidget(update)

        elif get_classname(self.block_ref) == 'PropertyBlock':
            # Create edit button
            edit = CustomButton('Edit', context='PropertyBlock')
            edit.clicked.connect(lambda: self.block_ref.scene_ref.edit_property(self.block_ref))

            self.buttons_layout.addWidget(edit)

    def add_property(self):
        pass

    def init_validators(self):
        """
        This method sets the proper validator for the parameters widgets

        """

        for param_name, param_info in self.params_ref.items():
            qt_wdg = self.wdg_param_dict[param_name][0]

            # Type check
            if isinstance(qt_wdg, CustomTextBox):
                prev = qt_wdg.text()
                check_value = partial(self.check_value_proxy, param_name, prev)
                qt_wdg.editingFinished.connect(check_value)

                # Set proper validator
                validator = None
                param_type = param_info['type']
                if param_type == 'int':
                    validator = ArithmeticValidator.INT
                elif param_type == 'float':
                    validator = ArithmeticValidator.FLOAT
                elif param_type == 'Tensor' or param_type == 'list of ints':
                    validator = ArithmeticValidator.TENSOR
                qt_wdg.setValidator(validator)

    def on_button_click(self):
        """
        Handler for the button operations. It checks for properties before allowing modifications.

        """

        if self.block_ref.scene_ref.post_block is not None:
            dialog = ConfirmDialog('Confirmation required',
                                   'Editing the network will remove the output property.\nProceed?')
            dialog.exec()

            if dialog.confirm:
                self.block_ref.scene_ref.remove_out_prop()
        else:
            button_type = self.sender().text()

            if button_type == 'Update':  # Only for FunctionalBlocks
                if self.block_ref.title == 'Input' and self.block_ref.scene_ref.pre_block is not None:

                    dialog = ConfirmDialog('Confirmation required',
                                           'Editing the network will remove the input property.\nProceed?')
                    dialog.exec()

                    if dialog.confirm:
                        self.block_ref.scene_ref.remove_in_prop()
                        self.save_func_params()
                else:
                    self.save_func_params()

            elif button_type == 'Restore defaults':
                self.restore_default()
            elif button_type == 'Save':
                self.save_layer_params()

    def check_value_proxy(self, name, value):
        self.check_values(name, value)

    def check_values(self, name, value):
        pass

    def save_func_params(self):
        pass

    def restore_default(self):
        pass

    def save_layer_params(self):
        pass
