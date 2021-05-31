from typing import Optional

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QStatusBar, QAction, QLabel, QGraphicsRectItem

import coconet.view.styles as style
from coconet.core.controller.project import Project
from coconet.view.drawing.element import Line, NodeBlock
from coconet.view.drawing.scene import DrawingMode, Canvas
from coconet.view.widget.dialog.dialogs import ConfirmDialog, MessageDialog, MessageType, HelpDialog
from coconet.view.widget.toolbar import BlocksToolbar, NodeButton
from coconet.view.widget.toolbar import ParamToolbar


class MainWindow(QtWidgets.QMainWindow):
    """
    This class is the main window of the program, containing all the graphics
    objects such as the toolbar, the state bar, the menu and the canvas scene.

    Attributes
    ----------
    SYSNAME : str
        The application name displayed in the window.
    nav_menu_bar : QMenuBar
        Menu bar of the window, containing different menus.
    status_bar : QStatusBar
        Status bar of the window.
    toolbar : BlocksToolbar
        Toolbar appearing on the left of the window, showing several buttons to
        add elements to the canvas.
    parameters : ParamToolbar
        Fixed toolbar on the right of the window, displaying details about a
        certain block.
    canvas : Canvas
        Central view of the window, containing a blank space in which the
        blocks appear.

    Methods
    ----------
    connect_events()
        Connects to all signals of the elements.
    init_menu_bar()
        Sets all menus of the menu bar and their actions.
    update_status()
        Changes the status bar displaying on it the canvas mode and the
        selected items.
    change_draw_mode(newmode)
        Changes the drawing mode of the canvas.
    create_from(NodeButton)
        Draws in the canvas the block corresponding to the button pressed.
    reset()
        Clears both graphical and logical network.
    open()
        Procedure to open an existing network.
    save(bool)
        Saves the current network in a new file or in the opened one.

    """

    def __init__(self):
        super(MainWindow, self).__init__()

        # Init window appearance
        self.SYSNAME = "CoCoNet"
        self.setWindowTitle(self.SYSNAME)
        self.setWindowIcon(QtGui.QIcon('coconet/res/icons/CCNN_logo.svg'))
        self.setStyleSheet("background-color: " + style.GREY_1)

        # Navigation menu
        self.nav_menu_bar = self.menuBar()
        self.init_nav_menu_bar()

        # Blocks toolbar
        self.toolbar = BlocksToolbar('coconet/res/json/blocks.json')

        # Parameters toolbar
        self.parameters = ParamToolbar()

        # Project in use
        self.project = Project()
        self.project.opened_net.connect(lambda: self.canvas.draw_network(self.project.NN))

        # Drawing Canvas
        self.canvas = Canvas(self.project, self.toolbar.blocks)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(style.STATUS_BAR_STYLE)
        self.status_bar_mode_label = QLabel()
        self.status_bar_mode_label.setStyleSheet(style.STATUS_BAR_WIDGET_STYLE)
        self.status_bar_selections_label = QLabel()
        self.status_bar_selections_label.setStyleSheet(style.STATUS_BAR_WIDGET_STYLE)
        self.status_bar.addPermanentWidget(self.status_bar_selections_label)
        self.status_bar.addPermanentWidget(self.status_bar_mode_label)

        # And adding them to the window
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameters, Qt.Vertical)
        self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.toolbar)
        self.setCentralWidget(self.canvas.view)

        self.connect_events()

    def connect_events(self):
        """
        Associate the various events coming from button signals and other
        graphical objects to the correct actions.

        """

        # NodeBlock buttons
        for b in self.toolbar.b_buttons.values():
            b.clicked.connect(self.create_from(b))

        # Draw line button
        self.toolbar.f_buttons["draw_line"].clicked \
            .connect(lambda: self.change_draw_mode(DrawingMode.DRAW_LINE))

        # Insert block button
        self.toolbar.f_buttons["insert_block"].clicked \
            .connect(lambda: self.change_draw_mode(DrawingMode.INSERT_BLOCK))

        # Parameters box appearing
        self.canvas.param_requested \
            .connect(lambda: self.parameters.display(self.canvas.block_to_show))

        # State bar updating
        self.canvas.scene.has_changed_mode \
            .connect(lambda: self.update_status())
        self.canvas.scene.selectionChanged \
            .connect(lambda: self.update_status())

    def init_nav_menu_bar(self):
        """
        This method creates the navigation bar by adding the menus and
        corresponding actions.

        """

        self.nav_menu_bar.setStyleSheet(style.MENU_BAR_STYLE)
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        # Create top-level menu
        menu_file = self.nav_menu_bar.addMenu("File")
        menu_file.setStyleSheet(style.MENU_BAR_STYLE)
        menu_edit = self.nav_menu_bar.addMenu("Edit")
        menu_edit.setStyleSheet(style.MENU_BAR_STYLE)
        menu_view = self.nav_menu_bar.addMenu("View")
        menu_view.setStyleSheet(style.MENU_BAR_STYLE)

        # File actions
        new_action = QAction("New...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(lambda: self.reset())
        open_action = QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(lambda: self.open())
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(lambda: self.save(False))
        save_as_action = QAction("Save as...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(lambda: self.save())

        # Edit actions
        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(lambda: self.canvas.copy_selected())
        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(lambda: self.canvas.paste_selected())
        cut_action = QAction("Cut", self)
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(lambda: self.canvas.cut_selected())
        del_action = QAction("Delete", self)
        del_action.setShortcut("DEL")
        del_action.triggered.connect(lambda: self.canvas.delete_selected())
        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+Shift+C")
        clear_action.triggered.connect(lambda: self.clear())
        draw_line_action = QAction("Draw connection", self)
        draw_line_action.setShortcut("Ctrl+D")
        draw_line_action.triggered.connect(lambda: self.change_draw_mode(DrawingMode.DRAW_LINE))
        insert_block_action = QAction("Insert block in connection", self)
        insert_block_action.setShortcut("Ctrl+I")
        insert_block_action.triggered.connect(lambda: self.change_draw_mode(DrawingMode.INSERT_BLOCK))
        prop_action = QAction("Edit", self)
        prop_action.setShortcut("Ctrl+E")
        prop_action.triggered.connect(lambda: self.canvas.scene.edit_block(self.edit_action_validation()))

        # View actions
        z_in_action = QAction("Zoom in", self)
        z_in_action.setShortcut("Ctrl+")
        z_in_action.triggered.connect(lambda: self.canvas.zoom_in())
        z_out_action = QAction("Zoom out", self)
        z_out_action.setShortcut("Ctrl-")
        z_out_action.triggered.connect(lambda: self.canvas.zoom_out())
        dims_action = QAction("Dimensions", self)
        dims_action.setCheckable(True)
        dims_action.setChecked(True)
        dims_action.toggled.connect(lambda: self.canvas.scene.switch_dim_visibility())
        toolbar_action = QAction("Tools", self)
        toolbar_action.setCheckable(True)
        toolbar_action.setChecked(True)
        toolbar_action.toggled.connect(lambda: self.toolbar.change_tools_mode())
        blocks_action = QAction("Blocks library", self)
        blocks_action.setCheckable(True)
        blocks_action.setChecked(True)
        blocks_action.toggled.connect(lambda: self.toolbar.change_blocks_mode())
        details_action = QAction("Parameters", self)
        details_action.setShortcut("Ctrl+P")
        details_action.triggered.connect(lambda: self.canvas.show_parameters(self.parameters_action_validation()))

        # Build File menu
        menu_file.addAction(new_action)
        menu_file.addAction(open_action)
        menu_file.addSeparator()
        menu_file.addAction(save_action)
        menu_file.addAction(save_as_action)

        # Build Edit menu
        menu_edit.addSeparator()
        menu_edit.addAction(copy_action)
        menu_edit.addAction(paste_action)
        menu_edit.addAction(cut_action)
        menu_edit.addAction(del_action)
        menu_edit.addSeparator()
        menu_edit.addAction(clear_action)
        menu_edit.addSeparator()
        menu_edit.addAction(draw_line_action)
        menu_edit.addAction(insert_block_action)
        menu_edit.addSeparator()
        menu_edit.addAction(prop_action)

        # Build View menu
        menu_view.addAction(z_in_action)
        menu_view.addAction(z_out_action)
        menu_view.addSeparator()
        toolbars_menu = menu_view.addMenu("Show")
        toolbars_menu.addAction(toolbar_action)
        toolbars_menu.addAction(blocks_action)
        toolbars_menu.addAction(dims_action)
        menu_view.addSeparator()
        menu_view.addAction(details_action)

        # Help menu
        self.nav_menu_bar.addAction("Help", self.show_help)

    def create_from(self, button: NodeButton):
        """
        This method draws on the canvas the block corresponding to the pressed
        BlockButton.

        Parameters
        ----------
        button : NodeButton
            The pressed button.

        """

        def pressed():
            self.canvas.draw_node(button.node_type)

        return pressed

    def update_status(self):
        """
        This method updates the widget in the status bar, displaying the
        items selected and the current drawing mode.

        """

        # Show the canvas drawing mode
        if self.canvas.scene.mode is DrawingMode.DRAW_LINE:
            self.status_bar_mode_label.setText("Line drawing")
        elif self.canvas.scene.mode is DrawingMode.INSERT_BLOCK:
            self.status_bar_mode_label.setText("NodeBlock insertion")
        else:
            self.status_bar_mode_label.setText("")

        # Show the selected items, if any
        if not self.canvas.scene.selectedItems():
            self.status_bar_selections_label.setText("")
        else:
            selections = ""
            semicolons = ["; " for _ in range(len(self.canvas.scene.selectedItems()))]
            semicolons[-1] = ""  # No semicolon for the last element in the selections list

            for counter, item in enumerate(self.canvas.scene.selectedItems()):
                if type(item) is QGraphicsRectItem:
                    # If the item is a rect, prev_node_id is written
                    selections += self.canvas.scene.blocks[item].block_id
                    selections += semicolons[counter]
                elif type(item) is Line:
                    # If the item is a line, origin and destination ids are written
                    origin = self.canvas.scene.blocks[item.origin].block_id
                    destination = self.canvas.scene.blocks[item.destination].block_id
                    selections += origin + "->" + destination
                    selections += semicolons[counter]

            self.status_bar_selections_label.setText(selections)

    def change_draw_mode(self, newmode: DrawingMode = None):
        """
        This method changes the drawing mode of the canvas when the user
        clicks on the corresponding button. The mode changes depending
        on the previous one.

        Parameters
        ----------
        newmode : DrawingMode, optional
            Specifies the new DrawingMode to use. (Default: None)

        """

        curmode = self.canvas.scene.mode
        # If the mode is specified, the given CanvasMode is set
        if newmode == DrawingMode.DRAW_LINE:
            if curmode == DrawingMode.DRAW_LINE:
                self.canvas.scene.set_mode(DrawingMode.IDLE)
            else:
                self.canvas.scene.set_mode(DrawingMode.DRAW_LINE)
        elif newmode == DrawingMode.INSERT_BLOCK:
            if curmode == DrawingMode.INSERT_BLOCK:
                self.canvas.scene.set_mode(DrawingMode.IDLE)
            else:
                self.canvas.scene.set_mode(DrawingMode.INSERT_BLOCK)
        else:
            # If the mode is not specified, the canvas mode is set to IDLE
            if curmode == DrawingMode.DRAW_LINE or curmode == DrawingMode.INSERT_BLOCK:
                self.canvas.scene.set_mode(DrawingMode.IDLE)

    def clear(self):
        """
        Utility for deleting the content of the window. Before taking effect,
        it prompts the user to confirm.

        """

        if self.canvas.num_blocks > 0:
            alert_dialog = ConfirmDialog("Clear workspace",
                                         "The network will be erased and your work will be lost.\n"
                                         "Do you wish to continue?")
            alert_dialog.exec()
            if alert_dialog.confirm:
                self.canvas.clear_scene()
                self.canvas.scene.has_changed_mode.connect(lambda: self.update_status())
                self.canvas.scene.selectionChanged.connect(lambda: self.update_status())
                self.update_status()
        else:
            self.canvas.clear_scene()
            self.canvas.scene.has_changed_mode.connect(lambda: self.update_status())
            self.canvas.scene.selectionChanged.connect(lambda: self.update_status())
            self.update_status()

    def reset(self):
        """
        This method clears the scene and the network, stops to work on the file
        and restarts from scratch.

        """

        self.clear()
        self.project.file_name = ("", "")
        self.project.file_path = ""
        self.setWindowTitle(self.SYSNAME)

    def open(self):
        """
        This method handles the opening of a file.

        """

        if self.canvas.renderer.disconnected_network:
            # If there is already a network in the canvas, it is asked to the
            # user if continuing with opening.
            confirm_dialog = ConfirmDialog("Open network",
                                           "A new network will be opened "
                                           "cancelling the current nodes.\n"
                                           "Do you wish to continue?")
            confirm_dialog.exec()
            # If the user clicks on "yes", the canvas is cleaned, a net is
            # opened and the window title is updated.
            if confirm_dialog is not None:
                if confirm_dialog.confirm:
                    # The canvas is cleaned
                    self.canvas.clear_scene()
                    self.canvas.scene.has_changed_mode.connect(lambda: self.update_status())
                    self.canvas.scene.selectionChanged.connect(lambda: self.update_status())
                    self.update_status()
                    # A file is opened
                    self.project.open()
                    self.setWindowTitle(self.SYSNAME + " - " + self.project.NN.identifier)
        else:
            # If the canvas was already empty, the opening function is directly
            # called
            self.project.open()
            self.setWindowTitle(self.SYSNAME + " - " + self.project.NN.identifier)

    def save(self, _as: bool = True):
        """
        This method saves the current network if the format is correct

        Parameters
        ----------
        _as : bool, optional
            This attribute distinguishes between "save" and "save as".
            If _as is True the network will be saved in a new file, while
            if _as is False the network will overwrite the current one.
            (Default: True)

        """

        if len(self.canvas.renderer.project.NN.nodes) == 0 or \
                len(self.canvas.renderer.project.NN.edges) == 0:
            # Limit case: one disconnected node -> new network with one node
            if len(self.canvas.renderer.disconnected_network) == 1:
                for node in self.canvas.renderer.disconnected_network:
                    try:
                        self.canvas.renderer.add_node_to_nn(node)
                        self.project.save(_as)
                    except Exception as e:
                        error_dialog = MessageDialog(str(e), MessageType.ERROR)
                        error_dialog.exec()

            # More than one disconnected nodes cannot be saved
            elif len(self.canvas.renderer.disconnected_network) > 1:
                not_sequential_dialog = MessageDialog("The network is not sequential, and "
                                                      "cannot be saved.",
                                                      MessageType.ERROR)
                not_sequential_dialog.exec()
            else:
                # Network is empty
                message = MessageDialog("The network is empty!", MessageType.MESSAGE)
                message.exec()

        elif self.canvas.renderer.is_nn_sequential():
            # If there are logical nodes, the network is sequential
            every_node_connected = True
            # every node has to be in the nodes dictionary
            for node in self.canvas.renderer.disconnected_network:
                if node not in self.project.NN.nodes:
                    every_node_connected = False
                    break

            if every_node_connected:
                self.project.save(_as)
            else:
                # If there are disconnected nodes, a message is displayed to the
                # user to choose if saving only the connected network
                confirm_dialog = ConfirmDialog("Save network",
                                               "All the nodes outside the "
                                               "sequential network will lost.\n"
                                               "Do you wish to continue?")
                confirm_dialog.exec()
                if confirm_dialog.confirm:
                    self.project.save(_as)
        else:
            # If the network is not sequential, it cannot be saved.
            not_sequential_dialog = MessageDialog("The network is not sequential and "
                                                  "cannot be saved.",
                                                  MessageType.ERROR)
            not_sequential_dialog.exec()

    def edit_action_validation(self) -> Optional[NodeBlock]:
        """
        This method performs a check on the object on which the edit
        action is called, in order to prevent unwanted operations.

        Returns
        ----------
        NodeBlock
            The graphic wrapper of the NetworkNode selected, if present.

        """

        if self.canvas.scene.selectedItems():
            if type(self.canvas.scene.selectedItems()[0]) is QGraphicsRectItem:
                # Return block graphic object
                return self.canvas.scene.blocks[self.canvas.scene.selectedItems()[0]]
            elif type(self.canvas.scene.selectedItems()[0]) is Line:
                msg_dialog = MessageDialog("Can't edit lines, please select a block instead.",
                                           MessageType.ERROR)
                msg_dialog.show()
        else:
            err_dialog = MessageDialog("No block selected.", MessageType.MESSAGE)
            err_dialog.show()

    def parameters_action_validation(self) -> Optional[NodeBlock]:
        """
        This method performs a check on the object on which the parameters
        action is called, in order to prevent unwanted operations.

        Returns
        ----------
        NodeBlock
            The graphic wrapper of the NetworkNode selected, if present.

        """

        if self.canvas.scene.selectedItems():
            if type(self.canvas.scene.selectedItems()[0]) is QGraphicsRectItem:
                # Return block graphic object
                return self.canvas.scene.blocks[self.canvas.scene.selectedItems()[0]]
            elif type(self.canvas.scene.selectedItems()[0]) is Line:
                msg_dialog = MessageDialog("No parameters available for connections.", MessageType.ERROR)
                msg_dialog.show()
        else:
            err_dialog = MessageDialog("No block selected.", MessageType.MESSAGE)
            err_dialog.show()

    @staticmethod
    def show_help():
        help_dialog = HelpDialog()
        help_dialog.exec()
