from typing import Optional

from PyQt6 import QtCore

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QStatusBar
)

from vispy.app import Application
from vispy.scene import SceneCanvas
from .engine_scene_canvas import (
    EngineSceneCanvas,
    CanvasConfig,
    LocationGroupInfoCanvas,
    ScatterPlotSceneCanvas,
    VoltagePlotSceneCanvas)

from .ui_panels import MainUILeft, MenuBar, ButtonMenuActions, GroupInfoPanel
from .gui_element import RenderedObjectSliders
from network import PlottingConfig


class BaseWindow(QMainWindow):

    def __init__(self, name: str, parent=None):

        super().__init__(parent)

        self.setWindowTitle(name)
        self.setObjectName(name)
        self.resize(1600, 900)
        self.setCentralWidget(QWidget(self))

    def frame_canvas(self, canvas: SceneCanvas):
        frame = QFrame(self.centralWidget())
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Raised)
        frame_layout = QVBoxLayout(frame)
        frame_layout.addWidget(canvas.native)
        return frame


class EngineWindow(BaseWindow):

    def __init__(self,
                 name: str,
                 app, # : Optional[Application],
                 plotting_config: PlottingConfig,
                 keys=None
                 ):
        super().__init__(name)

        self.app = app

        for attr in ['ui', 'scene_3d']:
            if hasattr(self, attr):
                raise AttributeError(f'\'{attr}\' ')

        self.ui_elements = ButtonMenuActions(self)
        self.menubar = MenuBar(self)

        self.ui_panel_left = MainUILeft(self)

        self.setMenuBar(self.menubar)
        self.setStatusBar(QStatusBar(self))
        self.scene_3d = EngineSceneCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)
        if plotting_config.group_info_view_mode.split is True:
            self.group_info_scene = LocationGroupInfoCanvas(
                conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)
            self.scene_3d.network_view.camera.link(
                self.group_info_scene.view.camera)
            self.ui_right = GroupInfoPanel(self)

        self.splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.ui_panel_left)
        self.splitter.addWidget(self.frame_canvas(self.scene_3d))
        self.splitter.setStretchFactor(0, 6)
        self.splitter.setStretchFactor(1, 3)

        if plotting_config.group_info_view_mode.scene is True:
            self.ui_right = GroupInfoPanel(self)
            self.splitter.addWidget(self.ui_right)
            self.splitter.setStretchFactor(2, 30)

        hbox = QHBoxLayout(self.centralWidget())
        hbox.addWidget(self.splitter)

    def set_keys(self, keys):
        self.scene_3d.set_keys(keys)

    def add_group_info_scene_to_splitter(self, plotting_config):
        if plotting_config.group_info_view_mode.split is True:
            self.splitter.addWidget(self.frame_canvas(self.group_info_scene))
            self.splitter.setStretchFactor(2, 2)
            self.splitter.addWidget(self.ui_right)
            self.splitter.setStretchFactor(3, 6)


class NeuronPlotWindow(BaseWindow):

    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 plotting_config: PlottingConfig,
                 keys=None,
                 parent=None,
                 ):
        super().__init__(name=name, parent=parent)
        self.resize(1200, 800)
        self.voltage_plot_sc = VoltagePlotSceneCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)

        self.scatter_plot_sc = ScatterPlotSceneCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)

        self.frame_left = QFrame(self.centralWidget())

        self.splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.frame_left)
        self.splitter.addWidget(self.frame_canvas(self.voltage_plot_sc))
        self.splitter.addWidget(self.frame_canvas(self.scatter_plot_sc))

        # keep order

        hbox = QHBoxLayout(self.centralWidget())
        hbox.addWidget(self.splitter)


class LocationGroupInfoWindow(BaseWindow):

    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 plotting_config: PlottingConfig,
                 keys=None,
                 parent: EngineWindow = None
                 ):

        super().__init__(name=name, parent=parent)

        self.scene_3d = LocationGroupInfoCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)

        self.scene_3d.view.camera.link(parent.scene_3d.network_view.camera)

        self.ui_panel_left = GroupInfoPanel(self)

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.ui_panel_left)
        splitter.addWidget(self.frame_canvas(self.scene_3d))
        splitter.setStretchFactor(1, 3)

        hbox = QHBoxLayout(self.centralWidget())
        hbox.addWidget(splitter)
