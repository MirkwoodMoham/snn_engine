from typing import Optional

from PyQt6 import QtCore

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QMainWindow
)

from vispy.app import Application
from vispy.scene import SceneCanvas
from .engine_scene_canvas import (
    EngineSceneCanvas,
    CanvasConfig,
    LocationGroupInfoCanvas,
    ScatterPlotSceneCanvas,
    VoltagePlotSceneCanvas)

from .gui import UI


class BaseWindow(QMainWindow):

    def __init__(self, name: str, parent=None):

        super().__init__(parent)

        self.setWindowTitle(name)
        self.setObjectName(name)
        self.resize(1200, 800)
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
                 app: Optional[Application],
                 plotting_config,
                 keys=None
                 ):
        super().__init__(name)

        for attr in ['ui', 'scene_3d']:
            if hasattr(self, attr):
                raise AttributeError(f'\'{attr}\' ')

        self.scene_3d = EngineSceneCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)

        self.ui = UI(self)

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.ui.ui_left.frame)
        splitter.addWidget(self.frame_canvas(self.scene_3d))

        hbox = QHBoxLayout(self.centralWidget())
        hbox.addWidget(splitter)

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 3)

    def set_keys(self, keys):
        self.scene_3d.set_keys(keys)


class NeuronPlotWindow(BaseWindow):

    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 plotting_config,
                 keys=None,
                 parent=None,
                 ):
        super().__init__(name=name, parent=parent)

        self.voltage_plot_sc = VoltagePlotSceneCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)

        self.scatter_plot_sc = ScatterPlotSceneCanvas(
            conf=CanvasConfig(keys=keys), app=app, plotting_config=plotting_config)

        self.frame_left = QFrame(self.centralWidget())

        self.splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.frame_left)

        # keep order

        hbox = QHBoxLayout(self.centralWidget())
        hbox.addWidget(self.splitter)
        self.splitter.addWidget(self.frame_canvas(self.voltage_plot_sc))
        self.splitter.addWidget(self.frame_canvas(self.scatter_plot_sc))


class LocationGroupInfoWindow(BaseWindow):

    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 keys=None,
                 parent=None
                 ):

        super().__init__(name=name, parent=parent)

        self.scene_3d = LocationGroupInfoCanvas(conf=CanvasConfig(keys=keys), app=app)

        self.frame_left = QFrame(self.centralWidget())

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.frame_left)
        splitter.addWidget(self.frame_canvas(self.scene_3d))

        hbox = QHBoxLayout(self.centralWidget())
        hbox.addWidget(splitter)
