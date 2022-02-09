from dataclasses import asdict, dataclass
from typing import Optional, Union
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QIcon, QAction

from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget
)

from vispy.app import Application, Canvas
from vispy.color import Color
from vispy.gloo.context import GLContext
from vispy.scene import SceneCanvas, ViewBox, visuals
from vispy.scene.cameras import MagnifyCamera, Magnify1DCamera


def menubar(main_window):
    exit_act = QAction(QIcon('exit.png'), '&Exit', main_window)
    exit_act.setShortcut('Ctrl+Q')
    exit_act.setStatusTip('Exit application')

    exit_act.triggered.connect(QApplication.instance().quit)

    mb = QtWidgets.QMenuBar(main_window)
    mb.setGeometry(QtCore.QRect(0, 0, 440, 18))
    mb.setObjectName("menubar")
    file_menu = mb.addMenu('&File')
    file_menu.addAction(exit_act)
    return mb


# def statusbar(main_window):
# 	sb = QtWidgets.QStatusBar(main_window)
# 	sb.setObjectName("statusbar")
# 	return sb

# self.menubar.addAction(exit_act)

class UiButtons:

    def __init__(self, window):
        self.ok = QPushButton("OK")
        self.window = window

        self.ok.clicked.connect(self.button_clicked)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.button_clicked)

        self.toggle_outergrid = QPushButton('Show OuterGrid')
        self.toggle_outergrid.setCheckable(True)
        self.toggle_outergrid.clicked.connect(self.button_clicked)

    def button_clicked(self):
        sender = self.window.sender()
        msg = f'Clicked: {sender.text()}'
        self.window.statusBar().showMessage(msg)


class UiMainWindow(object):

    def __init__(self, main_window):
        self.window = main_window
        self.window.setMenuBar(menubar(self.window))

        self.central_widget = QWidget(self.window)
        # self.central_widget.setObjectName("central_widget")

        self.grid_layout = QGridLayout(self.central_widget)
        # self.grid_layout.setObjectName("grid_layout")

        # # self.left_up_widget = QWidget(self.window)
        # self.frame_voltage_plot = QFrame(self.central_widget)
        # self.left_layout = QVBoxLayout(self.frame_voltage_plot)
        # # self.left_layout.addWidget(self.frame_voltage_plot)
        # self.grid_layout.addWidget(self.frame_voltage_plot, 1, 0, 1, 1)

        self.frame_3d = QFrame(self.central_widget)
        self.frame_3d.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3d.setFrameShadow(QFrame.Shadow.Raised)
        self.central_layout = QVBoxLayout(self.frame_3d)
        self.grid_layout.addWidget(self.frame_3d, 0, 0, 1, 2)

        self.buttons = UiButtons(self.window)
        hbox = QHBoxLayout()
        # hbox.addStretch(1)
        hbox.addWidget(self.buttons.ok)
        hbox.addWidget(self.buttons.toggle_outergrid)
        hbox.addWidget(self.buttons.cancel)

        self.grid_layout.addLayout(hbox, 1, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter)

        self.horizontalSlider = QSlider(self.central_widget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")

        self.grid_layout.addWidget(self.horizontalSlider, 3, 0, 1, 2)

        self.window.setCentralWidget(self.central_widget)

        self.window.setStatusBar(QStatusBar(self.window))

        self.retranslate_ui(self.window)

    @staticmethod
    def retranslate_ui(main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("SNN Engine", "SNN Engine"))


@dataclass
class CanvasConfig:
    title: str = 'VisPy canvas'
    size: tuple = (1600, 1200)
    position: Optional[tuple] = None
    show: bool = False
    autoswap: bool = True

    create_native: bool = True
    vsync: bool = False
    resizable: bool = True
    decorate: bool = True
    fullscreen: bool = False
    config: Optional[dict] = None
    shared = Optional[Union[Canvas, GLContext]]
    keys: Optional[Union[str, dict]] = 'interactive'
    parent: Optional = None
    dpi: Optional[float] = None
    always_on_top: bool = False
    px_scale: int = 1
    bgcolor: Union[str, Color] = 'black'


class EngineSceneCanvas(SceneCanvas):

    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 network):
        conf = conf or CanvasConfig()
        super().__init__(**asdict(conf), app=app)
        self.unfreeze()
        # self._central_view = None
        grid = self.central_widget.add_grid()
        self.network_view = grid.add_view(row=0, col=1, col_span=4, row_span=2)
        self.network_view.camera = 'turntable'  # or try 'arcball'

        self.voltage_plot_view = grid.add_view(row=0, col=0, border_color=(1, 0, 0))
        grid1 = visuals.GridLines(parent=self.voltage_plot_view.scene)
        self.voltage_plot_view.camera = MagnifyCamera(mag=3, size_factor=0.3, radius_ratio=0.6, )
        self.voltage_plot_view.camera = 'panzoom'

        self.scatter_plot_view = grid.add_view(row=1, col=0, border_color=(1, 0, 0))
        # self.central_view = ViewBox()
        # add a colored 3D axis for orientation
        visuals.XYZAxis(parent=self.network_view.scene)
        self.freeze()

        if network is not None:
            self.add_scatter_plot(network)
            self.add_outer_box(network)
            self.add_selector_box(network)
            self.add_voltage_plot(network)

    @property
    def _window_id(self):
        # noinspection PyProtectedMember
        return self._backend._id

    def add_scatter_plot(self, network):
        plot = network.scatter_plot()
        plot.parent = self.network_view.scene
        self.network_view.add(plot)

    def add_outer_box(self, network):
        box: visuals.Box = network.outer_grid
        box.parent = self.network_view.scene
        self.network_view.add(box)

    def add_selector_box(self, network):
        box: visuals.Box = network.selector_box()
        box.parent = self.network_view.scene
        self.network_view.add(box)

    def add_voltage_plot(self, network):
        line: visuals.Box = network.voltage_plot()
        line.parent = self.voltage_plot_view.scene
        self.voltage_plot_view.add(line)

    def set_keys(self, keys):
        self.unfreeze()
        # noinspection PyProtectedMember
        self._set_keys(keys)
        self.freeze()


class EngineWindow(QtWidgets.QMainWindow):
    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 network=None,
                 show=True,
                 keys=None
                 ):
        super().__init__()

        for attr in ['ui', 'scene', 'main_view']:
            if hasattr(self, attr):
                raise AttributeError(f'\'{attr}\' ')

        self.setObjectName(name)
        self.resize(800, 600)

        # self.central_scene = self.set_central_scene(keys=keys, app=app)
        self.central_scene = EngineSceneCanvas(conf=CanvasConfig(keys=keys), app=app, network=network)

        self.ui = UiMainWindow(self)
        self.ui.central_layout.addWidget(self.central_scene.native)

        if show is True:
            self.show()

    def set_keys(self, keys):
        self.central_scene.set_keys(keys)

