from dataclasses import asdict, dataclass
from typing import Optional, Union
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QIcon, QAction

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
)

from vispy.app import Application, Canvas
from vispy.color import Color
from vispy.gloo.context import GLContext
from vispy.scene import SceneCanvas, ViewBox, visuals


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

        self.central_widget = QtWidgets.QWidget(self.window)
        # self.central_widget.setObjectName("central_widget")

        self.grid_layout = QtWidgets.QGridLayout(self.central_widget)
        # self.grid_layout.setObjectName("grid_layout")

        self.frame_3d = QtWidgets.QFrame(self.central_widget)
        self.frame_3d.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_3d.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.central_layout = QtWidgets.QVBoxLayout(self.frame_3d)
        self.grid_layout.addWidget(self.frame_3d, 0, 0, 1, 1)

        self.buttons = UiButtons(self.window)
        hbox = QHBoxLayout()
        # hbox.addStretch(1)
        hbox.addWidget(self.buttons.ok)
        hbox.addWidget(self.buttons.toggle_outergrid)
        hbox.addWidget(self.buttons.cancel)

        self.grid_layout.addLayout(hbox, 1, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignCenter)

        self.horizontalSlider = QtWidgets.QSlider(self.central_widget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")

        self.grid_layout.addWidget(self.horizontalSlider, 2, 0, 1, 1)

        self.window.setCentralWidget(self.central_widget)

        self.window.setStatusBar(QtWidgets.QStatusBar(self.window))

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


class EngineScene(SceneCanvas):

    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application]):
        conf = conf or CanvasConfig()
        super().__init__(**asdict(conf), app=app)

    @property
    def window(self):
        # noinspection PyProtectedMember
        return self._backend._id


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

        self.scene = EngineScene(conf=CanvasConfig(keys=keys), app=app)

        self.ui = UiMainWindow(self)

        self.central_view = ViewBox()
        self.central_view.camera = 'turntable'  # or try 'arcball'
        # add a colored 3D axis for orientation
        visuals.XYZAxis(parent=self.central_view.scene)
        self.scene.central_widget.add_widget(self.central_view)

        self.ui.central_layout.addWidget(self.scene.native)

        if network is not None:
            self.add_scatter_plot(network)
            self.add_outer_box(network)
            self.add_selector_box(network)

        if show is True:
            self.show()

    def add_scatter_plot(self, network):
        plot = network.scatter_plot()
        plot.parent = self.central_view.scene
        self.central_view.add(plot)

    def add_outer_box(self, network):
        box: visuals.Box = network.outer_grid
        box.parent = self.central_view.scene
        self.central_view.add(box)

    def add_selector_box(self, network):
        box: visuals.Box = network.selector_box()
        network.selector_box._parent = self.central_view.scene
        box.parent = self.central_view.scene
        self.central_view.add(box)

    def set_keys(self, keys):
        self.scene.unfreeze()
        self.scene._set_keys(keys)
        self.scene.freeze()

    # def keyPressEvent(self, e):
    #     if e.key() == QtCore.Qt.Key.Key_F5:
    #         print('FF')
    #         self.close()


