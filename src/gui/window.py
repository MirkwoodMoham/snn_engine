from dataclasses import asdict, dataclass
from typing import Optional, Union

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QAction

from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget
)

from vispy.app import Application
from .engine_scene_canvas import EngineSceneCanvas, CanvasConfig

from network import SpikingNeuronNetwork
from .ui_element import ButtonMenuAction, SpinBoxSlider
from .collapsible_widget.collapsible_widget import CollapsibleWidget


@dataclass
class ButtonMenuActions:

    window: Optional[QtWidgets.QMainWindow] = None

    START_SIMULATION: ButtonMenuAction = ButtonMenuAction(menu_name='&Start Simulation',
                                                          menu_short_cut='F9',
                                                          status_tip='Start Simulation',
                                                          icon_name='control.png')

    PAUSE_SIMULATION: ButtonMenuAction = ButtonMenuAction(menu_name='&Pause Simulation',
                                                          menu_short_cut='F10',
                                                          status_tip='Pause Simulation',
                                                          icon_name='control-pause.png',
                                                          disabled=True)

    EXIT_APP: ButtonMenuAction = ButtonMenuAction(menu_name='&Exit',
                                                  name='Exit',
                                                  status_tip='Close Application',
                                                  menu_short_cut='Ctrl+Q')

    TOGGLE_OUTERGRID: ButtonMenuAction = ButtonMenuAction(menu_name='&OuterGrid',
                                                          name='Show OuterGrid',
                                                          status_tip='Show/Hide OuterGrid',
                                                          menu_short_cut='Ctrl+G', checkable=True)

    def __post_init__(self):
        window_ = self.window
        self.window = None
        dct = asdict(self)
        self.window = window_
        print()
        for k in dct:
            v = getattr(self, k)
            if isinstance(v, ButtonMenuAction):
                if (v.menu_short_cut is not None) and (v.status_tip is not None):
                    v.status_tip = v.status_tip + f" ({v.menu_short_cut})"
                    print(v.status_tip)
                if v.window is None:
                    v.window = self.window


class UI(object):

    class MenuBar:
        class MenuActions:
            def __init__(self):
                self.start: QAction = ButtonMenuActions.START_SIMULATION.action()
                self.pause: QAction = ButtonMenuActions.PAUSE_SIMULATION.action()
                self.toggle_outergrid: QAction = ButtonMenuActions.TOGGLE_OUTERGRID.action()
                self.exit: QAction = ButtonMenuActions.EXIT_APP.action()
                self.exit.triggered.connect(QApplication.instance().quit)

        def __init__(self, window):
            self.actions = self.MenuActions()

            self.bar = QtWidgets.QMenuBar(window)

            self.bar.setGeometry(QtCore.QRect(0, 0, 440, 130))
            self.bar.setObjectName("menubar")

            self.file_menu = self.bar.addMenu('&File')
            self.file_menu.addAction(self.actions.start)
            self.file_menu.addAction(self.actions.pause)
            self.file_menu.addAction(self.actions.exit)

            self.view_menu = self.bar.addMenu('&View')
            self.view_menu.addAction(self.actions.toggle_outergrid)

    class UiLeft:

        class Buttons:
            def __init__(self):
                max_width = 140
                self.start: QPushButton = ButtonMenuActions.START_SIMULATION.button()
                self.pause: QPushButton = ButtonMenuActions.PAUSE_SIMULATION.button()
                self.exit: QPushButton = ButtonMenuActions.EXIT_APP.button()
                self.toggle_outergrid: QPushButton = ButtonMenuActions.TOGGLE_OUTERGRID.button()

                self.toggle_outergrid.setMinimumWidth(max_width)
                self.toggle_outergrid.setMaximumWidth(max_width)
                self.start.setMaximumWidth(max_width)
                self.exit.setMaximumWidth(max_width)

                self.exit.clicked.connect(QApplication.instance().quit)

        class Sliders:
            def __init__(self, window):

                self.thalamic_inh_input_current = SpinBoxSlider(name='Inhibitory Current [I]',
                                                                window=window,
                                                                status_tip='Thalamic Inhibitory Input Current [I]',
                                                                prop_id='thalamic_inh_input_current',
                                                                maximum_width=300,
                                                                _min_value=0, _max_value=250)
                self.thalamic_exc_input_current = SpinBoxSlider(name='Excitatory Current [I]',
                                                                window=window,
                                                                status_tip='Thalamic Excitatory Input Current [I]',
                                                                prop_id='thalamic_exc_input_current',
                                                                maximum_width=300,
                                                                _min_value=0, _max_value=250)

                self.sensory_input_current0 = SpinBoxSlider(name='Input Current 0 [I]',
                                                            window=window,
                                                            status_tip='Sensory Input Current 0 [I]',
                                                            prop_id='sensory_input_current0',
                                                            maximum_width=300,
                                                            _min_value=0, _max_value=200)
                self.sensory_input_current1 = SpinBoxSlider(name='Input Current 1 [I]',
                                                            window=window,
                                                            status_tip='Sensory Input Current 1 [I]',
                                                            prop_id='sensory_input_current1',
                                                            maximum_width=300,
                                                            _min_value=0, _max_value=200)

                self.sensory_weight = SpinBoxSlider(name='Sensory',
                                                    boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                                                    window=window,
                                                    func_=lambda x: float(x) / 100000 if x is not None else x,
                                                    func_inv_=lambda x: int(x * 100000) if x is not None else x,
                                                    status_tip='Sensory Weight',
                                                    prop_id='src_weight',
                                                    maximum_width=300,
                                                    single_step_spin_box=0.01,
                                                    single_step_slider=100,
                                                    _min_value=0, _max_value=5)

        def __init__(self, window, central_widget):

            self.frame = QScrollArea(central_widget)
            self.frame.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            self.frame.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            # self.frame.setGeometry(0, 0, 100, 100)
            self.frame.setWidgetResizable(True)

            widget = QWidget()
            self.frame.setWidget(widget)
            self.buttons = self.Buttons()
            self.sliders = self.Sliders(window)
            self.layout = QVBoxLayout()
            widget.setLayout(self.layout)

            self.layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            self.layout.setSpacing(2)

            play_pause_widget = QWidget(central_widget)
            play_pause_widget.setFixedSize(95, 45)
            play_pause_hbox = QHBoxLayout(play_pause_widget)
            play_pause_hbox.setContentsMargins(0, 0, 0, 0)
            play_pause_hbox.setSpacing(2)
            play_pause_hbox.addWidget(self.buttons.start)
            play_pause_hbox.addWidget(self.buttons.pause)

            self.sensory_input_collapsible = CollapsibleWidget(title='Sensory Input')
            self.sensory_input_collapsible.add(self.sliders.sensory_input_current0.widget)
            self.sensory_input_collapsible.add(self.sliders.sensory_input_current1.widget)

            self.thalamic_input_collapsible = CollapsibleWidget(title='Thalamic Input')
            self.thalamic_input_collapsible.add(self.sliders.thalamic_inh_input_current.widget)
            self.thalamic_input_collapsible.add(self.sliders.thalamic_exc_input_current.widget)

            self.weights_collapsible = CollapsibleWidget(title='Weights')
            self.weights_collapsible.add(self.sliders.sensory_weight.widget)

            self.objects_collapsible = CollapsibleWidget(title='Objects')

            self.layout.addWidget(play_pause_widget)
            self.layout.addWidget(self.buttons.toggle_outergrid)
            self.layout.addWidget(self.weights_collapsible, 1)
            self.layout.addWidget(self.sensory_input_collapsible, 1)
            self.layout.addWidget(self.thalamic_input_collapsible, 1)

            self.layout.addWidget(self.objects_collapsible, 1)

            self.layout.addWidget(self.buttons.exit)

    def __init__(self, window, scene):

        self.ui_elements = ButtonMenuActions(window)

        self.menubar = self.MenuBar(window)
        window.setMenuBar(self.menubar.bar)

        self.central_widget = QWidget(window)

        self.frame_3d = QFrame(self.central_widget)
        self.frame_3d.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3d.setFrameShadow(QFrame.Shadow.Raised)
        self.layout_3d = QVBoxLayout(self.frame_3d)
        self.layout_3d.addWidget(scene)
        # self.grid_layout.addWidget(self.frame_3d, 0, 1, 1, 30)

        self.ui_left = self.UiLeft(window, self.central_widget)
        
        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.ui_left.frame)
        splitter.addWidget(self.frame_3d)

        hbox = QHBoxLayout(self.central_widget)
        hbox.addWidget(splitter)

        window.setCentralWidget(self.central_widget)

        window.setStatusBar(QStatusBar(window))

        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 3)

        self.retranslate_ui(window)

    @staticmethod
    def retranslate_ui(window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("SNN Engine", "SNN Engine"))


class EngineWindow(QtWidgets.QMainWindow):

    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 network: SpikingNeuronNetwork,
                 show=True,
                 keys=None
                 ):
        super().__init__()

        for attr in ['gui', 'scene_3d', 'main_view']:
            if hasattr(self, attr):
                raise AttributeError(f'\'{attr}\' ')

        self.setObjectName(name)
        self.resize(1200, 800)

        # self.central_scene = self.set_central_scene(keys=keys, app=app)
        self.scene_3d = EngineSceneCanvas(conf=CanvasConfig(keys=keys), app=app, network=network)
        self.ui = UI(self, self.scene_3d.native)

        if show is True:
            self.show()

    def set_keys(self, keys):
        self.scene_3d.set_keys(keys)
