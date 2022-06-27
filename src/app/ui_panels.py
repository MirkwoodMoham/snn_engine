from dataclasses import dataclass, asdict
from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QLabel,
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import ButtonMenuAction
from app.collapsible_widget.collapsible_widget import CollapsibleWidget
from app.gui_element import SpinBoxSlider


@dataclass
class ButtonMenuActions:

    """

    Declarative style. Must be initialized once.

    """

    window: Optional[QMainWindow] = None

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


class MenuBar(QMenuBar):
    class MenuActions:
        def __init__(self):
            self.start: QAction = ButtonMenuActions.START_SIMULATION.action()
            self.pause: QAction = ButtonMenuActions.PAUSE_SIMULATION.action()
            self.toggle_outergrid: QAction = ButtonMenuActions.TOGGLE_OUTERGRID.action()
            self.exit: QAction = ButtonMenuActions.EXIT_APP.action()
            self.exit.triggered.connect(QApplication.instance().quit)

    def __init__(self, window):

        super().__init__(window)
        self.actions = self.MenuActions()

        self.setGeometry(QtCore.QRect(0, 0, 440, 130))
        self.setObjectName("menubar")

        self.file_menu = self.addMenu('&File')
        self.file_menu.addAction(self.actions.start)
        self.file_menu.addAction(self.actions.pause)
        self.file_menu.addAction(self.actions.exit)

        self.view_menu = self.addMenu('&View')
        self.view_menu.addAction(self.actions.toggle_outergrid)


class UIPanel(QScrollArea):

    def __init__(self, window):
        super().__init__(window.centralWidget())
        # self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setWidgetResizable(True)

        self.setWidget(QWidget(self))
        self.widget().setLayout(QVBoxLayout())

        self.widget().layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.widget().layout().setSpacing(2)

    # noinspection PyPep8Naming
    def addWidget(self, *args):
        self.widget().layout().addWidget(*args)


class MainUILeft(UIPanel):

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

    def __init__(self, window):

        super().__init__(window)

        self.buttons = self.Buttons()
        self.sliders = self.Sliders(window)

        play_pause_widget = QWidget(self)
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

        self.addWidget(play_pause_widget)
        self.addWidget(self.buttons.toggle_outergrid)
        self.addWidget(self.weights_collapsible, 1)
        self.addWidget(self.sensory_input_collapsible, 1)
        self.addWidget(self.thalamic_input_collapsible, 1)

        self.addWidget(self.objects_collapsible, 1)

        self.addWidget(self.buttons.exit)


class GroupInfoComboBox(QComboBox):

    def __init__(self, item_list=None):

        super().__init__()

        if item_list is not None:
            self.add_items(item_list)

        self.setFixedHeight(28)

    def add_items(self, item_list):
        for item in item_list:
            self.addItem(item)


class GroupInfoComboBoxFrame(QFrame):

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.combo_box = GroupInfoComboBox()

        self.layout().addWidget(QLabel(name))
        self.layout().addWidget(self.combo_box)

        self.setFixedHeight(32)
        # self.setFra

    def add_items(self, item_list):
        self.combo_box.add_items(item_list)

    def connect(self, func):
        # noinspection PyUnresolvedReferences
        self.combo_box.currentTextChanged.connect(func)

    # noinspection PyPep8Naming
    def setCurrentIndex(self, index):
        self.combo_box.setCurrentIndex(index)


class GroupInfoPanel(UIPanel):

    def __init__(self, window):

        super().__init__(window)

        self.combo_boxes_collapsible = CollapsibleWidget(title='Group Info Display 0')
        # self.text_display_collapsible1 = CollapsibleWidget(title='Text Display 1')
        # self.text_display_collapsible0.add(combobox1)
        self.addWidget(self.combo_boxes_collapsible)
        # self.addWidget(self.text_display_collapsible1)

        self.combo_boxes = []

        self.combo_box_frame0 = GroupInfoComboBoxFrame('combo_box0')
        self.add_combo_box(self.combo_box_frame0)

    def add_combo_box(self, combo_box):

        self.combo_boxes_collapsible.add(combo_box)
        # self.addWidget(combo_box)
        self.combo_boxes.append(combo_box)
