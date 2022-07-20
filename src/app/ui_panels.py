from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QValidator
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
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

from .gui_element import (
    ButtonMenuAction
)
from .rendered_object_collapsible import RenderedObjectCollapsible
from .neuron_properties_collapsible import IzhikevichNeuronCollapsible
from .collapsible_widget.collapsible_widget import CollapsibleWidget
from .gui_element import SpinBoxSlider
from network import IzhikevichModel, NetworkConfig, SpikingNeuronNetwork


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
                                                          menu_short_cut='Ctrl+G',
                                                          checkable=True)

    ACTUALIZE_G_FLAGS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&Refresh displayed G_flags',
                                                                menu_short_cut='F7',
                                                                icon_name='arrow-circle.png',
                                                                status_tip='Refresh displayed G_flags values')

    ACTUALIZE_G_PROPS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&Refresh displayed G2G_info',
                                                                menu_short_cut='F6',
                                                                icon_name='arrow-circle.png',
                                                                status_tip='Refresh displayed G2G_info values')

    ACTUALIZE_G2G_INFO_TEXT: ButtonMenuAction = ButtonMenuAction(
        menu_short_cut='F5',
        menu_name='&Refresh displayed G2G_flags ',
        icon_name='arrow-circle.png',
        status_tip='Refresh displayed G2G_flags values')

    TOGGLE_GROUP_IDS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&Group IDs',
                                                               menu_short_cut='Ctrl+F8',
                                                               checkable=True,
                                                               status_tip='Show/Hide Group IDs')

    TOGGLE_G_FLAGS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&G_flags Text',
                                                             checkable=True,
                                                             menu_short_cut='Ctrl+F7',
                                                             status_tip='Show/Hide G_flags values')

    TOGGLE_G_PROPS_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&G_Props Text',
                                                             checkable=True,
                                                             menu_short_cut='Ctrl+F6',
                                                             status_tip='Show/Hide G_props values')

    TOGGLE_G2G_INFO_TEXT: ButtonMenuAction = ButtonMenuAction(menu_name='&G2G_info Text',
                                                              checkable=True,
                                                              menu_short_cut='Ctrl+F5',
                                                              status_tip='Show/Hide G2G_info values')

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

            self.toggle_groups_ids: QAction = ButtonMenuActions.TOGGLE_GROUP_IDS_TEXT.action()
            self.toggle_g_flags: QAction = ButtonMenuActions.TOGGLE_G_FLAGS_TEXT.action()
            self.toggle_g_props: QAction = ButtonMenuActions.TOGGLE_G_PROPS_TEXT.action()
            self.toggle_g2g_info: QAction = ButtonMenuActions.TOGGLE_G2G_INFO_TEXT.action()

            self.actualize_g_flags: QAction = ButtonMenuActions.ACTUALIZE_G_FLAGS_TEXT.action()
            self.actualize_g_props: QAction = ButtonMenuActions.ACTUALIZE_G_PROPS_TEXT.action()
            self.actualize_g2g_info: QAction = ButtonMenuActions.ACTUALIZE_G2G_INFO_TEXT.action()

    def __init__(self, window):

        super().__init__(window)
        self.actions = self.MenuActions()

        self.setGeometry(QRect(0, 0, 440, 130))
        self.setObjectName("menubar")

        self.file_menu = self.addMenu('&File')
        self.file_menu.addAction(self.actions.start)
        self.file_menu.addAction(self.actions.pause)
        self.file_menu.addAction(self.actions.exit)

        self.view_menu = self.addMenu('&View')
        self.view_menu.addAction(self.actions.toggle_outergrid)

        self.view_menu.addAction(self.actions.toggle_groups_ids)

        self.view_menu.addAction(self.actions.toggle_g_flags)
        self.view_menu.addAction(self.actions.actualize_g_flags)
        self.view_menu.addAction(self.actions.toggle_g_props)
        self.view_menu.addAction(self.actions.actualize_g_props)
        self.view_menu.addAction(self.actions.toggle_g2g_info)
        self.view_menu.addAction(self.actions.actualize_g2g_info)


class UIPanel(QScrollArea):

    def __init__(self, window):
        super().__init__(window.centralWidget())
        self.setWidgetResizable(True)

        self.setWidget(QWidget(self))
        self.widget().setLayout(QVBoxLayout())

        self.widget().layout().setAlignment(Qt.AlignmentFlag.AlignTop)
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
                                                boxlayout_orientation=Qt.Orientation.Horizontal,
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

        self.window = window

        self.buttons = self.Buttons()
        self.sliders = self.Sliders(window)

        play_pause_widget = QWidget(self)
        play_pause_widget.setFixedSize(95, 45)
        play_pause_hbox = QHBoxLayout(play_pause_widget)
        play_pause_hbox.setContentsMargins(0, 0, 0, 0)
        play_pause_hbox.setSpacing(2)
        play_pause_hbox.addWidget(self.buttons.start)
        play_pause_hbox.addWidget(self.buttons.pause)

        self.neuron0 = None
        self.neuron1 = None
        self.neurons_collapsible = CollapsibleWidget(title='Neuron Info')
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
        self.addWidget(self.neurons_collapsible)
        self.addWidget(self.weights_collapsible)
        self.addWidget(self.sensory_input_collapsible)
        self.addWidget(self.thalamic_input_collapsible)

        self.addWidget(self.objects_collapsible)

        self.addWidget(self.buttons.exit)

    def add_3d_object_sliders(self, obj):

        collapsible = RenderedObjectCollapsible(obj, self)
        self.objects_collapsible.add(collapsible)
        collapsible.toggle_collapsed()
        self.objects_collapsible.toggle_collapsed()
        self.objects_collapsible.toggle_collapsed()

    def add_neurons_slider(self, network: SpikingNeuronNetwork, model=IzhikevichModel):
        self.neuron0 = IzhikevichNeuronCollapsible(network, title='Neuron0', model=model, window=self.window)
        self.neuron1 = IzhikevichNeuronCollapsible(network, title='Neuron1', model=model, window=self.window)
        self.neurons_collapsible.add(self.neuron0)
        self.neurons_collapsible.add(self.neuron1)
        # noinspection PyUnresolvedReferences
        # self.widget().layout().insertWidget(2, self.neurons_collapsible)
        self.neurons_collapsible.toggle_collapsed()


class GroupInfoComboBox(QComboBox):

    def __init__(self, item_list=None):

        super().__init__()

        if item_list is not None:
            self.add_items(item_list)

        self.setFixedHeight(28)

    def add_items(self, item_list, set_current=1):
        for item in item_list:
            self.addItem(item)
        if set_current is not None:
            self.setCurrentIndex(set_current)


class GroupInfoComboBoxFrame(QFrame):

    def __init__(self, name, actualize_button=None, parent=None):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.combo_box = GroupInfoComboBox()
        self.label = QLabel(name)
        self.label.setMaximumWidth(80)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.combo_box)
        if actualize_button is not None:
            self.actualize_button = actualize_button
            self.layout().addWidget(self.actualize_button)
        else:
            self.actualize_button = None

        # self.setBaseSize(32, 100)
        self.setFixedHeight(32)
        # self.setMaximumWidth(220)
        # self.setFra

    def __call__(self):
        return self.combo_box

    def connect(self, func):
        # noinspection PyUnresolvedReferences
        self.combo_box.currentTextChanged.connect(func)
        if self.actualize_button is not None:
            self.actualize_button.clicked.connect(func)


class GroupValidator(QValidator):

    def __init__(self, parent, group_ids):
        self.group_ids = group_ids
        super().__init__(parent)

    def validate(self, a0: str, a1: int) -> Tuple['QValidator.State', str, int]:
        if a0 in self.group_ids:
            state = QValidator.State.Acceptable
        else:
            state = QValidator.State.Invalid
        return (state, a0, a1)


class G2GInfoComboBoxFrame(QFrame):

    def __init__(self, name, actualize_button=None, parent=None):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.src_group_combo_box = GroupInfoComboBox()
        self.src_group_combo_box.setMaximumWidth(50)
        self.src_group_combo_box.setMaxVisibleItems(10)
        self.src_group_combo_box.setEditable(True)
        self.value_combo_box = GroupInfoComboBox()
        self.label = QLabel(name)
        self.label.setMaximumWidth(80)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.src_group_combo_box)
        self.layout().addWidget(self.value_combo_box)
        if actualize_button is not None:
            self.actualize_button = actualize_button
            self.layout().addWidget(self.actualize_button)
        else:
            self.actualize_button = None

        self.setFixedHeight(32)

    def __call__(self):
        return self.value_combo_box

    def connect(self, func):
        # noinspection PyUnresolvedReferences
        self.src_group_combo_box.currentTextChanged.connect(func)
        # noinspection PyUnresolvedReferences
        self.value_combo_box.currentTextChanged.connect(func)
        if self.actualize_button is not None:
            self.actualize_button.clicked.connect(func)

    def set_src_group_validator(self, groups_ids):
        v = GroupValidator(self, groups_ids)
        self.src_group_combo_box.setValidator(v)

    def init_src_group_combo_box(self, group_ids):
        self.src_group_combo_box.add_items(group_ids, 0)
        self.set_src_group_validator(group_ids)


class GroupInfoPanel(UIPanel):

    def __init__(self, window):

        super().__init__(window)

        self.combo_boxes_collapsible0 = CollapsibleWidget(title='Group Info Display 0')

        self.combo_boxes = []

        self.group_ids_combobox = GroupInfoComboBoxFrame('Group IDs')
        self.add_combo_box(self.group_ids_combobox)

        self.g_flags_combobox = GroupInfoComboBoxFrame(
            'G_flags', ButtonMenuActions.ACTUALIZE_G_FLAGS_TEXT.button())
        self.add_combo_box(self.g_flags_combobox)

        self.g_props_combobox = GroupInfoComboBoxFrame(
            'G_props', ButtonMenuActions.ACTUALIZE_G_PROPS_TEXT.button())
        self.add_combo_box(self.g_props_combobox)

        self.combo_boxes_collapsible1 = CollapsibleWidget(title='Group Info Display 1')
        self.g2g_info_combo_box = G2GInfoComboBoxFrame(
            'G2G_info', ButtonMenuActions.ACTUALIZE_G2G_INFO_TEXT.button())
        self.combo_boxes_collapsible1.add(self.g2g_info_combo_box)

        self.addWidget(self.combo_boxes_collapsible0)
        self.addWidget(self.combo_boxes_collapsible1)

    def add_combo_box(self, combo_box):

        self.combo_boxes_collapsible0.add(combo_box)
        # self.addWidget(combo_box)
        self.combo_boxes.append(combo_box)
