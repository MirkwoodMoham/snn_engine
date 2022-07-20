from dataclasses import asdict, dataclass

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget
)
from typing import Type

from app.collapsible_widget.collapsible_widget import CollapsibleWidget
from app.gui_element import SpinBoxSlider, SubCollapsibleFrame
from network import IzhikevichModel, NetworkConfig, SpikingNeuronNetwork, StateRow
from interfaces import IzhikevichNeuronsInterface


class IzhikevichNeuronPropertiesFrame(SubCollapsibleFrame):

    @dataclass
    class Sliders:
        pt: SpinBoxSlider = None
        # v: SpinBoxSlider = None
        a: SpinBoxSlider = None
        b: SpinBoxSlider = None
        c: SpinBoxSlider = None
        d: SpinBoxSlider = None
        # u: SpinBoxSlider = None

    def __init__(self, parent, window, model,
                 interface: IzhikevichNeuronsInterface,
                 fixed_width=300):

        super().__init__(parent, fixed_width=fixed_width)
        self.layout().addWidget(QLabel('Model'))

        self.sliders = self.Sliders()

        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)

        self._keys = list(asdict(self.sliders).keys())

        for x in self._keys:
            row_def: StateRow = getattr(model._rows, x)
            sbs = SpinBoxSlider(name=x + ':',
                                window=window,
                                _min_value=row_def.interval[0],
                                _max_value=row_def.interval[1],
                                boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                                prop_id=x,
                                single_step_spin_box=row_def.step_size,
                                single_step_slider=row_def.step_size * 1000)
            setattr(self.sliders, x, sbs)
            sbs.connect_property(interface)
            sliders_layout.addWidget(sbs.widget)

        max_height = 25 + 35 * len(self._keys)
        self.setFixedHeight(max_height)
        sliders_widget.setMaximumHeight(max_height - 5)
        self.layout().addWidget(sliders_widget)

    def actualize_values(self):
        for k in self._keys:
            getattr(self.sliders, k).actualize_values()


class NeuronIDFrame(SubCollapsibleFrame):

    def __init__(self, parent, N: int, fixed_width=300):

        super().__init__(parent, fixed_width=fixed_width)

        self.layout().addWidget(QLabel('ID'))
        self.spinbox = QSpinBox(self)
        self.layout().addWidget(self.spinbox)
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(N-1)


class SingleNeuronPlotFrame:
    pass


class IzhikevichNeuronCollapsible(CollapsibleWidget):

    def __init__(self, network: SpikingNeuronNetwork, model: Type[IzhikevichModel], title, window, parent=None):

        super().__init__(parent=parent, title=title)
        self.neuron_interface = IzhikevichNeuronsInterface(0, network)

        self.id = NeuronIDFrame(self, network.network_config.N)
        self.model = IzhikevichNeuronPropertiesFrame(self, window, model, interface=self.neuron_interface)

        self.add(self.id)
        self.add(self.model)

        self.id.spinbox.valueChanged.connect(self.actualize_values)

    def actualize_values(self):
        self.neuron_interface.id = self.id.spinbox.value()
        self.model.actualize_values()

