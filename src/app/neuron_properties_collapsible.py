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
from .engine_scene_canvas import SingleNeuronPlotCanvas, CanvasConfig
from interfaces import IzhikevichNeuronsInterface
from network import IzhikevichModel, NetworkConfig, SpikingNeuronNetwork, StateRow


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
        # self.layout().addWidget(QLabel('Model'))
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
        self.setFixedHeight(28)


class SingleNeuronPlotFrame:
    pass


class IzhikevichNeuronCollapsible(CollapsibleWidget):

    def __init__(self, parent, network: SpikingNeuronNetwork,
                 model: Type[IzhikevichModel], title, window, app):

        super().__init__(parent=parent, title=title)

        self.neuron_interface = IzhikevichNeuronsInterface(0, network)

        self.id = NeuronIDFrame(self, network.network_config.N)
        self.model_collapsible = CollapsibleWidget(self, title='model')
        self.model = IzhikevichNeuronPropertiesFrame(self.model_collapsible, window, model,
                                                     interface=self.neuron_interface)
        self.add(self.id)
        self.model_collapsible.add(self.model)
        self.add(self.model_collapsible)

        self.id.spinbox.valueChanged.connect(self.actualize_values)

        self.plot_collapsible = CollapsibleWidget(self, title='plot')
        width_min = 300
        width_max = 300
        height_min = 150
        height_max = 150
        self.plot_canvas = SingleNeuronPlotCanvas(
            conf=CanvasConfig(), app=app, plotting_config=network.plotting_config,
            width_min=width_min, width_max=width_max,
            height_min=height_min, height_max=height_max
        )
        self.plot_canvas.plot_widget.view.add(self.neuron_interface.plot)
        plot_frame: QFrame = self._canvas_frame(self.plot_canvas)
        plot_frame.setFixedSize(width_max+60, height_max+40)

        self.plot_collapsible.add(plot_frame)
        self.add(self.plot_collapsible)
        self.plot_canvas.set_current()
        self.neuron_interface.register_vbo()

    def actualize_values(self):
        self.neuron_interface.id = self.id.spinbox.value()
        self.model.actualize_values()

    def update_plots(self, t) -> None:
        self.neuron_interface.update_plot(t)
        self.plot_canvas.update()
