from dataclasses import asdict, dataclass

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QLabel,
    QSpinBox
)
from typing import Optional, Type

from app.content.widgets.collapsible_widget.collapsible_widget import CollapsibleWidget
from .widgets.spin_box_sliders import SpinBoxSlider, SubCollapsibleFrame, SliderCollection
from app.content.widgets.combobox_frame import ComboBoxFrame
from .scenes import SingleNeuronPlotCanvas
from interfaces import IzhikevichNeuronsInterface, NeuronInterface, SignalModel, SignalVariable
from network import IzhikevichPreset, IzhikevichModel, SpikingNeuronNetwork, StateRow
from app.content.widgets.scene_canvas_frame import SceneCanvasFrame, CanvasConfig


class NeuronPropertiesFrame(SubCollapsibleFrame):

    @dataclass
    class Sliders(SliderCollection):

        pt: Optional[SpinBoxSlider] = None
        a: Optional[SpinBoxSlider] = None
        b: Optional[SpinBoxSlider] = None
        c: Optional[SpinBoxSlider] = None
        d: Optional[SpinBoxSlider] = None

    def __init__(self, parent, window, model,
                 interface: NeuronInterface,
                 fixed_width=300):

        super().__init__(parent, fixed_width=fixed_width)

        self.preset_combo_box_frame = ComboBoxFrame(
            'preset', max_width=300,
            item_list=list(interface.preset_dct.keys()))
        self.preset_combo_box_frame.connect(interface.use_preset)
        self.preset_combo_box_frame.connect(self.actualize_values)

        self.sliders = self.Sliders(parent=self)
        self.sliders.add(self.preset_combo_box_frame)

        for x in self.sliders.keys:
            row_def: StateRow = getattr(model._rows, x)
            self.sliders.add_slider(
                x, interface,
                name=x + ':',
                window=window,
                _min_value=row_def.interval[0],
                _max_value=row_def.interval[1],
                boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                prop_id=x,
                single_step_spin_box=row_def.step_size,
                single_step_slider=row_def.step_size * 1000)

        self.setFixedHeight(self.sliders.widget.maximumHeight() + 5)
        self.layout().addWidget(self.sliders.widget)

    def actualize_values(self):
        print('actualize_values')
        for k in self.sliders.keys:
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


class CurrentControlFrame(SubCollapsibleFrame):

    @dataclass
    class Sliders(SliderCollection):
        amplitude: Optional[SpinBoxSlider] = None
        step_time: Optional[SpinBoxSlider] = None
        period: Optional[SpinBoxSlider] = None
        duty: Optional[SpinBoxSlider] = None
        duty_period: Optional[SpinBoxSlider] = None
        phase: Optional[SpinBoxSlider] = None

    def __init__(self, parent, model: SignalModel, window, fixed_width=300):

        super().__init__(parent, fixed_width=fixed_width)

        self.sliders = self.Sliders(parent=self)

        for x in self.sliders.keys:
            if hasattr(model.VariableConfig, x):
                var_conf: SignalVariable = getattr(model.VariableConfig, x)
                self.sliders.add_slider(
                    x, model,
                    name=x + ':',
                    window=window,
                    _min_value=var_conf.interval[0],
                    _max_value=var_conf.interval[1],
                    boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                    prop_id=x,
                    single_step_spin_box=var_conf.step_size,
                    single_step_slider=var_conf.step_size * 1000,
                    suffix=f' [{var_conf.unit}]')

        self.setFixedHeight(self.sliders.widget.maximumHeight() + 5)
        self.layout().addWidget(self.sliders.widget)


class SingleNeuronPlotCollapsible(CollapsibleWidget):

    def __init__(self, parent, app, network, interface: NeuronInterface, title='plot'):

        super().__init__(parent, title=title)

        width_min = 200
        width_max = 800
        height_min = 150
        height_max = 250
        self.canvas = SingleNeuronPlotCanvas(
            conf=CanvasConfig(), app=app, plotting_config=network.plotting_config,
            width_min=width_min, width_max=width_max,
            height_min=height_min, height_max=height_max
        )

        interface.link_plot_widget(self.canvas.plot_widget)
        plot_frame = SceneCanvasFrame(self, self.canvas)
        plot_frame.setFixedSize(width_max+80, height_max+50)

        self.add(plot_frame)

        self.canvas.set_current()
        interface.register_vbos()


class SingleNeuronCollapsible(CollapsibleWidget):

    def __init__(self, parent, network: SpikingNeuronNetwork,
                 model: Type[IzhikevichModel], title, window, app):

        super().__init__(parent=parent, title=title)

        self.neuron_interface = IzhikevichNeuronsInterface(0, network)

        self.id = NeuronIDFrame(self, network.network_config.N)
        self.model_collapsible = CollapsibleWidget(self, title='model')
        self.model_collapsible._title_frame.layout()

        self.model = NeuronPropertiesFrame(self.model_collapsible, window, model,
                                           interface=self.neuron_interface)
        self.add(self.id)
        self.model_collapsible.add(self.model)
        self.add(self.model_collapsible)

        self.id.spinbox.valueChanged.connect(self.actualize_values)

        self.plot = SingleNeuronPlotCollapsible(self, app=app, network=network,
                                                interface=self.neuron_interface)

        self.add(self.plot)

        self.current_control_collapsible = CollapsibleWidget(self, 'current_control')

        self.set_current_control_widget(interface=self.neuron_interface, window=window)

        self.add(self.current_control_collapsible)

    def set_current_control_widget(self, interface: NeuronInterface, window):
        current_control_frame = CurrentControlFrame(
            parent=self,
            model=interface.current_injection_function,
            window=window
        )
        self.current_control_collapsible.add(current_control_frame)

    def actualize_values(self):
        self.model.preset_combo_box_frame.combo_box.setCurrentIndex(0)
        self.neuron_interface.id = self.id.spinbox.value()
        self.model.actualize_values()

    def update_plots(self, t, t_mod) -> None:
        self.neuron_interface.update_plot(t, t_mod)
        self.plot.canvas.update()

    def set_id(self, neuron_id):
        self.id.spinbox.setValue(neuron_id)
