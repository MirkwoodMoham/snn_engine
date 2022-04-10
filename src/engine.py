import numba.cuda
import numpy as np
from vispy.app import Timer
import sys

from network import SpikingNeuronNetwork, NetworkConfig, PlottingConfig
from ui import EngineWindow, BackendApp
from simulation import vbodata2host


class EngineConfig:

    N: int = 3 * 10 ** 5
    T: int = 2000  # Max simulation duration

    device: int = 0

    max_batch_size_mb: int = 3000

    network_config = NetworkConfig(N=N, N_pos_shape=(1, 1, 1))
    plotting_config = PlottingConfig(N=N,
                                     n_voltage_plots=100, voltage_plot_length=100,
                                     n_scatter_plots=1000, scatter_plot_length=1000)


class Engine:

    def __init__(self):

        numba.cuda.select_device(EngineConfig.device)

        self.app = BackendApp()

        # keep order for vbo numbers (1/3)
        self.network = SpikingNeuronNetwork(
            network_config=EngineConfig.network_config,
            plotting_config=EngineConfig.plotting_config,
            max_batch_size_mb=EngineConfig.max_batch_size_mb,
            T=EngineConfig.T)

        # keep order for vbo numbers (2/3)
        self.window = EngineWindow(name="SNN Engine",
                                   app=self.app.vs,
                                   network=self.network)

        # keep order for vbo numbers (3/3)
        self.network.initialize_GPU_arrays(EngineConfig.device)

        self.window.set_keys({
            'left': self.network.selector_box.mv_left,
            'right': self.network.selector_box.mv_right,
            'up': self.network.selector_box.mv_fw,
            'down': self.network.selector_box.mv_bw,
            'pageup': self.network.selector_box.mv_up,
            'pagedown': self.network.selector_box.mv_down,
        })

        self.time_elapsed_until_last_off = 0
        # self.set_scale(0)
        self.update_switch = False
        self.started = False
        self.timer_on = Timer('auto', connect=self.update, start=False)
        self.buttons.start.clicked.connect(self.trigger_update_switch)
        self.buttons.pause.clicked.connect(self.trigger_update_switch)
        self.actions.start.triggered.connect(self.trigger_update_switch)
        self.actions.pause.triggered.connect(self.trigger_update_switch)

        # self.buttons.cancel.clicked.connect(self.print_vbo_data)
        self.buttons.toggle_outergrid.clicked.connect(self.toggle_outergrid)
        self.actions.toggle_outergrid.triggered.connect(self.toggle_outergrid)

        self.sliders.thalamic_inh_input_current.connect(
            self.network.GPU.G_props,
            EngineConfig.network_config.DefaultValues.ThalamicInput.inh_current)

        # self.sliders.thalamic_inh_input_current.slider.valueChanged[int].connect(
        #     self.change_inh_thalamic_input_current_slider)
        # self.sliders.thalamic_inh_input_current.line_edit.returnPressed.connect(
        #     self.change_inh_thalamic_input_current_text)
        # self.sliders.thalamic_inh_input_current.value = (
        #     EngineConfig.network_config.DefaultValues.ThalamicInput.inh_current
        # )

        # self.sliders.thalamic_exc_input_current.slider.valueChanged[int].connect(
        #     self.change_exc_thalamic_input_current_slider)
        # self.sliders.thalamic_exc_input_current.line_edit.textChanged[str].connect(
        #     self.change_exc_thalamic_input_current_text)
        # self.sliders.thalamic_exc_input_current.value = (
        #     EngineConfig.network_config.DefaultValues.ThalamicInput.exc_current)

    @property
    def actions(self):
        return self.window.ui.menubar.actions

    @property
    def buttons(self):
        return self.window.ui.ui_left.buttons

    @property
    def sliders(self):
        return self.window.ui.ui_left.sliders

    def run(self):
        self.app.vs.run()

    def update(self, event):
        if self.update_switch is True:
            # elapsed = event.elapsed + self.time_elapsed_until_last_off
            # self.set_scale(elapsed)
            self.network.GPU.update()
            self.window.scene_3d.time_txt2.text = str(self.network.GPU.Simulation.t)

    def change_inh_thalamic_input_current_slider(self, value):
        print('slider:', value)
        if self.sliders.thalamic_inh_input_current.change_from_text is False:
            self.sliders.thalamic_inh_input_current.change_from_slider = True
            self.sliders.thalamic_inh_input_current.text_value = self.sliders.thalamic_inh_input_current.func(value)
            self.network.GPU.G_props.thalamic_inh_input_current = value
        else:
            self.sliders.thalamic_inh_input_current.change_from_text = False

    def change_inh_thalamic_input_current_text(self):
        value = self.sliders.thalamic_inh_input_current.text_value
        print('text:', value)
        if self.sliders.thalamic_inh_input_current.change_from_slider is False:
            self.sliders.thalamic_inh_input_current.change_from_text = True
            if value != '':
                self.sliders.thalamic_inh_input_current.set_slider_value(value)
                # self.network.GPU.G_props.thalamic_inh_input_current = self.sliders.thalamic_inh_input_current.value
                self.network.GPU.G_props.thalamic_inh_input_current = float(value)
        else:
            self.sliders.thalamic_inh_input_current.change_from_slider = False

    def change_exc_thalamic_input_current_slider(self, value):
        # print(value)
        self.sliders.thalamic_exc_input_current.text_value = value
        self.network.GPU.G_props.thalamic_exc_input_current = value

    def change_exc_thalamic_input_current_text(self, value):
        self.sliders.thalamic_exc_input_current.set_slider_value(value)
        self.network.GPU.G_props.thalamic_exc_input_current = self.sliders.thalamic_exc_input_current.value

    def toggle_outergrid(self):
        self.network.outer_grid.visible = not self.network.outer_grid.visible
        if self.network.outer_grid.visible is True:
            # self.buttons.toggle_outergrid.setText('Hide OuterGrid')
            self.buttons.toggle_outergrid.setChecked(True)
            self.actions.toggle_outergrid.setChecked(True)
        else:
            # self.buttons.toggle_outergrid.setText('Show OuterGrid')
            self.buttons.toggle_outergrid.setChecked(False)
            self.actions.toggle_outergrid.setChecked(False)

    def trigger_update_switch(self):
        self.update_switch = not self.update_switch
        if self.update_switch is True:
            self.timer_on.start()
            self.buttons.start.setDisabled(True)
            self.actions.start.setDisabled(True)
            self.buttons.pause.setDisabled(False)
            self.actions.pause.setDisabled(False)
            # self.buttons.start.setText('Pause')
        else:
            self.time_elapsed_until_last_off += self.timer_on.elapsed
            self.timer_on.stop()
            self.buttons.start.setDisabled(False)
            self.actions.start.setDisabled(False)
            self.buttons.pause.setDisabled(True)
            self.actions.pause.setDisabled(True)
            # self.buttons.start.setText('Start')

    # def print_vbo_data(self):
    #     print(vbodata2host(self.network.scatter_plot.pos_vbo))

    # def set_scale(self, elapsed):
    #     scale = [np.sin(np.pi * elapsed + np.pi/2) + 2,
    #              np.cos(np.pi * elapsed) + 2]
    #     self.window.central_scene.network_view.transform.scale = scale


if __name__ == '__main__':
    eng = Engine()
    if sys.flags.interactive != 1:
        eng.run()
