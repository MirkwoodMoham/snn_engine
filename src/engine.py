import numba.cuda
import numpy as np
from vispy.app import Timer
import sys

from network import SpikingNeuronNetwork, NetworkConfig, PlottingConfig
from ui import EngineWindow, BackendApp
from simulation import vbodata2host


class EngineConfig:

    N: int = 1 * 10 ** 3
    T: int = 2000  # Max simulation duration

    device: int = 0

    max_batch_size_mb: int = 3000

    network_config = NetworkConfig(N=N, N_pos_shape=(1, 1, 1))
    plotting_config = PlottingConfig(N=N,
                                     n_voltage_plots=10, voltage_plot_length=100,
                                     n_scatter_plots=10, scatter_plot_length=1000)


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
        self.actions.start.triggered.connect(self.trigger_update_switch)

        # self.buttons.cancel.clicked.connect(self.print_vbo_data)
        self.buttons.toggle_outergrid.clicked.connect(self.toggle_outergrid)
        self.actions.toggle_outergrid.triggered.connect(self.toggle_outergrid)

    @property
    def buttons(self):
        return self.window.ui.buttons

    @property
    def actions(self):
        return self.window.ui.menubar.actions

    def run(self):
        self.app.vs.run()

    @property
    def pos_tensor(self):
        return self.network.GPU.N_pos.tensor

    def toggle_outergrid(self):
        # d = self.network.GPU.N_pos.tensor
        # d.add_(d, )
        self.network.outer_grid.visible = not self.network.outer_grid.visible
        if self.network.outer_grid.visible is True:
            self.buttons.toggle_outergrid.setText('Hide OuterGrid')
            self.buttons.toggle_outergrid.setChecked(True)
            self.actions.toggle_outergrid.setChecked(True)
        else:
            self.buttons.toggle_outergrid.setText('Show OuterGrid')
            self.buttons.toggle_outergrid.setChecked(False)
            self.actions.toggle_outergrid.setChecked(False)

    # def print_vbo_data(self):
    #     print(vbodata2host(self.network.scatter_plot.pos_vbo))

    def trigger_update_switch(self):
        self.update_switch = not self.update_switch
        if self.update_switch is True:
            self.timer_on.start()
            self.buttons.start.setText('Pause')
        else:
            self.time_elapsed_until_last_off += self.timer_on.elapsed
            self.timer_on.stop()
            self.buttons.start.setText('Start')

    # def set_scale(self, elapsed):
    #     scale = [np.sin(np.pi * elapsed + np.pi/2) + 2,
    #              np.cos(np.pi * elapsed) + 2]
    #     self.window.central_scene.network_view.transform.scale = scale

    def update(self, event):
        if self.update_switch is True:
            # elapsed = event.elapsed + self.time_elapsed_until_last_off
            # self.set_scale(elapsed)
            self.network.GPU.update()
            self.window.central_scene.time_txt2.text = str(self.network.GPU.Simulation.t)


if __name__ == '__main__':
    eng = Engine()
    if sys.flags.interactive != 1:
        eng.run()
