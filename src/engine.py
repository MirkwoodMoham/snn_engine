from dataclasses import dataclass
import numba.cuda
import numpy as np
from vispy.app import Timer
import sys

from network import SpikingNeuronNetwork, NetworkConfig
from ui import EngineWindow, BackendApp
from simulation import vbodata2host


class EngineConfig:

    device: int = 0

    T: int = 2000  # Max simulation duration
    max_batch_size_mb: int = 3000

    @dataclass
    class NetworkConfig(NetworkConfig):
        N: int = 2 * 10 ** 5
        N_pos_shape: tuple = (1, 1, 1)


class Engine:

    def __init__(self):

        numba.cuda.select_device(EngineConfig.device)

        self.app = BackendApp()
        self.network = SpikingNeuronNetwork(
            config=EngineConfig.NetworkConfig(),
            max_batch_size_mb=EngineConfig.max_batch_size_mb,
            T=EngineConfig.T,
        )

        self.window = EngineWindow(name="SNN Engine",
                                   app=self.app.vs,
                                   network=self.network)

        self.network.initialize_GPU_arrays(EngineConfig.device)

        self.window.set_keys({
            'left': self.network.selector_box.mv_left,
            'right': self.network.selector_box.mv_right,
            'up': self.network.selector_box.mv_fw,
            'down': self.network.selector_box.mv_bw,
            'pageup': self.network.selector_box.mv_up,
            'pagedown': self.network.selector_box.mv_down,
        })

        self.timer_on = Timer('auto', connect=self.update, start=False)
        self.time_elapsed_until_last_off = 0
        # self.set_scale(0)

        self.update_switch = False

        self.buttons.ok.clicked.connect(self.trigger_update_switch)
        self.buttons.cancel.clicked.connect(self.print_vbo_data)
        self.buttons.toggle_outergrid.clicked.connect(self.toggle_outergrid)

    @property
    def buttons(self):
        return self.window.ui.buttons

    def run(self):
        self.app.vs.run()

    @property
    def pos_tensor(self):
        return self.network.GPU.tensor

    def toggle_outergrid(self):
        d = self.network.GPU.N_pos.tensor
        # d.add_(d, )
        self.network.outer_grid.visible = not self.network.outer_grid.visible

    def print_vbo_data(self):
        print(vbodata2host(self.network.scatter_plot.vbo))

    def trigger_update_switch(self):
        self.update_switch = not self.update_switch
        if self.update_switch is True:
            self.timer_on.start()
        else:
            self.time_elapsed_until_last_off += self.timer_on.elapsed
            self.timer_on.stop()

    def set_scale(self, elapsed):
        scale = [np.sin(np.pi * elapsed + np.pi/2) + 2,
                 np.cos(np.pi * elapsed) + 2]
        self.window.central_view.transform.scale = scale

    def update(self, event):
        if self.update_switch is True:
            elapsed = event.elapsed + self.time_elapsed_until_last_off
            self.set_scale(elapsed)


if __name__ == '__main__':
    eng = Engine()
    if sys.flags.interactive != 1:
        eng.run()
