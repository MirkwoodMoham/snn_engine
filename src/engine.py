import numba.cuda
from vispy import gloo
import sys

from network import SpikingNeuronNetwork, NetworkConfig, PlottingConfig
from app import (
    App,
)


class EngineConfig:

    N: int = 25 * 10 ** 3
    T: int = 2000  # Max simulation duration

    device: int = 0

    max_batch_size_mb: int = 3000

    network = NetworkConfig(N=N, N_pos_shape=(4, 4, 1))
    plotting = PlottingConfig(n_voltage_plots=100, voltage_plot_length=100,
                              n_scatter_plots=1000, scatter_plot_length=1000,
                              windowed_neuron_plots=False,
                              group_info_view_mode='split',
                              # group_info_view_mode='windowed',
                              network_config=network)


class Engine:

    def __init__(self, config=None):

        # noinspection PyUnresolvedReferences
        from pycuda import autoinit

        self.config = EngineConfig() if config is None else config

        numba.cuda.select_device(self.config.device)

        # keep order for vbo id (1/4)
        self.network = SpikingNeuronNetwork(self.config)
        # keep order for vbo id (2/4)
        self.app = App(self.network)
        # keep order for vbo id (3/4)
        self.network.initialize_GPU_arrays(self.config.device, self.app)
        # keep order (4/4)
        self.app._bind_ui()

    def run(self):
        self.app.vs.run()


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')

    eng = Engine()

    # gloo.set_state(cull_face=True, depth_test=True, blend=True)
    if sys.flags.interactive != 1:
        eng.run()
