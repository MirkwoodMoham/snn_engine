import numba.cuda
from vispy import gloo
import sys

from network import (
    NetworkConfig,
    PlottingConfig,
    SpikingNeuronNetwork
)
from app import BaseApp


# TODO: group_info_mesh face sizes
# TODO: better stdp G2G config
# TODO: monitor learning

# TODO: overflow? (N=25000, t=2106)

# TODO (optional) gpu side group_info_mesh face color actualization
# TODO (optional) group selection via selector box
# TODO (optional, difficult) group selection via click

class EngineConfig:

    N: int = 25 * 10 ** 3
    T: int = 2000  # Max simulation duration

    device: int = 0

    max_batch_size_mb: int = 3000

    network = NetworkConfig(N=N, N_pos_shape=(4, 4, 1))
    plotting = PlottingConfig(n_voltage_plots=100, voltage_plot_length=100,
                              n_scatter_plots=1000, scatter_plot_length=1000,
                              windowed_neuron_plots=True,
                              group_info_view_mode='split',
                              network_config=network)


class Engine(BaseApp):

    def __init__(self, config=None):

        # noinspection PyUnresolvedReferences
        from pycuda import autoinit

        self.config = EngineConfig() if config is None else config

        numba.cuda.select_device(self.config.device)

        # keep order for vbo id (1/4)
        network = SpikingNeuronNetwork(self.config)
        # keep order for vbo id (2/4)
        super().__init__(network)
        # keep order for vbo id (3/4)
        self.network.initialize_GPU_arrays(self.config.device, self)
        # keep order (4/4)
        self._bind_ui()


if __name__ == '__main__':

    gloo.gl.use_gl('gl+')
    eng = Engine()
    if sys.flags.interactive != 1:
        eng.run()
