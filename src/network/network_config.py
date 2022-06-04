# import ctypes
from dataclasses import dataclass, asdict
import numpy as np
from typing import Optional


@dataclass
class PlottingConfig:

    N: int

    n_voltage_plots: int
    voltage_plot_length: int

    n_scatter_plots: int
    scatter_plot_length: int

    _max_length: int = 10000
    _max_n_voltage_plots: int = 1000
    _max_n_scatter_plots: int = 1000

    def __post_init__(self):
        self.n_voltage_plots = min(min(self.N, self.n_voltage_plots), self._max_n_voltage_plots)
        self.n_scatter_plots = min(min(self.N, self.n_scatter_plots), self._max_n_scatter_plots)
        self.voltage_plot_length = min(self.voltage_plot_length, self._max_length)
        self.scatter_plot_length = min(self.scatter_plot_length, self._max_length)


@dataclass
class NetworkConfig:

    N: int = 100000
    S: Optional[int] = None
    D: Optional[int] = None
    G: Optional[int] = None

    pos: np.array = None
    N_pos_shape: tuple = (1, 1, 1)

    vispy_scatter_plot_stride: int = 13  # enforced by the vispy scatterplot memory layout

    grid_segmentation: tuple = None

    N_G_n_cols: int = 2
    N_G_neuron_type_col: int = 0
    N_G_group_id_col: int = 1

    sensory_groups: Optional[list[int]] = None
    output_groups: Optional[list[int]] = None

    max_z: float = 999.

    class InitValues:
        class ThalamicInput:
            inh_current: float = 25.
            exc_current: float = 15.

        class SensoryInput:
            input_current0: float = 65.
            input_current1: float = 25.

        class Weights:
            Inh2Exc: float = -.49
            Exc2Inh: float = .75
            Exc2Exc: float = .75
            SensorySource: float = .75

    def __str__(self):
        name = self.__class__.__name__
        line = '-' * len(name)
        m0 = f'\n  +--{line}--+' \
             f'\n  |  {name}  |' \
             f'\n  +--{line}--+' \
             f'\n\tN={self.N}, \n\tS={self.S}, \n\tD={self.D}, \n\tG={self.G},'
        m1 = f'\n\tN_pos_shape={self.N_pos_shape},'
        m2 = f'\n\tgrid_segmentation={self.grid_segmentation}\n'
        return m0 + m1 + m2

    def __post_init__(self):

        assert self.N % 2 == 0

        if self.N <= 4000:
            self.N_pos_shape = (1, 1, 1)

        if self.S is None:
            self.S = int(min(1000, max(np.sqrt(self.N), 2)))
        if self.D is None:
            self.D = min(int(max(np.log10(self.N) * (1 + np.sqrt(np.log10(self.N))), 2)), 20)

        # assert self.N >= 20
        # assert self.N <= 2 * 10 ** 6
        assert isinstance(self.N, int)
        assert self.S <= 1000
        assert isinstance(self.S, int)
        assert self.D <= 100
        assert isinstance(self.D, int)
        assert len(self.N_pos_shape) == 3
        # assert self.is_cube
        # assert self.shape[0] == 1
        min_shape_el = min(self.N_pos_shape)
        assert all([
            isinstance(s, int) and (s / min_shape_el == int(s / min_shape_el)) for s in self.N_pos_shape
        ])
        assert self.vispy_scatter_plot_stride == 13  # enforced by the vispy scatterplot memory layout

        self.swap_tensor_shape_multiplicators: tuple = (self.S, 10)


@dataclass
class BufferCollection:
    N_pos: int
    voltage: int
    firings: int
    selected_group_boxes_vbo: int
    selected_group_boxes_ibo: int

    def __post_init__(self):
        for k, v in asdict(self).items():
            setattr(self, k, int(v))
