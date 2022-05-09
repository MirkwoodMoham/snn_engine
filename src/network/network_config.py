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

    G_shape: tuple = None

    grid_unit_shape: tuple = None

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
            Exc2Inh: float = .5
            Exc2Exc: float = .51
            SensorySource: float = .52

    def __str__(self):
        name = self.__class__.__name__
        line = '-' * len(name)
        m0 = f'\n  +--{line}--+' \
             f'\n  |  {name}  |' \
             f'\n  +--{line}--+' \
             f'\n\tN={self.N}, \n\tS={self.S}, \n\tD={self.D}, \n\tG={self.G},'
        m1 = f'\n\tN_pos_shape={self.N_pos_shape},'
        m2 = f'\n\tG_shape={self.G_shape}\n'
        return m0 + m1 + m2

    @staticmethod
    def is_cube(shape):
        return (shape[0] == shape[1]) and (shape[0] == shape[2])

    def __post_init__(self):

        if self.S is None:
            self.S = int(min(1000, max(np.sqrt(self.N), 2)))
        if self.D is None:
            self.D = min(int(max(np.log10(self.N) * (1 + np.sqrt(np.log10(self.N))), 2)), 20)

        # assert self.N >= 20
        assert self.N <= 2 * 10 ** 6
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

        if self.pos is None:
            self.pos = (np.random.rand(self.N, 3).astype(np.float32) * np.array(self.N_pos_shape, dtype=np.float32))
            self.pos[self.pos == max(self.N_pos_shape)] = self.pos[self.pos == max(self.N_pos_shape)] * 0.999999
            self.validate_pos(self.pos)
        # noinspection PyPep8Naming
        G_shape_list = []
        for s in self.N_pos_shape:
            f = max(self.N_pos_shape) / min(self.N_pos_shape)
            G_shape_list.append(
                int(int(max(self.D / (np.sqrt(3) * f), 2)) * (s / min(self.N_pos_shape))))

        self.G_shape = tuple(G_shape_list)
        self.G = self.G_shape[0] * self.G_shape[1] * self.G_shape[2]

        min_g_shape = min(self.G_shape)

        assert all([isinstance(s, int)
                    and (s / min_g_shape == int(s / min_g_shape)) for s in self.G_shape])

        self.N_grid_pos = (np.floor((self.pos
                                     / np.array(self.N_pos_shape, dtype=np.float32))
                                    * np.array(self.G_shape, dtype=np.float32)).astype(int))

        self.grid_unit_shape = (float(self.N_pos_shape[0] / self.G_shape[0]),
                                float(self.N_pos_shape[1] / self.G_shape[1]),
                                float(self.N_pos_shape[2] / self.G_shape[2]))

        assert self.is_cube(self.grid_unit_shape)
        assert self.pos.shape[1] == 3
        assert len(self.pos.shape) == 2

        print('\n', self, '\n')

        assert np.max(self.pos[:, 2]) < self.max_z

        self._g_pos = self.init_g_pos()
        self.G_grid_pos = (np.floor((self.g_pos
                                     / np.array(self.N_pos_shape, dtype=np.float32))
                                    * np.array(self.G_shape, dtype=np.float32)).astype(int))
        self._g_pos_end = None

        groups = np.arange(self.G)
        if self.sensory_groups is None:
            # self.sensory_group_mask = (self.G_grid_pos[:, 1] == self.G_shape[1] - 1)[:-1]
            self.sensory_group_mask = (self.G_grid_pos[:, 1] == 0)[:-1]
            self.sensory_groups = groups[self.sensory_group_mask]

        if self.output_groups is None:
            self.output_group_mask = ((self.G_grid_pos[:, 1] == self.G_shape[1] - 1)
                                      & (self.G_grid_pos[:, 2] == self.G_shape[2] - 1))[:-1]
            self.output_groups = groups[self.output_group_mask]

    def init_g_pos(self):
        groups = np.arange(self.G)
        z = np.floor(groups / (self.G_shape[0] * self.G_shape[1]))
        r = groups - z * (self.G_shape[0] * self.G_shape[1])
        y = np.floor(r / self.G_shape[0])
        x = r - y * self.G_shape[0]
        g_pos = np.zeros((self.G + 1, 3), dtype=np.float32)

        # The last entry will be ignored by the geometry shader (i.e. invisible).
        # We could also use a primitive restart index instead.
        # The current solution is simpler w.r.t. vispy.
        g_pos[:, 2] = self.max_z + 1

        g_pos[:self.G, 0] = x * self.grid_unit_shape[0]
        g_pos[:self.G, 1] = y * self.grid_unit_shape[1]
        g_pos[:self.G, 2] = z * self.grid_unit_shape[2]

        assert np.max(g_pos[:self.G, 2]) < self.max_z
        self.validate_pos(g_pos[:self.G, :])
        # noinspection PyAttributeOutsideInit
        return g_pos

    @property
    def g_pos(self):
        return self._g_pos

    @property
    def g_pos_end(self):
        if self._g_pos_end is None:
            # noinspection PyAttributeOutsideInit
            self._g_pos_end = self.g_pos.copy()
            self._g_pos_end[:, 0] = self._g_pos_end[:, 0] + self.grid_unit_shape[0]
            self._g_pos_end[:, 1] = self.g_pos_end[:, 1] + self.grid_unit_shape[1]
            self._g_pos_end[:, 2] = self.g_pos_end[:, 2] + self.grid_unit_shape[2]
        return self._g_pos_end

    @property
    def sensory_grid_pos(self):
        return self.G_grid_pos[self.sensory_groups]

    @property
    def output_grid_pos(self):
        return self.G_grid_pos[self.output_groups]

    def validate_pos(self, pos):
        for i in range(3):
            assert np.min(pos[:, i]) >= 0
            assert np.max(pos[:, i]) <= self.N_pos_shape[i]


def pos_cloud(size=100000):

    pos = np.random.normal(size=(size, 3), scale=0.2)
    # one could stop here for the data generation, the rest is just to make the
    # data look more interesting. Copied over from magnify.py
    centers = np.random.normal(size=(50, 3))
    indexes = np.random.normal(size=size, loc=centers.shape[0] / 2.,
                               scale=centers.shape[0] / 3.)
    indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)
    scales = 10 ** (np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
    pos *= scales
    pos += centers[indexes]

    return pos


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
