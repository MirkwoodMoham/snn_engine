import ctypes
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional


@dataclass
class NetworkConfig:

    N: int = 100000
    S: Optional[int] = None
    D: Optional[int] = None
    G: Optional[int] = None

    pos: np.array = None
    N_pos_shape: tuple = (1, 1, 1)

    N_pos_n_cols: int = 13  # enforced by the vispy scatterplot memory layout

    G_shape: tuple = None

    grid_unit_shape: tuple = None

    N_G_n_cols: int = 2
    N_G_neuron_type_col: int = 0
    N_G_group_id_col: int = 1

    n_group_properties: int = 10

    def __str__(self):
        m0 = f'NetworkConfig:\n\tN={self.N}, \n\tS={self.S}, \n\tD={self.D}, \n\tG={self.G},'
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
        assert self.N_pos_n_cols == 13  # enforced by the vispy scatterplot memory layout

        if self.pos is None:
            self.pos = (np.random.rand(self.N, 3).astype(np.float32) * np.array(self.N_pos_shape, dtype=np.float32))
            self.pos[self.pos == max(self.N_pos_shape)] = self.pos[self.pos == max(self.N_pos_shape)] * 0.999999
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

        self.grid_pos = (np.floor((self.pos
                                   / np.array(self.N_pos_shape, dtype=np.float32))
                                  * np.array(min_g_shape, dtype=np.float32)).astype(int))
        self.grid_unit_shape = (self.N_pos_shape[0] / self.G_shape[0],
                                self.N_pos_shape[1] / self.G_shape[1],
                                self.N_pos_shape[2] / self.G_shape[2])

        assert self.is_cube(self.grid_unit_shape)
        assert self.pos.shape[1] == 3
        assert len(self.pos.shape) == 2

        print('\n', self)
        print()


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


