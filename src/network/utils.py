import ctypes
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional


class ExternalMemory(object):
    """
    Provide an externally managed memory.
    Interface requirement: __cuda_memory__, device_ctypes_pointer, _cuda_memsize_
    """
    __cuda_memory__ = True

    def __init__(self, ptr, size):
        self.device_ctypes_pointer = ctypes.c_void_p(ptr)
        self._cuda_memsize_ = size


@dataclass
class DefaultNetworkConfig:

    N: int = 100000
    S: Optional[int] = None
    D: Optional[int] = None

    pos: np.array = None
    shape: tuple = (1, 1, 1)

    def __repr__(self):
        return f'NetworkConfig(N={self.N}, S={self.S}, D={self.D}), G={self.G}'

    @property
    def is_cube(self):
        return (self.shape[0] == self.shape[1]) and (self.shape[2] == self.shape[1])

    def __post_init__(self):

        if self.S is None:
            self.S = int(min(1000, max(np.sqrt(self.N), 2)))
        if self.D is None:
            self.D = int(max(np.log10(self.N) * (1 + np.sqrt(np.log10(self.N))), 2))

        assert self.N >= 20
        assert isinstance(self.N, int)
        assert self.S <= 1000
        assert isinstance(self.S, int)
        assert self.D <= 100
        assert isinstance(self.D, int)
        assert len(self.shape) == 3
        # assert self.is_cube
        # assert self.shape[0] == 1
        min_shape_el = min(self.shape)
        assert all([
            isinstance(s, int) and (s / min_shape_el == int(s / min_shape_el)) for s in self.shape
        ])

        if self.pos is None:
            self.pos = (np.random.rand(self.N, 3).astype(np.float32)
                        * np.array(self.shape).astype(np.float32)).round(5)

        G_shape_list = []
        min_length = int(max(self.D / np.sqrt(3), 2))
        for s in self.shape:
            G_shape_list.append(int(min_length * (s / min_shape_el)))

        # lg_shape_x = lg_shape_y = lg_shape_z = int(max(self.D/np.sqrt(3), 2))

        self.G_shape = tuple(G_shape_list)

        self.G = self.G_shape[0] * self.G_shape[1] * self.G_shape[2]

        self.grid_unit_shape = (self.shape[0] / self.G_shape[0],
                                self.shape[1] / self.G_shape[1],
                                self.shape[2] / self.G_shape[2])

        assert self.pos.shape[1] == 3
        assert len(self.pos.shape) == 2


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


