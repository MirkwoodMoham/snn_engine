from dataclasses import dataclass
import numpy as np

from geometry import grid_coordinates, validate_pos, DirectedObject
from .network_config import NetworkConfig


class GridMovements(DirectedObject):

    def __init__(self, unit_shape):
        obj = np.zeros((6, 3))
        obj[0] = np.array([unit_shape[0], 0., 0.])
        obj[1] = np.array([-unit_shape[0], 0., 0.])
        obj[2] = np.array([0., unit_shape[1], 0.])
        obj[3] = np.array([0., -unit_shape[1], 0.])
        obj[4] = np.array([0., 0., unit_shape[2]])
        obj[5] = np.array([0., 0., -unit_shape[2]])

        # struct = np.array(6, [('pos', np.float32, 3), ('coord', np.int32, 3)])

        super().__init__(obj)


class NetworkGrid:

    def __init__(self, config: NetworkConfig):

        self.config = config
        self.segmentation = self._segmentation(config.grid_segmentation)
        self.config.G = self.segmentation[0] * self.segmentation[1] * self.segmentation[2]

        self.unit_shape = (float(self.config.N_pos_shape[0] / self.segmentation[0]),
                           float(self.config.N_pos_shape[1] / self.segmentation[1]),
                           float(self.config.N_pos_shape[2] / self.segmentation[2]))
        assert self.is_cube(self.unit_shape)

        self.movements = GridMovements(self.unit_shape)

        self.pos = self._pos(config.max_z)

        self.pos_end = self.pos.copy()
        self.pos_end[:, 0] = self.pos_end[:, 0] + self.unit_shape[0]
        self.pos_end[:, 1] = self.pos_end[:, 1] + self.unit_shape[1]
        self.pos_end[:, 2] = self.pos_end[:, 2] + self.unit_shape[2]

        self.grid_coord = self.grid_coordinates(self.pos)

        self.groups = np.arange(self.config.G).reshape(tuple(reversed(self.segmentation))).T

        self.sensory_groups = config.sensory_groups
        self.output_groups = config.output_groups

        if self.sensory_groups is None:
            self.sensory_group_mask = ((self.grid_coord[:, 1] == 0)
                                       & (self.grid_coord[:, 2] == (self.segmentation[2] - 1)))[:-1]
            self.sensory_groups = self.groups.T.flatten()[self.sensory_group_mask]

        if self.output_groups is None:
            self.output_group_mask = ((self.grid_coord[:, 1] == (self.segmentation[1] - 1))
                                      & (self.grid_coord[:, 2] == (self.segmentation[2] - 1)))[:-1]
            self.output_groups = self.groups.T.flatten()[self.output_group_mask]

        n_forward_groups = int(self.segmentation[1]/2) - 1
        if self.segmentation[1] > 2:
            self.forward_groups = np.zeros((n_forward_groups + 2,
                                            len(self.sensory_groups)
                                            ),).astype(np.int32)

            coord = self.sensory_grid_coord.T

            for i in range(self.forward_groups.shape[0]):
                self.forward_groups[i, :] = np.ravel_multi_index(coord, self.segmentation, order='F')
                coord[1] += 1
                # self.forward_groups[i, :, 1] = np.ravel_multi_index(coord, self.segmentation, order='F')
        else:
            self.forward_groups = None

        config.sensory_groups = self.sensory_groups
        config.output_groups = self.output_groups

    def grid_coordinates(self, pos, as_struct: bool = False):
        return grid_coordinates(pos, outer_shape=self.config.N_pos_shape, grid_segmentation=self.segmentation,
                                as_struct=as_struct)

    @staticmethod
    def is_cube(shape):
        return (shape[0] == shape[1]) and (shape[0] == shape[2])

    def _segmentation(self, grid_segmentation):
        if grid_segmentation is None:
            segmentation_list = []
            for s in self.config.N_pos_shape:
                f = max(self.config.N_pos_shape) / min(self.config.N_pos_shape)
                segmentation_list.append(
                    int(int(max(self.config.D / (np.sqrt(3) * f), 2)) * (s / min(self.config.N_pos_shape))))
            grid_segmentation = tuple(segmentation_list)
        min_g_shape = min(grid_segmentation)
        assert all([isinstance(s, int)
                    and (s / min_g_shape == int(s / min_g_shape)) for s in grid_segmentation])
        self.config.grid_segmentation = grid_segmentation
        return grid_segmentation

    def _pos(self, max_z):
        groups = np.arange(self.config.G)
        z = np.floor(groups / (self.segmentation[0] * self.segmentation[1]))
        r = groups - z * (self.segmentation[0] * self.segmentation[1])
        y = np.floor(r / self.segmentation[0])
        x = r - y * self.segmentation[0]
        g_pos = np.zeros((self.config.G + 1, 3), dtype=np.float32)

        # The last entry will be ignored by the geometry shader (i.e. invisible).
        # We could also use a primitive restart index instead.
        # The current solution is simpler w.r.t. vispy.
        g_pos[:, 2] = max_z + 1

        g_pos[:self.config.G, 0] = x * self.unit_shape[0]
        g_pos[:self.config.G, 1] = y * self.unit_shape[1]
        g_pos[:self.config.G, 2] = z * self.unit_shape[2]

        assert np.max(g_pos[:self.config.G, 2]) < max_z
        validate_pos(g_pos[:self.config.G, :], self.segmentation)
        # noinspection PyAttributeOutsideInit
        return g_pos

    @property
    def sensory_grid_coord(self):
        return self.grid_coord[self.sensory_groups]

    @property
    def output_grid_coord(self):
        return self.grid_coord[self.output_groups]
