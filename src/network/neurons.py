import numpy as np

from vispy.scene import visuals

from rendering import RenderedObjectNode
from geometry import grid_coordinates, validate_pos
from .network_config import NetworkConfig
from .network_grid import NetworkGrid


# noinspection PyAbstractClass
class Neurons(RenderedObjectNode):

    def __init__(self, config: NetworkConfig, grid_segmentation: NetworkGrid, type_groups):
        # self._canvas = None
        self._shape = config.N_pos_shape
        if config.pos is None:
            config.pos = (np.random.rand(config.N, 3).astype(np.float32) * np.array(self._shape, dtype=np.float32))
            config.pos[config.pos == max(self._shape)] = config.pos[config.pos == max(self._shape)] * 0.999999
            validate_pos(config.pos, self._shape)

        self.pos = config.pos
        self.grid_coord = grid_coordinates(self.pos, self._shape, grid_segmentation)
        self.sort_pos(type_groups=type_groups)

        assert self.pos.shape[1] == 3
        assert len(self.pos.shape) == 2
        assert np.max(self.pos[:, 2]) < config.max_z

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers()
        self._obj.set_data(self.pos,
                           face_color=(1, 1, 1, .3),
                           edge_color=(0, 0.02, 0.01, .5),
                           size=7, edge_width=1)

        super().__init__(name='Neurons', subvisuals=[self._obj])

        # noinspection PyTypeChecker
        self.set_gl_state('translucent', blend=True, depth_test=True)

        # self._obj.name = 'Neurons'

    def sort_pos(self, type_groups):
        """
        Sort neuron positions w.r.t. location-based groups and neuron types.
        """

        for g in type_groups:

            grid_pos = self.grid_coord[g.start_idx: g.end_idx + 1]

            p0 = grid_pos[:, 0].argsort(kind='stable')
            p1 = grid_pos[p0][:, 1].argsort(kind='stable')
            p2 = grid_pos[p0][p1][:, 2].argsort(kind='stable')

            self.grid_coord[g.start_idx: g.end_idx + 1] = grid_pos[p0][p1][p2]
            self.pos[g.start_idx: g.end_idx + 1] = \
                self.pos[g.start_idx: g.end_idx + 1][p0][p1][p2]
            if len(self.pos) <= 100:
                print('\n', self.pos[g.start_idx:g.end_idx+1])
        if len(self.pos) <= 100:
            print()

    @property
    def vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id
