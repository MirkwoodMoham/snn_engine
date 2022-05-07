from vispy.scene import visuals

from network.network_config import (
    NetworkConfig
)

from rendering import RenderedObjectNode


# noinspection PyAbstractClass
class RenderedNeurons(RenderedObjectNode):

    def __init__(self, config: NetworkConfig):
        # self._canvas = None

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers()
        self._obj.set_data(config.pos,
                           face_color=(1, 1, 1, .3),
                           edge_color=(0, 0.02, 0.01, .5),
                           size=7, edge_width=1)

        super().__init__(name='Neurons', subvisuals=[self._obj])

        # noinspection PyTypeChecker
        self.set_gl_state('translucent', blend=True, depth_test=True)

        # self._obj.name = 'Neurons'
        self._shape = config.N_pos_shape

    @property
    def vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id