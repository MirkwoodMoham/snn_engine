from vispy.scene.visuals import Line
from typing import Optional

from gpu import RegisteredVBO
from .rendered_cuda_object import CudaObject
from .rendered_object import get_buffer_id


class CudaLine(Line, CudaObject):

    def __init__(self, pos=None, color=(0.5, 0.5, 0.5, 1), width=1,
                 connect='strip', method='gl', antialias=False, parent=None):

        Line.__init__(self, pos=pos, color=color, width=width, connect=connect, method=method, antialias=antialias,
                      parent=parent)
        CudaObject.__init__(self)

        self.unfreeze()
        self.pos_gpu: Optional[RegisteredVBO] = None
        self.colors_gpu: Optional[RegisteredVBO] = None
        self.freeze()

    @property
    def pos_vbo(self):
        return get_buffer_id(self.pos_vbo_glir_id)

    @property
    def color_vbo(self):
        return get_buffer_id(self.color_vbo_glir_id)

    @property
    def pos_vbo_glir_id(self):
        return self._line_visual._pos_vbo.id

    @property
    def color_vbo_glir_id(self):
        return self._line_visual._color_vbo.id

    def init_cuda_arrays(self):
        self.pos_gpu = RegisteredVBO(self.pos_vbo, shape=self.pos.shape, device=self._cuda_device)
        self.colors_gpu = RegisteredVBO(self.color_vbo, shape=self.color.shape, device=self._cuda_device)

        self.registered_buffers.append(self.pos_gpu)
        self.registered_buffers.append(self.colors_gpu)
