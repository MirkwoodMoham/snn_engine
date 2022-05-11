from typing import Optional, Union

from geometry import initial_normal_vertices
from .cuda_box_arrows import InteractiveBoxNormals
from .rendered_cuda_object import CudaObject
from .box import Box


class CudaBox(Box, CudaObject):

    def __init__(self,
                 select_parent,
                 shape: tuple,
                 segments: tuple = (1, 1, 1),
                 translate=None,
                 scale=None,
                 color: Optional[Union[str, tuple]] = None,
                 edge_color: Union[str, tuple] = 'white',
                 name: str = None,
                 depth_test=True, border_width=1, parent=None,
                 init_normals=True):

        Box.__init__(self, shape=shape,
                     segments=segments,
                     scale=scale,
                     translate=translate,
                     name=name,
                     color=color,
                     edge_color=edge_color,
                     depth_test=depth_test,
                     border_width=border_width,
                     parent=parent)

        if init_normals:
            assert segments == (1, 1, 1)
            self.normals = InteractiveBoxNormals(select_parent, shape)

        CudaObject.__init__(self)
