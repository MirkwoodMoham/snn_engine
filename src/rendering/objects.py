import numpy as np

from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from typing import Optional, Union


# from .cuda_objects import NormalArrow

def default_cube_transform(edge_lengths):
    return STTransform(translate=[edge_lengths[0] / 2, edge_lengths[1] / 2, edge_lengths[2] / 2])


class Box(visuals.Box):

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 shape: tuple,
                 segments: tuple = (1, 1, 1),
                 translate=None,
                 scale=None,
                 color: Optional[Union[str, tuple]] = None,
                 edge_color: Union[str, tuple] = 'white',
                 name: str = None,
                 depth_test=True, border_width=1, parent=None, interactive=False,
                 use_parent_transform: bool = True):

        if translate is None:
            translate = (shape[0] / 2, shape[1] / 2, shape[2] / 2)

        # self._parent = parent
        super().__init__(width=shape[0],
                         height=shape[2],
                         depth=shape[1],
                         color=color,
                         name=name,
                         # color=(0.5, 0.5, 1, 0.5),
                         width_segments=segments[0],  # X/RED
                         height_segments=segments[2],  # Y/Blue
                         depth_segments=segments[1],  # Z/Green
                         edge_color=edge_color, parent=parent)

        self.mesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=depth_test)
        self._border.update_gl_state(line_width=max(border_width, 1))

        self.interactive = interactive

        self.unfreeze()
        if segments == (1, 1, 1):
            isv = np.unique(self._border._meshdata._vertices, axis=0)[[0, 4, 2, 1]]
            assert ((isv[1, ] - isv[0, ]) == (np.array([isv[0, 0], 0, 0]) * - 2)).all()
            assert ((isv[2, ] - isv[0, ]) == (np.array([0, isv[0, 1], 0]) * - 2)).all()
            assert ((isv[3, ] - isv[0, ]) == (np.array([0, 0, isv[0, 2]]) * - 2)).all()
            self._initial_selection_vertices = isv

        # if (parent is not None) and (use_parent_transform is True):
        #     # self.transform.scale = scale
        #     self.parent.transform.translate = translate
        #     self.parent.transform.scale = scale
        #     # self.transform = self.parent.transform
        # else:
        #     self.transform = STTransform()
        #     self.transform.scale = scale
        #     self.transform.translate = translate

        if use_parent_transform is False:
            self.transform = STTransform()
            self.transform.scale = scale
            self.transform.translate = translate