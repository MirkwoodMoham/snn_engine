# from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from typing import Optional, Union

from vispy.scene import visuals, Node
from vispy.gloo.context import get_current_canvas
from vispy.visuals.transforms import STTransform


class RenderedObject(Node):

    def __init__(self, parent=None, name=None, transforms=None, selectable=False):

        super().__init__(parent=parent, name=name, transforms=transforms)

        self._obj: Optional[Union[visuals.visuals.MarkersVisual,
                                  visuals.visuals.LineVisual]] = None

        self.scale: Optional[Scale] = None

        self._vbo = None
        self._pos_vbo = None
        self._ibo = None
        self._parent = None

        self._shape = None

        self._glir = None

        self.transform_connected = True

        self.selectable = selectable
        self.selected = False

        self.set_transform('st')
        # self._transform = STTransform()

    def __call__(self):
        return self._obj

    @property
    def unique_vertices_cpu(self):
        raise NotImplementedError

    @property
    def obj(self):
        return self._obj

    # @property
    # def transform(self) -> STTransform:
    #     return self._obj.transform

    # @property
    # def name(self):
    #     try:
    #         # noinspection PyUnresolvedReferences
    #         return self._obj.name
    #     except AttributeError:
    #         return str(self)

    @property
    def glir(self):
        if self._glir is None:
            self._glir = get_current_canvas().context.glir
        return self._glir

    @property
    def shape(self):
        return self._shape

    @property
    def pos_vbo_glir_id(self):
        return self.vbo_glir_id

    @property
    def ibo_glir_id(self):
        raise NotImplementedError

    @property
    def vbo_glir_id(self):
        raise NotImplementedError

    def transform_changed(self):
        pass

    # noinspection PyProtectedMember
    @property
    def pos_vbo(self):
        if self._pos_vbo is None:
            self._pos_vbo = get_current_canvas().context.shared.parser._objects[self.pos_vbo_glir_id].handle
        return self._pos_vbo

    @property
    def vbo(self):
        if self._vbo is None:
            self._vbo = get_current_canvas().context.shared.parser._objects[self.vbo_glir_id].handle
        return self._vbo

    @property
    def ibo(self):
        if self._ibo is None:
            self._ibo = get_current_canvas().context.shared.parser._objects[self.ibo_glir_id].handle
        return self._ibo

    def on_select_callback(self, v: bool):
        raise NotImplementedError

    def select(self, v):
        if self.selectable is True:
            self.selected = v
            self.on_select_callback(v)


class NormalArrowVisual(visuals.Tube):

    def __init__(self, points, color=None, name=None, parent=None,
                 tube_points=4, radius=np.array([.01, .01, .025, .0])):
        if color is None:
            if points[:, 0].any():
                color = np.array([1, 0, 0, 0.3])
            elif points[:, 1].any():
                color = np.array([0, 1, 0, 0.3])
            else:
                color = np.array([0, 0, 1, 0.3])
        name = name or parent.name
        if (points[:, 0] > 0).any():
            name += ':x+'
        elif (points[:, 0] < 0).any():
            name += ':x-'
        elif (points[:, 1] > 0).any():
            name += ':y+'
        elif (points[:, 1] < 0).any():
            name += ':y-'
        elif (points[:, 2] > 0).any():
            name += ':z+'
        else:
            name += ':z-'

        super().__init__(name=name, points=points, tube_points=tube_points, radius=radius, color=color, parent=parent)
        # self.name = name
        # self.transform: STTransform = parent.transform
        self.interactive = True


        # self.unfreeze()
        # self._obj_transform: STTransform = transform
        # self._transform: STTransform = deepcopy(transform)
        # self._fixed_scale = np.array([1., 1., 1., 1.])
        # self.freeze()

    # @property
    # def transform(self):
    #
    #     self._transform.translate = self._obj_transform.translate
    #     self._transform.scale = self._fixed_scale
    #
    #     return self._transform


class NormalArrow(RenderedObject):

    def __init__(self, points, color=None, name=None, tube_points=4,
                 radius=np.array([.01, .01, .025, .0]), parent=None, selectable=True):
        super().__init__(parent=parent, selectable=selectable,
                         name=name or parent.name + f'.{self.__class__.__name__}')
        self.transform: STTransform = parent.parent.transform

        self._obj = NormalArrowVisual(points=points, name=name, parent=self,
                                      tube_points=tube_points, radius=radius, color=color)

    def on_select_callback(self, v):
        print(f'selected arrow({v}):', self)



@dataclass
class _STR:
    parent: RenderedObject
    prop_id: str = 'some key'

    def change_prop(self, i, v):
        sc_old = getattr(self.transform, self.prop_id)
        sc_new = sc_old.copy()
        sc_new[i] = v
        setattr(self.transform, self.prop_id, sc_new)
        if self.parent.transform_connected is True:
            self.parent.transform_changed()
        # print()
        # print(self.parent.unique_vertices_cpu)

    @property
    def transform(self) -> STTransform:
        return self.parent.transform

    @property
    def x(self):
        return getattr(self.transform, self.prop_id)[0]

    @x.setter
    def x(self, v):
        self.change_prop(0, v)

    @property
    def y(self):
        return getattr(self.transform, self.prop_id)[1]

    @y.setter
    def y(self, v):
        self.change_prop(1, v)

    @property
    def z(self):
        return getattr(self.transform, self.prop_id)[2]

    @z.setter
    def z(self, v):
        self.change_prop(2, v)

    @property
    def a(self):
        return getattr(self.transform, self.prop_id)[3]

    @a.setter
    def a(self, v):
        self.change_prop(3, v)


@dataclass
class Scale(_STR):
    prop_id: str = 'scale'


@dataclass
class Position(_STR):
    _grid_unit_shape: Optional[tuple] = (1, 1, 1)
    prop_id: str = 'translate'

    def __post_init__(self):
        self._grid_coordinates = np.zeros(3)

    def _move(self, i, d=1):
        tr = np.zeros(3)
        tr[i] += d * self._grid_unit_shape[i]

        self.transform.move(tr)
        self._grid_coordinates[i] += 1 * d

        if self.parent.transform_connected is True:
            self.parent.transform_changed()

    def mv_left(self):
        self._move(0)

    def mv_right(self):
        self._move(0, -1)

    def mv_fw(self):
        self._move(1, -1)

    def mv_bw(self):
        self._move(1)

    def mv_up(self):
        self._move(2)

    def mv_down(self):
        self._move(2, -1)



