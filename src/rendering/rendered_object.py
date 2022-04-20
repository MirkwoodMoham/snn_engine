from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
from vispy.scene import visuals
from vispy.gloo.context import get_current_canvas
from vispy.visuals.transforms import STTransform


class RenderedObject:

    def __init__(self):

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

    def __call__(self):
        return self._obj

    @property
    def unique_vertices_cpu(self):
        raise NotImplementedError

    @property
    def obj(self):
        return self._obj

    @property
    def transform(self) -> STTransform:
        return self._obj.transform

    @property
    def name(self):
        try:
            # noinspection PyUnresolvedReferences
            return self._obj.name
        except AttributeError:
            return str(self)

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



