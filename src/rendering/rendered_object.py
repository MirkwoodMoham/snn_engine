# from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from typing import Optional, Union

from vispy.visuals import CompoundVisual, Visual
from vispy.scene import visuals, Node
from vispy.gloo.context import get_current_canvas
from vispy.visuals.transforms import STTransform


class RenderedObject:
    def __init__(self, selectable=False):

        if not hasattr(self, '_obj'):
            self._obj = None
        if not hasattr(self, '_select_children'):
            self._select_children = []

        self._vbo = None
        self._pos_vbo = None
        self._color_vbo = None
        self._ibo = None
        # self._parent = None

        self._shape = None

        self._glir = None

        self.transform_connected = True

        self.selectable = selectable
        self.selected = False

        self._cuda_device: Optional[str] = None
        self.scale: Optional[Scale] = None
        self._transform = None

        self._select_parent = None

    @property
    def select_parent(self):
        return self._select_parent

    @select_parent.setter
    def select_parent(self, v):
        self._select_parent = v
        v._select_children.append(self)

    def is_select_child(self, v):
        return v in self._select_children

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
    def color_vbo_glir_id(self):
        return self.vbo_glir_id

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

    @staticmethod
    def buffer_id(glir_id):
        return int(get_current_canvas().context.shared.parser._objects[glir_id].handle)

    @property
    def color_vbo(self):
        # print(self.buffer_id(self.color_vbo_glir_id))
        # return self.buffer_id(self.color_vbo_glir_id)
        if self._color_vbo is None:
            self._color_vbo = self.buffer_id(self.color_vbo_glir_id)
        return self._color_vbo

    @property
    def pos_vbo(self):
        if self._pos_vbo is None:
            self._pos_vbo = self.buffer_id(self.pos_vbo_glir_id)
        return self._pos_vbo

    @property
    def vbo(self):
        if self._vbo is None:
            self._vbo = self.buffer_id(self.vbo_glir_id)
        return self._vbo

    @property
    def ibo(self):
        if self._ibo is None:
            self._ibo = self.buffer_id(self.ibo_glir_id)
        return self._ibo

    def on_select_callback(self, v: bool):
        raise NotImplementedError

    def select(self, v):
        if self.selectable is True:
            self.selected = v
            self.on_select_callback(v)


class RenderedObjectVisual(CompoundVisual, RenderedObject):

    # def __init__(self, parent=None, name=None, transforms=None, selectable=False):
    def __init__(self, subvisuals, parent=None, selectable=False):

        # super().__init__(parent=parent, name=name, transforms=transforms)

        self.unfreeze()
        # self._obj: Optional[Union[visuals.visuals.MarkersVisual,
        #                           visuals.visuals.LineVisual]] = None
        RenderedObject.__init__(self, selectable=selectable)

        CompoundVisual.__init__(self, subvisuals)
        # for v in subvisuals:
        #     v.parent = self

        self.freeze()
        # self.unfreeze()

        if parent is not None:
            self.parent = parent

        # self.set_transform('st')

        # self._transform = STTransform()

def add_children(parent: Node, children: list):
    for child in children:
        parent._add_child(child)


RenderedObjectNode = visuals.create_visual_node(RenderedObjectVisual)


@dataclass
class _STR:
    parent: RenderedObjectNode
    prop_id: str = 'some key'

    def change_prop(self, i, v):
        p = getattr(self.transform, self.prop_id)
        p[i] = v
        setattr(self.transform, self.prop_id, p)
        if self.parent.transform_connected is True:
            self.parent.transform_changed()

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



