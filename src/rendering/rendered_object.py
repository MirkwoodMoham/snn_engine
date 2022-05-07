# from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from typing import Optional, Union

import pandas as pd
from vispy.visuals import CompoundVisual, Visual, BaseVisual
from vispy.scene import visuals, Node
from vispy.gloo.context import get_current_canvas
from vispy.visuals.transforms import STTransform


class RenderedObject:
    def __init__(self, select_parent=None, selectable=False, draggable=False):

        if not hasattr(self, '_visual'):
            self._visual = None

        if select_parent is not None:
            self.select_parent = select_parent
        if not hasattr(self, '_select_children'):
            self._select_children = []
        if not hasattr(self, '_select_parent'):
            self._select_parent = None
        if not hasattr(self, 'original_color'):
            self.original_color = None
        if not hasattr(self, '_shape'):
            self._shape = None

        self._vbo = None
        self._pos_vbo = None
        self._color_vbo = None
        self._ibo = None
        # self._parent = None
        self._glir = None

        self.transform_connected = False

        self.selectable = selectable
        self.draggable = draggable
        self.selected = False
        self.select_color = 'white'
        self._color = None
        self.color = self.original_color

        self._cuda_device: Optional[str] = None
        self.scale: Optional[Scale] = None
        self._transform = None

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, v):
        self._color = v

    def swap_select_color(self, v):
        if v is True:
            self.color = self.select_color
        else:
            self.color = self.original_color

    @property
    def select_parent(self):
        return self._select_parent

    @select_parent.setter
    def select_parent(self, v):
        self._select_parent = v
        if v is not None:
            v._select_children.append(self)

    def is_select_child(self, v):
        return v in self._select_children

    @property
    def unique_vertices_cpu(self):
        raise NotImplementedError

    @property
    def visual(self):
        return self._visual

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
        raise NotImplementedError

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

    def on_drag_callback(self, v: bool, mode: int):
        raise NotImplementedError

    def select(self, v):
        if self.selectable is True:
            self.selected = v
            self.on_select_callback(v)

    def update(self):
        self.visual.update()


# noinspection PyAbstractClass
class RenderedObjectVisual(CompoundVisual, RenderedObject):

    def __init__(self, subvisuals, parent=None, selectable=False, draggable=False):

        self.unfreeze()
        RenderedObject.__init__(self, selectable=selectable, draggable=draggable)
        CompoundVisual.__init__(self, subvisuals)
        self.freeze()

        if parent is not None:
            self.parent = parent


def add_children(parent: Node, children: list):
    for child in children:
        parent._add_child(child)


# RenderedObjectNode = visuals.create_visual_node(RenderedObjectVisual)


# noinspection PyAbstractClass
class RenderedObjectNode(visuals.VisualNode, RenderedObjectVisual):

    clsname = RenderedObjectVisual.__name__
    if not (clsname.endswith('Visual') and
            issubclass(RenderedObjectVisual, BaseVisual)):
        raise RuntimeError('Class "%s" must end with Visual, and must '
                           'subclass BaseVisual' % clsname)
    clsname = clsname[:-6]
    # noinspection PyBroadException
    try:
        __doc__ = visuals.generate_docstring(RenderedObjectVisual, clsname)
    except Exception:
        __doc__ = RenderedObjectVisual.__doc__

    def __init__(self, *args, **kwargs):
        parent = kwargs.pop('parent', None)
        name = kwargs.pop('name', None)
        if not hasattr(self, 'name'):
            self.name = name  # to allow __str__ before Node.__init__
        self._visual_superclass = RenderedObjectVisual
        RenderedObjectVisual.__init__(self, *args, **kwargs)
        self.unfreeze()
        visuals.VisualNode.__init__(self, parent=parent, name=self.name)
        self.freeze()


@dataclass
class XTZ:
    x: object = None
    y: object = None
    z: object = None

    def __post_init__(self):
        self._tuple = (self.x, self.y, self.z)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._tuple[item]
        else:
            return getattr(self, item)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key != '_tuple':
            super().__setattr__('_tuple', (self.x, self.y, self.z))


@dataclass
class _STR:

    parent: RenderedObjectNode
    prop_id: str = 'some key'

    spin_box_sliders: Optional[XTZ] = None

    value_ranges: Optional[XTZ] = None

    _min_value: Optional[Union[int, float]] = None
    _max_value: Optional[Union[int, float]] = None

    def __call__(self):
        return getattr(self.parent.transform, self.prop_id)

    # noinspection PyArgumentList
    def __post_init__(self):
        if self.spin_box_sliders is None:
            self.spin_box_sliders = XTZ()

        if (self._min_value is not None) or (self._max_value is not None):
            if self.value_ranges is not None:
                raise ValueError('multiple values for self.value_intervals.')
            self.value_ranges = XTZ(
                x=pd.Interval(self._min_value, self._max_value, closed='both'),
                y=pd.Interval(self._min_value, self._max_value, closed='both'),
                z=pd.Interval(self._min_value, self._max_value, closed='both'))

    def change_prop(self, i, v):
        if self.value_ranges is not None:
            interval = self.value_ranges[i]
            v = min(interval.right, max(interval.left, v))
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

    min_value: Optional[int] = 0
    max_value: Optional[int] = 10


@dataclass
class Translate(_STR):
    _grid_unit_shape: Optional[tuple] = (1, 1, 1)
    prop_id: str = 'translate'

    min_value: Optional[int] = -5
    max_value: Optional[int] = 5

    def __post_init__(self):
        super().__post_init__()
        self._grid_coordinates = np.zeros(3)

    def _move(self, i, d=1):
        tr = np.zeros(3)
        tr[i] += d * self._grid_unit_shape[i]

        self.transform.move(tr)
        self._grid_coordinates[i] += 1 * d

        if self.parent.transform_connected is True:
            self.parent.transform_changed()

        self.spin_box_sliders[i].actualize_values()

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



