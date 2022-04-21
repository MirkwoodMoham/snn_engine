import numpy as np
import torch
from typing import Optional, Union
from vispy import gloo
from vispy.color import Color
from vispy.visuals.shaders.function import Function
from vispy.scene import visuals, VisualNode
from vispy.visuals import Visual
from vispy.visuals.line.line import _GLLineVisual, _AggLineVisual, LineVisual
from vispy.visuals.transforms import STTransform
from vispy.util.profiler import Profiler

from network.network_config import NetworkConfig
from network.network_states import LocationGroupProperties
from rendering import (
    NormalArrow,
    Position,
    RenderedObject,
    Scale,
)


def default_cube_transform(edge_lengths):
    return STTransform(translate=[edge_lengths[0] / 2, edge_lengths[1] / 2, edge_lengths[2] / 2])


class DefaultBox(visuals.Box):

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 shape: tuple,
                 segments: tuple = (1, 1, 1),
                 translate=None,
                 scale=None,
                 color: Optional[Union[str, tuple]] = None,
                 edge_color: Union[str, tuple] = 'white',
                 name: str = None,
                 depth_test=True, border_width=1, init_normals=False, parent=None, interactive=False):
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

        if parent is not None:
            # self.transform.scale = scale
            self.parent.transform.translate = translate
            self.parent.transform.scale = scale
            # self.transform = self.parent.transform
        else:
            self.transform = STTransform()
            self.transform.scale = scale
            self.transform.translate = translate

        if init_normals:
            assert segments == (1, 1, 1)
            self.normals = []
            inv = self.initial_normal_vertices(shape)
            for i in range(6):
                self.normals.append(NormalArrow(points=inv[i], parent=self))

        # meshdata = self.mesh.mesh_data
        # self.vertex_normals = visuals.MeshNormals(meshdata, primitive='vertex', color='orange', width=2)
        # self.face_normals = visuals.MeshNormals(meshdata, primitive='face', color='yellow')
        # self.vertex_normals.transform = self.transform
        # self.vertex_normals.visible = True
        # self.face_normals.transform = self.transform
        # self.face_normals.visible = True
        self.freeze()


    @staticmethod
    def initial_normal_vertices(shape):
        # isv = self._initial_selection_vertices

        points = np.zeros((6, 4, 3), dtype=np.float32)

        x0 = shape[0]/2
        y0 = shape[1]/2
        z0 = shape[2]/2

        points[0] = np.array([[x0, 0, 0], [x0 + x0/2, 0, 0], [x0 + x0/2, 0, 0], [x0 + 2 * x0/3, 0, 0]])
        points[1] = -1 * points[0]
        points[2] = np.array([[0, y0, 0], [0, y0 + y0/2, 0], [0, y0 + y0/2, 0], [0, y0 + 2 * y0/3, 0]])
        points[3] = -1 * points[2]
        points[4] = np.array([[0, 0, z0], [0, 0, z0 + z0/2], [0, 0, z0 + z0/2], [0, 0, z0 + 2 * z0/3]])
        points[5] = -1 * points[4]

        return points


# noinspection PyAbstractClass
class SelectorBox(RenderedObject):
    count: int = 0

    def __init__(self, network_config: NetworkConfig, parent=None, name=None):
        super().__init__(name=name or f'{self.__class__.__name__}{SelectorBox.count}',
                         parent=parent, selectable=True)
        self.network_config = network_config
        self._obj: visuals.Box = DefaultBox(name=self.name + '.obj',
                                            shape=self.shape,
                                            color=(1, 0.65, 0, 0.1),
                                            edge_color=(1, 0.65, 0, 0.5),
                                            scale=[1.1, 1.1, 1.1],
                                            depth_test=False,
                                            border_width=2, init_normals=True,
                                            parent=self, interactive=self.selectable)
        # self._obj.name = self.name
        # self._obj.__class__.__name__ = self._obj.name

        SelectorBox.count += 1

        self.scale = Scale(self)
        self.pos = Position(self, _grid_unit_shape=self.shape)

        self.states_gpu: Optional[LocationGroupProperties] = None
        self.cuda_device: Optional[str] = None

        # noinspection PyPep8Naming
        G = self.network_config.G

        self.selected_masks = np.zeros((G, 4), dtype=np.int32, )

        self.g_pos_end = self.g_pos.copy()
        self.g_pos_end[:, 0] = self.g_pos_end[:, 0] + self.shape[0]
        self.g_pos_end[:, 1] = self.g_pos_end[:, 1] + self.shape[1]
        self.g_pos_end[:, 2] = self.g_pos_end[:, 2] + self.shape[2]

        self.group_numbers = np.arange(G)  # .reshape((G, 1))

    @property
    def g_pos(self):
        return self.network_config.g_pos[:self.network_config.G, :]

    @property
    def shape(self):
        return self.network_config.grid_unit_shape

    @property
    def vbo_glir_id(self):
        return self._obj._border._vertices.id

    @property
    def selection_vertices(self):
        return (self.obj._initial_selection_vertices
                * self.transform.scale[:3]
                + self.transform.translate[:3])

    def transform_changed(self):
        g_pos = self.g_pos
        v = self.selection_vertices
        self.selected_masks[:, 0] = (g_pos[:, 0] >= v[0, 0]) & (self.g_pos_end[:, 0] <= v[1, 0])
        self.selected_masks[:, 1] = (g_pos[:, 1] >= v[0, 1]) & (self.g_pos_end[:, 1] <= v[2, 1])
        self.selected_masks[:, 2] = (g_pos[:, 2] >= v[0, 2]) & (self.g_pos_end[:, 2] <= v[3, 2])
        self.selected_masks[:, 3] = self.network_config.G
        self.selected_masks[:, 3] = np.where(~self.selected_masks[:, :3].all(axis=1),
                                             self.selected_masks[:, 3], self.group_numbers)

        self.states_gpu.selected = torch.from_numpy(self.selected_masks[:, [3]]).to(self.cuda_device)

    def on_select_callback(self, v: bool):
        print(f'selected({v}): ', self.name)
        for c in self.obj.children:
            # c: NormalArrow
            c.visible = v
            # c.obj.color = 'white'
            # c.obj.update()


class _GSGLLineVisual(_GLLineVisual):
    _shaders = {
        'vertex': """
            varying out vec4 v_color;
            
            void main(void) {
                gl_Position = $transform($to_vec4($position));
                v_color = $color;
            }
        """,
        'fragment': """

            varying in vec4 g_color;
            
            void main() {
                gl_FragColor = g_color;
            }
        """
    }

    def __init__(self, parent, gcode):
        self._parent = parent
        self._pos_vbo = gloo.VertexBuffer()
        self._color_vbo = gloo.VertexBuffer()
        self._connect_ibo = gloo.IndexBuffer()
        self._connect = None

        Visual.__init__(self, vcode=self._shaders['vertex'],
                        gcode=gcode,
                        fcode=self._shaders['fragment'])
        self.set_gl_state('translucent')

    def _prepare_transforms(self, view):
        xform = view.transforms.get_transform()

        view.view_program.vert['transform'] = xform
        if view.view_program.geom is not None:
            # xform = view.transforms.get_transform(map_from='visual', map_to='render')
            view.view_program.geom['transform'] = xform
            view.view_program.geom['transform_inv'] = view.transforms.get_transform(map_from='render', map_to='visual')

    # noinspection DuplicatedCode
    def _prepare_draw(self, view):
        prof = Profiler()

        if self._parent._changed['pos']:
            if self._parent._pos is None:
                return False
            # todo: does this result in unnecessary copies?
            pos = np.ascontiguousarray(self._parent._pos.astype(np.float32))
            self._pos_vbo.set_data(pos)
            self._program.vert['position'] = self._pos_vbo
            # self._program.geom['position'] = self._pos_vbo
            self._program.vert['to_vec4'] = self._ensure_vec4_func(pos.shape[-1])
            # self._program.geom['to_vec4'] = self._ensure_vec4_func(pos.shape[-1])
            self._parent._changed['pos'] = False

        if self._parent._changed['color']:
            color, cmap = self._parent._interpret_color()
            # If color is not visible, just quit now
            if isinstance(color, Color) and color.is_blank:
                return False
            if isinstance(color, Function):
                # TODO: Change to the parametric coordinate once that is done
                self._program.vert['color'] = color(
                    '(gl_Position.x + 1.0) / 2.0')
            else:
                if color.ndim == 1:
                    self._program.vert['color'] = color
                else:
                    self._color_vbo.set_data(color)
                    self._program.vert['color'] = self._color_vbo
            self._parent._changed['color'] = False

            self.shared_program['texture2D_LUT'] = cmap and cmap.texture_lut()

            # noinspection PyTypeChecker
        # noinspection PyTypeChecker
        self.update_gl_state(line_smooth=bool(self._parent._antialias))
        px_scale = self.transforms.pixel_scale
        width = px_scale * self._parent._width
        self.update_gl_state(line_width=max(width, 1.0))

        if self._parent._changed['connect']:
            self._connect = self._parent._interpret_connect()
            if isinstance(self._connect, np.ndarray):
                self._connect_ibo.set_data(self._connect)
            self._parent._changed['connect'] = False
        if self._connect is None:
            return False

        prof('prepare')

        # Draw
        if isinstance(self._connect, str) and \
                self._connect == 'strip':
            self._draw_mode = 'line_strip'
            self._index_buffer = None
        elif isinstance(self._connect, str) and \
                self._connect == 'segments':
            self._draw_mode = 'lines'
            self._index_buffer = None
        elif isinstance(self._connect, np.ndarray):
            self._draw_mode = 'lines'
            self._index_buffer = self._connect_ibo
        else:
            raise ValueError("Invalid line connect mode: %r" % self._connect)

        prof('draw')


class GSLineVisual(LineVisual):

    def __init__(self, gcode, pos=None, color=(1., 1., 1., 1.), width=1,
                 connect='strip', method='gl', antialias=False):
        self.gcode = gcode
        super().__init__(pos=pos, color=color, width=width, connect=connect, method=method, antialias=antialias)
        # self.unfreeze()
        #
        # self.freeze()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):

        if method not in ('agg', 'gl'):
            raise ValueError('method argument must be "agg" or "gl".')
        if method == self._method:
            return

        self._method = method
        if self._line_visual is not None:
            self.remove_subvisual(self._line_visual)

        if method == 'gl':
            self._line_visual = _GSGLLineVisual(self, gcode=self.gcode)
        elif method == 'agg':
            self._line_visual = _AggLineVisual(self)
        self.add_subvisual(self._line_visual)

        for k in self._changed:
            self._changed[k] = True


GSLine = visuals.create_visual_node(GSLineVisual)


# noinspection PyAbstractClass
class BoxSystem(RenderedObject):

    def __init__(self, pos, grid_unit_shape, max_z, color=(0.1, 1., 1., 1.), **kwargs):
        gcode = f"""
                // #version 430

                layout (lines) in;
                layout (line_strip, max_vertices=16) out;

                in vec4 v_color[];
                out vec4 g_color;

                void main(void){{

                    float size_x = {grid_unit_shape[0]}f;
                    float size_y = {grid_unit_shape[1]}f;
                    float size_z = {grid_unit_shape[2]}f;

                    g_color = v_color[0];

                    // gl_PointSize = 75;

                    vec4 source_pos = gl_in[0].gl_Position;
                    
                    if ($transform_inv(source_pos).z > {max_z}f){{
                        return;
                    }}

                    vec4 x_offset = $transform(vec4(size_x, 0.f, 0.f, 0.f));
                    vec4 y_offset = $transform(vec4(0.f, size_y, 0.f, 0.f));
                    vec4 z_offset = $transform(vec4(0.f, 0.f, size_z, 0.f));

                    vec4 o = source_pos;
                    vec4 x = source_pos + x_offset;
                    vec4 y = source_pos + y_offset;
                    vec4 z = source_pos + z_offset;
                    vec4 xy = source_pos + x_offset + y_offset;
                    vec4 xz = source_pos + x_offset + z_offset;
                    vec4 yz = source_pos + y_offset + z_offset;
                    vec4 xyz = source_pos + x_offset + y_offset + z_offset;

                    gl_Position = o;
                    EmitVertex();	
                    gl_Position = x;
                    EmitVertex();		
                    gl_Position = xz;
                    EmitVertex();		
                    gl_Position = z;
                    EmitVertex();		
                    gl_Position = yz;
                    EmitVertex();		
                    gl_Position = xyz;
                    EmitVertex();		
                    gl_Position = xz;
                    EmitVertex();		
                    gl_Position = xyz;
                    EmitVertex();		
                    gl_Position = xy;
                    EmitVertex();		
                    gl_Position = x;
                    EmitVertex();		
                    gl_Position = xy;
                    EmitVertex();		
                    gl_Position = y;
                    EmitVertex();		
                    gl_Position = o;
                    EmitVertex();		
                    gl_Position = z;
                    EmitVertex();		
                    gl_Position = yz;
                    EmitVertex();		
                    gl_Position = y;
                    EmitVertex();	
                    EndPrimitive();

                }}
            """
        # gcode = None
        super().__init__()
        # noinspection PyArgumentList
        self._obj = GSLine(gcode=gcode, pos=pos, color=color, parent=self, **kwargs)
        self._obj.transform = STTransform(translate=(0, 0, 0), scale=(1, 1, 1))

    @property
    def vbo_glir_id(self):
        return self._obj._line_visual._pos_vbo.id

    @property
    def ibo_glir_id(self):
        return self._obj._line_visual._connect_ibo.id
