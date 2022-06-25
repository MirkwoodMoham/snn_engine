import numpy as np

from vispy.geometry import create_plane
from vispy.gloo.context import get_current_canvas
from vispy.visuals import MeshVisual

from .network_config import NetworkConfig
from .network_grid import NetworkGrid


class GroupMeshVisual(MeshVisual):

    def __init__(
            self,
            network_config: NetworkConfig, grid: NetworkGrid,
            orientation, grid_coord, face_colors=np.array([0., 0., 0., 1.])):
        assert orientation in ['+x', '-x', '+y', '-y', '+z', '-z']
        vertices, faces, outline = self.create_planes(network_config, grid, orientation, grid_coord)
        if face_colors.shape == (4,):
            face_colors = np.repeat(face_colors.reshape(1, 4), len(faces), axis=0)
        self.orientation = orientation
        self._color_vbo = None
        super().__init__(vertices['position'], faces, None, face_colors, None)

    @staticmethod
    def create_planes(
            network_config: NetworkConfig, grid: NetworkGrid,
            dir_, grid_coord, height=None, width=None, width_segments=None, height_segments=None):

        # dirs = ('x', 'y', 'z')

        if 'y' in dir_:
            i = 1
        elif 'z' in dir_:
            i = 2
        else:
            i = 0
        i_planes = np.unique(grid_coord[:, i]) * grid.unit_shape[i]
        # i_planes = np.nunique(grid_pos[:, (i + 1) % 3])

        if i == 2:
            i_planes = i_planes[i_planes < network_config.max_z]

        n_planes = len(i_planes)

        j = (i + 1) % 3
        j_segments = np.unique(grid_coord[:, j])
        n_j_segments = len(j_segments)
        j_pos_shape = network_config.N_pos_shape[j]
        k = (i + 2) % 3
        k_segments = np.unique(grid_coord[:, k])
        n_k_segments = len(k_segments)
        k_pos_shape = network_config.N_pos_shape[k]

        height = height or [n_j_segments * grid.unit_shape[j]] * n_planes
        width = width or [n_k_segments * grid.unit_shape[k]] * n_planes
        width_segments = width_segments or [n_j_segments] * n_planes
        height_segments = height_segments or [n_k_segments] * n_planes

        planes_m = []

        for idx, y in enumerate(i_planes):
            vertices_p, faces_p, outline_p = create_plane(height[idx], width[idx],
                                                          width_segments[idx], height_segments[idx], dir_)
            vertices_p['position'][:, k] += ((np.min(k_segments) * grid.unit_shape[k] + k_pos_shape)
                                             / 2)
            vertices_p['position'][:, j] += ((np.min(j_segments) * grid.unit_shape[j] + j_pos_shape)
                                             / 2)
            vertices_p['position'][:, i] = y + grid.unit_shape[i] * int('+' in dir_)
            planes_m.append((vertices_p, faces_p, outline_p))

        # noinspection DuplicatedCode
        positions = np.zeros((0, 3), dtype=np.float32)
        texcoords = np.zeros((0, 2), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)

        faces = np.zeros((0, 3), dtype=np.uint32)
        outline = np.zeros((0, 2), dtype=np.uint32)
        offset = 0
        for vertices_p, faces_p, outline_p in planes_m:
            positions = np.vstack((positions, vertices_p['position']))
            texcoords = np.vstack((texcoords, vertices_p['texcoord']))
            normals = np.vstack((normals, vertices_p['normal']))

            faces = np.vstack((faces, faces_p + offset))
            outline = np.vstack((outline, outline_p + offset))
            offset += vertices_p['position'].shape[0]

        vertices = np.zeros(positions.shape[0],
                            [('position', np.float32, 3),
                             ('texcoord', np.float32, 2),
                             ('normal', np.float32, 3),
                             ('color', np.float32, 4)])

        colors = np.ravel(positions)
        colors = np.hstack((np.reshape(np.interp(colors,
                                                 (np.min(colors),
                                                  np.max(colors)),
                                                 (0, 1)),
                                       positions.shape),
                            np.ones((positions.shape[0], 1))))

        vertices['position'] = positions
        vertices['texcoord'] = texcoords
        vertices['normal'] = normals
        vertices['color'] = colors

        return vertices, faces, outline

    def face_color_array(self, device):
        from gpu import RegisteredVBO
        return RegisteredVBO(buffer=self.color_vbo, shape=(self._meshdata.n_faces * 3, 4), device=device)

    @staticmethod
    def buffer_id(glir_id):
        return int(get_current_canvas().context.shared.parser._objects[glir_id].handle)

    @property
    def color_vbo(self):
        if self._color_vbo is None:
            self._color_vbo = self.buffer_id(self.shared_program.vert['base_color'].id)
        return self._color_vbo

    @property
    def pos_vbo(self):
        return self.buffer_id(self.shared_program.vert['position'].id)

    def vbo_array(self, device):

        from gpu import RegisteredVBO
        return RegisteredVBO(buffer=self.pos_vbo, shape=(self._meshdata.n_faces * 3, 3), device=device)

