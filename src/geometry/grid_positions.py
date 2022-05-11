import numpy as np


class DirectedObject:

    directions = {
        '+x': 0,
        '-x': 1,
        '+y': 2,
        '-y': 3,
        '+z': 4,
        '-z': 5,
    }

    coord = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ])

    def __init__(self, obj):
        self._index = 0
        self._obj = obj

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._obj[item]
        elif isinstance(item, str):
            return self._obj[self.directions[item]]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index > 5:
            raise StopIteration
        next_ = self[self._index]
        self._index += 1
        return next_


def validate_pos(pos, shape):
    for i in range(3):
        assert np.min(pos[:, i]) >= 0
        assert np.max(pos[:, i]) <= shape[i]


def grid_coordinates(pos, outer_shape, grid_segmentation, as_struct=False):
    arr = (np.floor((pos
           / np.array(outer_shape, dtype=np.float32))
           * np.array(grid_segmentation, dtype=np.float32)).astype(int))
    if as_struct is True:
        # struct = np.zeros((len(arr), 1), [('x', np.int32), ('y', np.int32), ('z', np.int32)])
        struct = np.zeros(len(arr), [('x', np.int32), ('y', np.int32), ('z', np.int32)])
        struct['x'] = arr[:, 0]
        struct['y'] = arr[:, 1]
        struct['z'] = arr[:, 2]
        # struct['x'] = arr[:, [0]]
        # struct['y'] = arr[:, [1]]
        # struct['z'] = arr[:, [2]]
        return struct
    return arr


def pos_cloud(size=100000):

    pos = np.random.normal(size=(size, 3), scale=0.2)
    # one could stop here for the data generation, the rest is just to make the
    # data look more interesting. Copied over from magnify.py
    centers = np.random.normal(size=(50, 3))
    indexes = np.random.normal(size=size, loc=centers.shape[0] / 2.,
                               scale=centers.shape[0] / 3.)
    indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)
    scales = 10 ** (np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
    pos *= scales
    pos += centers[indexes]

    return pos


class GridPositions:

    def __init__(self):
        pass


def initial_normal_vertices(shape):
    # isv = self._initial_selection_vertices

    points = np.zeros((6, 4, 3), dtype=np.float32)

    x0 = shape[0] / 2
    y0 = shape[1] / 2
    z0 = shape[2] / 2

    points[0] = np.array([[x0, 0, 0], [x0 + x0 / 2, 0, 0], [x0 + x0 / 2, 0, 0], [x0 + 2 * x0 / 3, 0, 0]])
    points[1] = -1 * points[0]
    points[2] = np.array([[0, y0, 0], [0, y0 + y0 / 2, 0], [0, y0 + y0 / 2, 0], [0, y0 + 2 * y0 / 3, 0]])
    points[3] = -1 * points[2]
    points[4] = np.array([[0, 0, z0], [0, 0, z0 + z0 / 2], [0, 0, z0 + z0 / 2], [0, 0, z0 + 2 * z0 / 3]])
    points[5] = -1 * points[4]

    return points
