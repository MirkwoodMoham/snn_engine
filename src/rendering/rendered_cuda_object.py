import numpy as np
from .rendered_object import RenderedObjectNode
from gpu import GPUArrayConfig, RegisteredGPUArray


class CudaObject:
    def __init__(self):
        try:
            # noinspection PyUnresolvedReferences
            self.unfreeze()
            self._cuda_device = None
            self._gpu_array = None
            # noinspection PyUnresolvedReferences
            self.freeze()
        except AttributeError:
            self._cuda_device = None
            self._gpu_array = None

    def _init_cuda_attributes(self, device, attr_list):
        for a in attr_list:
            try:
                for o in getattr(self, a):
                    o.init_cuda_attributes(device)
            except AttributeError:
                pass

    def init_cuda_attributes(self, device):
        self._cuda_device = device
        self.init_cuda_arrays()
        self._init_cuda_attributes(device, attr_list=['children', '_subvisuals', 'normals'])

    @property
    def gpu_array(self):
        return self._gpu_array

    def init_cuda_arrays(self):
        pass


# noinspection PyAbstractClass
class RenderedCudaObjectNode(RenderedObjectNode, CudaObject):

    def __init__(self,
                 subvisuals,
                 parent=None,
                 name=None,
                 selectable=False,
                 draggable=False):
        RenderedObjectNode.__init__(self,
                                    subvisuals,
                                    parent=parent,
                                    name=name,
                                    selectable=selectable,
                                    draggable=draggable)

        CudaObject.__init__(self)

    def face_color_array(self, meshvisual):
        nbytes = 4
        shape = (meshvisual._meshdata.n_faces * 3, 4)
        return RegisteredGPUArray.from_buffer(
            self.color_vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes, nbytes),
                                                  dtype=np.float32, device=self._cuda_device))
