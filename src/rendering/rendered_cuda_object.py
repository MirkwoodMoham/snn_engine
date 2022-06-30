from .rendered_object import RenderedObjectNode
from gpu import RegisteredGPUArray


class CudaObject:
    def __init__(self):
        try:
            # noinspection PyUnresolvedReferences
            self.unfreeze()
            self._cuda_device = None
            self._gpu_array = None
            self.registered_buffers: list[RegisteredGPUArray] = []
            # noinspection PyUnresolvedReferences
            self.freeze()
        except AttributeError:
            self._cuda_device = None
            self._gpu_array = None
            self.registered_buffers: list[RegisteredGPUArray] = []

    def _init_cuda_attributes(self, device, attr_list):
        for a in attr_list:
            if hasattr(self, a):
                for o in getattr(self, a):
                    if hasattr(o, 'init_cuda_attributes'):
                        o.init_cuda_attributes(device)

    def init_cuda_attributes(self, device):
        self._cuda_device = device
        self.init_cuda_arrays()
        self._init_cuda_attributes(device, attr_list=['children', '_subvisuals', 'normals'])

    @property
    def gpu_array(self):
        return self._gpu_array

    def init_cuda_arrays(self):
        pass

    def unregister_registered_buffers(self):
        for rb in self.registered_buffers:
            rb.unregister()


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

    def face_color_array(self, mesh_data, buffer):
        from gpu import RegisteredVBO
        return RegisteredVBO(buffer=buffer, shape=(mesh_data.n_faces * 3, 4), device=self._cuda_device)
