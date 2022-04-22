from .rendered_object import RenderedObjectNode


class CudaObject:
    def __init__(self):
        try:
            self.unfreeze()
            self._cuda_device = None
            self._gpu_array = None
            self.freeze()
        except AttributeError:
            self._cuda_device = None
            self._gpu_array = None

    @property
    def cuda_device(self):
        return self._cuda_device

    @cuda_device.setter
    def cuda_device(self, v):
        self._cuda_device = v
        for child in self.children:
            child.cuda_device = v
        try:
            for subv in self._subvisuals:
                subv.cuda_device = v
        except AttributeError:
            pass
        try:
            for subv in self.normals:
                subv.cuda_device = v
        except AttributeError:
            pass

    def ini_cuda_attributes(self):
        pass


# noinspection PyAbstractClass
class RenderedCudaObjectNode(RenderedObjectNode, CudaObject):

    def __init__(self,
                 subvisuals,
                 parent=None,
                 # name=None,
                 # transforms=None,
                 selectable=False):
        RenderedObjectNode.__init__(self,
                                    subvisuals,
                                    parent=parent,
                                    # name=name,
                                    # transforms=transforms,
                                    selectable=selectable)

        CudaObject.__init__(self)
