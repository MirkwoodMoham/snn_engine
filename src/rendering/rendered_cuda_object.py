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

        # for child in self.children:
        #     child.cuda_device = v
        # try:
        #     for subvisual in self._subvisuals:
        #         subvisual.cuda_device = v
        # except AttributeError:
        #     pass
        # try:
        #     for normal in self.normals:
        #         normal.cuda_device = v
        # except AttributeError:
        #     pass

    def _init_cuda_attributes(self, device, attr_list):
        for a in attr_list:
            try:
                for o in getattr(self, a):
                    o.init_cuda_attributes(device)
            except AttributeError:
                pass

    def init_cuda_attributes(self, device, *args, **kwargs):
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
                 # transforms=None,
                 selectable=False,
                 draggable=False):
        RenderedObjectNode.__init__(self,
                                    subvisuals,
                                    parent=parent,
                                    name=name,
                                    # transforms=transforms,
                                    selectable=selectable,
                                    draggable=draggable)

        CudaObject.__init__(self)
