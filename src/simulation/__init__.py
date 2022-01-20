import ctypes
from numba import cuda as nbcuda
import numpy as np
from gpu import snn_construction_gpu
from pycuda import autoinit, gl as pycudagl


class ExternalMemory(object):
    """
    Provide an externally managed memory.
    Interface requirement: __cuda_memory__, device_ctypes_pointer, _cuda_memsize_
    """
    __cuda_memory__ = True

    def __init__(self, ptr, size):
        self.device_ctypes_pointer = ctypes.c_void_p(ptr)
        self._cuda_memsize_ = size


def vbodata2host(v):

    # dev = autoinit.device
    # pycudagl.make_context(dev, flags=0)
    reg3: pycudagl.RegisteredBuffer = pycudagl.RegisteredBuffer(v)
    map3: pycudagl.RegisteredMapping = reg3.map(None)
    ptr, size = map3.device_ptr_and_size()
    map3.unmap()
    # p = snn_construction_gpu.CudaGLResource()
    source_ptr = ExternalMemory(ptr, size)
    nbcuda.select_device(0)
    print(v)
    raise NotImplementedError
    # snn_construction_gpu.pyadd(3, ptr, ptr)
    # device_array = nbcuda.cudadrv.devicearray.DeviceNDArray(shape=(10, 13), strides=(13 * 4, 4), dtype=np.float32,
    #                                                         stream=0, gpu_data=source_ptr)
    # host_array = device_array.copy_to_host()
    # print(host_array)
    # print('done')


