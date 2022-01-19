import ctypes
import numba.cuda
import numpy as np
import pandas as pd
from pycuda.gl import RegisteredBuffer, RegisteredMapping
import torch
from typing import Optional


class ExternalMemory(object):
    """
    Provide an externally managed memory.
    Interface requirement: __cuda_memory__, device_ctypes_pointer, _cuda_memsize_
    """
    __cuda_memory__ = True

    def __init__(self, ptr, size):
        self.device_ctypes_pointer = ctypes.c_void_p(ptr)
        self._cuda_memsize_ = size


class GPUArrayConfig:

    def __init__(self, shape=None, strides=None, dtype=None, stream=0, device: torch.device = None):

        self.shape: Optional[tuple] = shape
        self.strides:  tuple = strides
        self.dtype: np.dtype = dtype

        self.stream: int = stream

        self.device: torch.device = device

    @classmethod
    def from_cpu_array(cls, cpu_array, dev: torch.device = None, stream=0):
        shape: tuple = cpu_array.shape
        strides:  tuple = cpu_array.strides
        dtype: np.dtype = cpu_array.dtype
        return GPUArrayConfig(shape=shape, strides=strides, dtype=dtype, stream=stream, device=dev)


class RegisteredGPUArray:

    def __init__(self,
                 gpu_data: ExternalMemory = None,
                 reg: RegisteredBuffer = None,
                 mapping: RegisteredMapping = None,
                 ptr: int = None,
                 config: GPUArrayConfig = None):

        self.reg: RegisteredBuffer = reg
        self.mapping: RegisteredMapping = mapping
        self.ptr: int = ptr
        self.conf: GPUArrayConfig = config

        self.gpu_data: ExternalMemory = gpu_data
        self.device_array = self._numba_device_array()
        self._tensor = None

    def __call__(self, *args, **kwargs):
        return self.tensor

    def _numba_device_array(self):
        # noinspection PyUnresolvedReferences
        return numba.cuda.cudadrv.devicearray.DeviceNDArray(
            shape=self.conf.shape,
            strides=self.conf.strides,
            dtype=self.conf.dtype,
            stream=self.conf.stream,
            gpu_data=self.gpu_data)

    def copy_to_host(self):
        return self.device_array.copy_to_host()

    @property
    def ctype_ptr(self):
        return self.gpu_data.device_ctypes_pointer

    @property
    def size(self):
        # noinspection PyProtectedMember
        return self.gpu_data._cuda_memsize_

    # noinspection PyArgumentList
    @classmethod
    def from_vbo(cls, vbo, config: GPUArrayConfig = None, cpu_array: np.array = None):

        if config is not None:
            assert cpu_array is None
        else:
            config = GPUArrayConfig.from_cpu_array(cpu_array)

        reg = RegisteredBuffer(vbo)
        mapping: RegisteredMapping = reg.map(None)
        ptr, size = mapping.device_ptr_and_size()
        gpu_data = ExternalMemory(ptr, size)
        mapping.unmap()

        return RegisteredGPUArray(gpu_data=gpu_data, reg=reg, mapping=mapping, ptr=ptr, config=config)

    def map(self):
        self.reg.map(None)

    def unmap(self):
        # noinspection PyArgumentList
        self.mapping.unmap()

    @property
    def tensor(self) -> torch.Tensor:
        self.map()
        if self._tensor is None:
            self._tensor = torch.as_tensor(self.device_array, device=self.conf.device)
        return self._tensor

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.tensor.cpu().numpy())
