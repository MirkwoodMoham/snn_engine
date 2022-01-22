import ctypes
import numba.cuda
import numpy as np
import pandas as pd
from pycuda.gl import RegisteredBuffer, RegisteredMapping
import torch
from typing import Optional


def shape_size(shape, unit=None):
    if not unit:
        return np.array(shape).cumprod()[-1]
    elif isinstance(unit, int):
        return shape_size(shape) / unit
    elif unit == 'mb':
        return shape_size(shape, pow(10, 6))


def reshape_wrt_size(shape, max_size_mb, n_bytes=4):

    if ((len(shape) != 2)
            or (not isinstance(shape[0], int))
            or (not isinstance(shape[1], int))):
        raise TypeError

    max_size = np.ceil((max_size_mb * pow(10, 6)) / n_bytes)
    shape_size_ = shape_size(shape)

    if shape_size_ > max_size:
        max_square_length = np.sqrt(max_size)
        new_shape = np.array([max_square_length, max_square_length])

        for i in range(len(shape)):
            if shape[i] < new_shape[i]:
                new_shape[i] = shape[i]
                if i < (len(shape) - 1):
                    new_shape[i + 1] += (new_shape[i + 1] - new_shape[i])

        sqrt_new_shape = tuple(np.sqrt(new_shape))

        for i in range(len(shape)):
            while ((((shape[i] % (new_shape[i] - sqrt_new_shape[i])) - (shape[i] % new_shape[i])) > 0)
                   and ((shape[i] % new_shape[i]) > 0)):
                """
                Goal: find a better x such that batch sizes become closer to each other.
                As long a reducing x by sqrt(x) does not result in int(old_x/x) increasing, reduce x by sqrt(x). 
                """
                new_shape[i] -= sqrt_new_shape[i]

        new_shape_size = np.array(new_shape).cumprod()[-1]

        if not new_shape_size < max_size:
            raise ValueError

        return tuple(new_shape)
    return shape


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
