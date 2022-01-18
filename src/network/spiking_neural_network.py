# from dataclasses import dataclass
from enum import Enum, unique
import numba.cuda
import numpy as np
import pycuda.autoinit
import pycuda.driver
from pycuda.gl import RegisteredBuffer, RegisteredMapping
import torch
from typing import Optional
from vispy.scene import visuals

from .utils import (
    DefaultNetworkConfig,
    ExternalMemory
)
from .rendering import (
    NetworkScatterPlot,
    RenderedObject,
    SelectorBox,
    default_box
)

from gpu import func2, snn_engine_gpu

@unique
class NeuronTypes(Enum):
    INHIBITORY = 0
    EXCITATORY = 1


class NeuronTypeGroup:

    """
    Index-container for type-neuron-groups.
    """

    # noinspection PyPep8Naming
    def __init__(self, ID, start_idx, end_idx, S, neuron_type, group_dct, verbose=True):

        if (ID in group_dct) or ((len(group_dct) > 0) and (ID < max(group_dct))):
            raise AssertionError

        self.id = ID
        self.type = neuron_type if isinstance(neuron_type, NeuronTypes) else NeuronTypes(neuron_type)
        self.start_idx = start_idx  # index of the first neuron of this group
        self.end_idx = end_idx  # index of the last neuron of this group
        self.S = S

        group_dct[ID] = self
        NeuronTypeGroup._last_instance = self

        if verbose is True:
            print('NEW:', self)

    @property
    def size(self):
        return self.end_idx - self.start_idx + 1

    def __len__(self):
        return self.size

    def __str__(self):
        return f'NeuronTypeGroup(id={self.id}, type={self.type.name}, [{self.start_idx}, {self.end_idx}])'

    # noinspection PyPep8Naming
    @classmethod
    def from_count(cls, ID, nN, S, neuron_type, group_dct):
        last_group = group_dct[max(group_dct)] if len(group_dct) > 0 else None
        if last_group is None:
            start_idx = 0
            end_idx = nN - 1
        else:
            start_idx = last_group.end_idx + 1
            end_idx = last_group.end_idx + nN
        return NeuronTypeGroup(ID, start_idx, end_idx, S=S, neuron_type=neuron_type, group_dct=group_dct)


class NetworkCPUArrays:
    # noinspection PyPep8Naming
    def __init__(self, config, T, n_N_states):
        # NEURONS (inhibitory + excitatory)
        self.N_pos = None  # initialized on the GPU
        N = config.N
        S = config.S
        D = config.D
        G = config.G

        # Network Representation
        self.N_rep = np.zeros((N, S), dtype=np.float32)
        self.N_weights = np.zeros((N, S), dtype=np.float32)
        self.N_delays = np.zeros((D + 1, N), dtype=np.int32)

        self.N_fired = np.zeros(N, dtype=np.int32)
        self.firing_times = np.zeros((15, N), dtype=np.float32)
        self.firing_idcs = np.zeros((15, N), dtype=np.int32)
        self.firing_counts = np.zeros(2 * T, dtype=np.int32)

        # pt, u, v, a, b, c, d, I
        self.N_states = np.zeros((n_N_states, N), dtype=np.float32)
        self.N_stpd = np.zeros((N, S * 2), dtype=np.float32)

        # GROUPS (location-based)

        # position of each location group
        self.G_pos = np.zeros((G, 3), dtype=np.int32)
        # delay between groups (as a distance matrix)
        self.G_delay_distance = np.zeros((G, G), dtype=np.int32)
        # number of groups per delays
        self.G_delay_counts = np.zeros((G, D + 1), dtype=np.int32)
        # [0, 1]: inhibitory count, excitatory count,
        # [2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        self.G_neuron_counts = np.zeros((2 + 2 * D, G), dtype=np.int32)
        self.G_neuron_typed_ccount = np.zeros((1, 2 * G + 1), dtype=np.int32)

        # expected cumulative sum of synapses per source types and delay (sources types: inhibitory or excitatory)
        self.G_exp_syn_ccount_per_src_type_and_delay = np.zeros((2 * (D + 1), G), dtype=np.int32)

        # expected cumulative sum of excitatory synapses per delay and per sink type
        # (sink types: inhibitory, excitatory)
        self.G_exp_exc_syn_ccount_per_snk_type_and_delay = np.zeros((2 * (D + 1), G), dtype=np.int32)

        self.G_conn_probs = np.zeros((2 * G, D), dtype=np.float32)
        self.local_autapse_idcs = np.zeros((3 * D, G), dtype=np.int32)

        self.G_props = np.zeros((10, G), dtype=np.int32)  # selected_p, thalamic input (on/off), ...


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
    def tensor(self):
        self.map()
        if self._tensor is None:
            self._tensor = torch.as_tensor(self.device_array, device=self.conf.device)
        return self._tensor


class NetworkGPUArrays:

    # noinspection PyPep8Naming,SpellCheckingInspection
    def __init__(self,
                 config: DefaultNetworkConfig,
                 pos_vbo: int,
                 type_group_dct: dict,
                 device: int,
                 G_shape: tuple,
                 ):

        self.device = torch.device(device)
        # nbcuda.select_device(device)

        N = config.N
        S = config.S
        D = config.D
        G = config.G

        nf32 = dict(dtype=np.float32, device=self.device)
        ti32 = dict(dtype=torch.int32, device=self.device)

        scatter_plot_layout = GPUArrayConfig(shape=(N, 13), strides=(13 * 4, 4), **nf32)
        self.N_pos = RegisteredGPUArray.from_vbo(pos_vbo, config=scatter_plot_layout)
        for g in type_group_dct.values():
            if g.type.value == NeuronTypes.INHIBITORY.value:
                orange = torch.Tensor([1, .5, .2])
                self.N_pos.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange

        self.N_G = torch.zeros([N, 3], **ti32)
        self.N_G[:, 0] = torch.arange(self.N_G.shape[0])  # Neuron Id
        for g in type_group_dct.values():
            self.N_G[g.start_idx:g.end_idx + 1, 1] = g.type.value  # Neuron Type

        self.G_neuron_counts = torch.zeros([2 + 2 * D, G], **ti32)

        ngp = self.N_G.data_ptr()
        npp = self.N_pos.ptr

        print(ngp)
        print(npp)
        print('\n')
        snn_engine_gpu.init_pos_gpu(N, G, npp, ngp, G_shape)
        print('\n')
        print(self.N_G)

        print()
        # self.N_G = torch.zeros((2, N), device=f'gpu:{self.device}', dtype=np.float32)
        # self.
        # print(config.pos)
        # cpu_arrays.N_pos = pos.copy_to_host()
        # print(cpu_arrays.N_pos)
        # print(self.N_pos)


class SpikingNeuronNetwork:

    # noinspection PyPep8Naming
    def __init__(self, config: DefaultNetworkConfig = DefaultNetworkConfig(), T: int = 2000):

        RenderedObject._grid_unit_shape = config.grid_unit_shape

        self.N = config.N
        self.S = config.S
        self.D = config.D
        self.G = config.G
        self.T = T

        self.n_N_states = 8

        self.config = config
        self._CPU = None
        self._scatter_plot = NetworkScatterPlot(self.config)
        print(self.config.pos)
        self.GPU: Optional[NetworkGPUArrays] = None

        self._outer_grid = None
        self._selector_box = None
        print()
        self.type_group_dct = {}
        n_inhN = int(.2 * self.N)
        NeuronTypeGroup.from_count(0, n_inhN, self.S, NeuronTypes.INHIBITORY, group_dct=self.type_group_dct)
        NeuronTypeGroup.from_count(1, self.N - n_inhN, self.S, NeuronTypes.EXCITATORY, group_dct=self.type_group_dct)
        print()

    # noinspection PyPep8Naming
    @property
    def CPU(self):
        if not self._CPU:
            self._CPU = NetworkCPUArrays(self.config, T=self.T, n_N_states=self.n_N_states)
        return self._CPU

    @property
    def scatter_plot(self) -> NetworkScatterPlot:
        return self._scatter_plot

    @property
    def outer_grid(self) -> visuals.Box:
        if self._outer_grid is None:
            self._outer_grid: visuals.Box = default_box(shape=self.config.shape,
                                                        scale=[.99, .99, .99],
                                                        segments=self.config.G_shape)
            # self._outer_grid.set_gl_state('translucent', blend=True, depth_test=True)
            # self._outer_grid.mesh.set_gl_state('translucent', blend=True, depth_test=True)
            self._outer_grid.visible = False

        return self._outer_grid

    @property
    def selector_box(self) -> visuals.Box:
        if self._selector_box is None:
            self._selector_box = SelectorBox(self.config.grid_unit_shape)
        return self._selector_box

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device):
        self.GPU = NetworkGPUArrays(
            config=self.config,
            pos_vbo=self._scatter_plot.vbo,
            type_group_dct=self.type_group_dct,
            G_shape=self.config.G_shape,
            device=device
        )
