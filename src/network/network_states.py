from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union

from .network_structures import NeuronTypes
from gpu import (
    GPUArrayConfig,
    RegisteredGPUArray,
    RegisteredVBO,
    GPUArrayCollection
)
from network.network_config import NetworkConfig
from .network_grid import NetworkGrid


class PropertyTensor:

    @dataclass(frozen=True)
    class Rows:
        def __len__(self):
            pass

    _rows = None

    def __init__(self, shape, tensor: Optional[torch.Tensor] = None):
        self._tensor: Optional[torch.Tensor] = None
        if tensor is not None:
            self.tensor: Optional[torch.Tensor] = tensor
        if shape[0] != len(self.rows):
            raise ValueError

    @classmethod
    def __len__(cls):
        return len(cls.Rows())

    def __str__(self):
        return f"{self.__class__.__name__}:\n" + str(self.tensor)

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def rows(cls):
        if cls._rows is None:
            cls._rows = cls.Rows()
        return cls._rows

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, v):
        if self._tensor is None:
            if ((not len(v.shape) == 2)
                    or (not v.shape[0] == len(self))):
                raise ValueError

            self._tensor: torch.Tensor = v
            self.has_tensor = True
        else:
            raise AttributeError

    def data_ptr(self):
        return self._tensor.data_ptr()

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._tensor.cpu().numpy())


@dataclass(frozen=True)
class StateRows:

    pt: int = 0
    v: int = 1
    i: int = 2

    def __post_init__(self):
        vs = np.array(list(asdict(self).values()))
        if not np.max(vs) == self.i:
            raise AttributeError

        if not np.math.factorial(self.i) == np.cumprod(vs[vs > 0])[-1]:
            raise AttributeError

    def __len__(self):
        return self.i + 1


# noinspection PyPep8Naming
class IzhikevichPresets(PropertyTensor):

    @dataclass(frozen=True)
    class Preset:
        a: float
        b: float
        c: float
        d: float

    @dataclass(frozen=True)
    class Rows:

        rs: int = 0
        ib: int = 1
        ch: int = 2
        fs: int = 3
        tc: int = 4
        rz: int = 5
        lts: int = 6

        def __len__(self):
            return 7

    def __init__(self, shape=(7, 4), device=None):
        self._rows = self.Rows()
        super().__init__(shape)

        # self.tensor = torch.zeros(shape, dtype=torch.float32, device=device)

        self.rs = self.Preset(a=0.02, b=0.2, c=-65., d=8.)
        self.ib = self.Preset(a=0.02, b=0.2, c=-55., d=4.)
        self.ch = self.Preset(a=0.02, b=0.2, c=-50., d=2.)
        self.fs = self.Preset(a=0.1, b=0.2, c=-65., d=2.)
        self.fs25 = self.Preset(a=0.09, b=0.24, c=-65., d=2.)
        self.tc = self.Preset(a=0.02, b=0.25, c=-65., d=0.05)
        self.rz = self.Preset(a=0.1, b=0.26, c=-65., d=2.)
        self.lts = self.Preset(a=0.02, b=0.25, c=-65., d=2.)


class IzhikevichModel(PropertyTensor):

    @dataclass(frozen=True)
    class Rows(StateRows):

        pt: int = 0
        u: int = 1
        v: int = 2
        a: int = 3
        b: int = 4
        c: int = 5
        d: int = 6
        i: int = 7

    def __init__(self, n_neurons, device, types_tensor):
        self._rows = self.Rows()
        shape = (len(self._rows), n_neurons)
        super().__init__(shape)
        self._N = n_neurons
        self.selected = None
        self.set_tensor(shape, device, types_tensor)
        self.presets = IzhikevichPresets()

    def set_tensor(self, shape, device, types_tensor):

        self.tensor = torch.zeros(shape, dtype=torch.float32, device=device)

        mask_inh = types_tensor == NeuronTypes.INHIBITORY.value
        mask_exc = types_tensor == NeuronTypes.EXCITATORY.value
        r = torch.rand(self._N, dtype=torch.float32, device=device)

        self.pt = torch.rand(self._N, dtype=torch.float32, device=device)
        self.v = -65
        self.a = .02 + .08 * r * mask_inh
        self.b = .2 + .05 * (1. - r) * mask_inh
        self.c = -65 + 15 * (r ** 2) * mask_exc
        self.d = 2 * mask_inh + (8 - 6 * (r ** 2)) * mask_exc
        self.u = self.b * self.v

        self.selected = torch.zeros(self._N, dtype=torch.int32, device=device)

    def use_preset(self, preset: Union[IzhikevichPresets.Preset, str], mask=None):

        if isinstance(preset, str):
            preset = getattr(self.presets, preset)

        if mask is None:
            mask = self.selected

        self.a[mask] = preset.a
        self.b[mask] = preset.b
        self.c[mask] = preset.c
        self.d[mask] = preset.d

    @property
    def pt(self):
        return self._tensor[self._rows.pt, :]

    @pt.setter
    def pt(self, v):
        self._tensor[self._rows.pt, :] = v

    @property
    def u(self):
        return self._tensor[self._rows.u, :]

    @u.setter
    def u(self, v):
        self._tensor[self._rows.u, :] = v

    @property
    def v(self):
        return self._tensor[self._rows.u, :]

    @v.setter
    def v(self, v):
        self._tensor[self._rows.v, :] = v

    @property
    def a(self):
        return self._tensor[self._rows.a, :]

    @a.setter
    def a(self, v):
        self._tensor[self._rows.a, :] = v

    @property
    def b(self):
        return self._tensor[self._rows.b, :]

    @b.setter
    def b(self, v):
        self._tensor[self._rows.b, :] = v

    @property
    def c(self):
        return self._tensor[self._rows.c, :]

    @c.setter
    def c(self, v):
        self._tensor[self._rows.c, :] = v

    @property
    def d(self):
        return self._tensor[self._rows.d, :]

    @d.setter
    def d(self, v):
        self._tensor[self._rows.d, :] = v

    @property
    def i(self):
        return self._tensor[self._rows.i, :]

    @i.setter
    def i(self, v):
        self._tensor[self._rows.i, :] = v


class Sliders:
    def __init__(self, rows):
        for k in asdict(rows):
            setattr(self, k, None)


class Row:

    def __init__(self, index, value_interval):
        self.index = index
        self.interval = value_interval


class LocationGroupFlags(PropertyTensor):

    @dataclass(frozen=True)
    class Rows:

        sensory_input_type: Row = Row(0, [-1, 1])
        b_thalamic_input: Row = Row(1, [0, 1])
        b_sensory_group: Row = Row(2, [0, 1])
        b_sensory_input: Row = Row(3, [0, 1])
        b_output_group: Row = Row(4, [0, 1])
        output_type: Row = Row(5, [-1, 1])
        b_monitor_group_firing_count: Row = Row(6, [0, 1])

        def __len__(self):
            return 7

    def __init__(self, n_groups, device, select_ibo, grid: NetworkGrid):
        self._rows = self.Rows()
        self._G = n_groups
        shape = (len(self._rows), n_groups)
        super().__init__(shape)
        nbytes = 4
        self.selected_array = RegisteredGPUArray.from_buffer(
            select_ibo, config=GPUArrayConfig(shape=(self._G+1, 1),
                                              strides=(2 * nbytes, nbytes),
                                              dtype=np.int32, device=device))

        # self.group_numbers_gpu: Optional[torch.Tensor] = None
        self.set_tensor(shape, device, grid)
        self.selection_flag = None

    def set_tensor(self, shape, device, grid: NetworkGrid):
        self.tensor = torch.zeros(shape, dtype=torch.int32, device=device)
        thalamic_input_arr = torch.zeros(self._G)
        thalamic_input_arr[: int(self._G/2)] = 1
        # self.b_thalamic_input = thalamic_input_arr
        self.b_thalamic_input = 0
        # self.thalamic_input = 1

        self.b_sensory_group = torch.from_numpy(grid.sensory_group_mask).to(device)
        self.sensory_input_type = -1

        self.group_ids = (torch.arange(self._G).to(device=device)
                          .reshape((self._G, 1)))

        # self.b_monitor_group_firing_count[67] = 1
        # self.b_monitor_group_firing_count[68] = 1
        # self.b_monitor_group_firing_count[69] = 1
        # self.b_monitor_group_firing_count[123] = 1
        # self.b_monitor_group_firing_count[125] = 1
        self.b_monitor_group_firing_count = 1

    @property
    def selected(self):
        # noinspection PyUnresolvedReferences
        return (self.selected_array.tensor != self._G).flatten()[: self._G]

    @selected.setter
    def selected(self, mask):
        self.selected_array.tensor[:self._G] = torch.where(mask.reshape((self._G, 1)), self.group_ids, self._G)
        if self.selection_flag is not None:
            setattr(self, self.selection_flag, mask)

    @property
    def sensory_input_type(self):
        return self._tensor[self._rows.sensory_input_type.index, :]

    @sensory_input_type.setter
    def sensory_input_type(self, v):
        self._tensor[self._rows.sensory_input_type.index, :] = v

    @property
    def b_thalamic_input(self):
        return self._tensor[self._rows.b_thalamic_input.index, :]

    @b_thalamic_input.setter
    def b_thalamic_input(self, v):
        self._tensor[self._rows.b_thalamic_input.index, :] = v

    @property
    def b_sensory_group(self):
        return self._tensor[self._rows.b_sensory_group.index, :]

    @b_sensory_group.setter
    def b_sensory_group(self, v):
        self._tensor[self._rows.b_sensory_group.index, :] = v

    @property
    def b_output_group(self):
        return self._tensor[self._rows.b_output_group.index, :]

    @b_output_group.setter
    def b_output_group(self, v):
        self._tensor[self._rows.b_output_group.index, :] = v

    @property
    def output_type(self):
        return self._tensor[self._rows.output_type.index, :]

    @output_type.setter
    def output_type(self, v):
        self._tensor[self._rows.output_type.index, :] = v

    @property
    def b_sensory_input(self):
        return self._tensor[self._rows.b_sensory_input.index, :]

    @b_sensory_input.setter
    def b_sensory_input(self, v):
        self._tensor[self._rows.b_sensory_input.index, :] = v

    @property
    def b_monitor_group_firing_count(self):
        return self._tensor[self._rows.b_monitor_group_firing_count.index, :]

    @b_monitor_group_firing_count.setter
    def b_monitor_group_firing_count(self, v):
        self._tensor[self._rows.b_monitor_group_firing_count.index, :] = v


class LocationGroupProperties(PropertyTensor):

    @dataclass(frozen=True)
    class Rows:

        thalamic_inh_input_current: int = 0
        thalamic_exc_input_current: int = 1
        sensory_input_current0: int = 2
        sensory_input_current1: int = 3

        def __len__(self):
            return 4

    def __init__(self, n_groups, device, config, grid: NetworkGrid):
        self._rows = self.Rows()
        self._G = n_groups
        shape = (len(self._rows), n_groups)
        super().__init__(shape)

        self.input_face_colors: Optional[torch.Tensor] = None
        self.output_face_colors: Optional[torch.Tensor] = None

        self.set_tensor(shape, device, config, grid)

        self.spin_box_sliders = Sliders(self.Rows())

    def set_tensor(self, shape, device, config: NetworkConfig, grid: NetworkGrid):
        self.tensor = torch.zeros(shape, dtype=torch.float32, device=device)

        self.thalamic_inh_input_current = config.InitValues.ThalamicInput.inh_current
        self.thalamic_exc_input_current = config.InitValues.ThalamicInput.exc_current
        self.sensory_input_current0 = config.InitValues.SensoryInput.input_current0
        self.sensory_input_current1 = config.InitValues.SensoryInput.input_current1

    @property
    def thalamic_inh_input_current(self):
        return self._tensor[self._rows.thalamic_inh_input_current, :]

    @thalamic_inh_input_current.setter
    def thalamic_inh_input_current(self, v):
        self._tensor[self._rows.thalamic_inh_input_current, :] = v

    @property
    def thalamic_exc_input_current(self):
        return self._tensor[self._rows.thalamic_exc_input_current, :]

    @thalamic_exc_input_current.setter
    def thalamic_exc_input_current(self, v):
        self._tensor[self._rows.thalamic_exc_input_current, :] = v

    @property
    def sensory_input_current0(self):
        return self._tensor[self._rows.sensory_input_current0, :]

    @sensory_input_current0.setter
    def sensory_input_current0(self, v):
        self._tensor[self._rows.sensory_input_current0, :] = v

    @property
    def sensory_input_current1(self):
        return self._tensor[self._rows.sensory_input_current1, :]

    @sensory_input_current1.setter
    def sensory_input_current1(self, v):
        self._tensor[self._rows.sensory_input_current1, :] = v


class G2GInfoArrays(GPUArrayCollection):

    float_arrays_list = [
        'G_distance',
        'G_avg_weight_inh',
        'G_avg_weight_exc',
    ]

    int_arrays_list = [
        'G_delay_distance',
        'G_stdp_config0',
        'G_stdp_config1',
        'G_syn_count_inh',
        'G_syn_count_exc',
    ]

    def __init__(self, network_config: NetworkConfig, group_ids, G_flags: LocationGroupFlags,
                 G_pos,
                 device, bprint_allocated_memory):
        super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)

        self._config: NetworkConfig = network_config

        self.group_ids: torch.Tensor = group_ids
        self.G_flags: LocationGroupFlags = G_flags

        self.G_distance, self.G_delay_distance = self._G_delay_distance(network_config, G_pos)

        self.G_stdp_config0 = self.izeros(self.shape)
        self.G_stdp_config1 = self.izeros(self.shape)

        self.G_avg_weight_inh = self.fzeros(self.shape)
        self.G_avg_weight_exc = self.fzeros(self.shape)
        self.G_syn_count_inh = self.izeros(self.shape)
        self.G_syn_count_exc = self.izeros(self.shape)

    @property
    def shape(self):
        return self.G_distance.shape

    # noinspection PyPep8Naming
    @staticmethod
    def _G_delay_distance(network_config: NetworkConfig, G_pos: RegisteredVBO):
        G_pos_distance = torch.cdist(G_pos.tensor[: -1], G_pos.tensor[:-1])
        return G_pos_distance, ((network_config.D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

    def _stdp_distance_based_config(self, target_group, anti_target_group, target_config: torch.Tensor):
        distance_to_target_group = self.G_distance[:, target_group]
        distance_to_anti_target_group = self.G_distance[:, anti_target_group]

        xx0 = distance_to_target_group.reshape(self._config.G, 1).repeat(1, self._config.G)
        xx1 = distance_to_anti_target_group.reshape(self._config.G, 1).repeat(1, self._config.G)

        mask0 = xx0 < distance_to_target_group
        mask1 = xx0 <= xx1

        mask = mask0 & mask1

        target_config[mask] = 1
        target_config[~mask] = -1
        target_config[(xx0 == distance_to_target_group) & mask1] = 0

    def set_active_output_groups(self, output_groups=None, ):
        if output_groups is None:
            output_groups = self.active_output_groups()
        assert len(output_groups) == 2
        output_group_types = self.G_flags.output_type[output_groups].type(torch.int64)

        group0 = output_groups[output_group_types == 0].item()
        group1 = output_groups[output_group_types == 1].item()

        self._stdp_distance_based_config(group0, anti_target_group=group1, target_config=self.G_stdp_config0)
        # noinspection PyUnusedLocal
        b = self.to_dataframe(self.G_stdp_config0)

        self._stdp_distance_based_config(group1, anti_target_group=group0, target_config=self.G_stdp_config1)

    def active_output_groups(self):
        return self.group_ids[self.G_flags.b_output_group.type(torch.bool)]
