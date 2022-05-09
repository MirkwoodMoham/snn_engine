from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union

from .network_structures import NeuronTypes
from gpu import (
    RegisteredGPUArray,
    GPUArrayConfig
)
from network.network_config import NetworkConfig


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
        return self.tensor.data_ptr()

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.tensor.cpu().numpy())


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

    def __init__(self, shape, device, types_tensor):
        self._rows = self.Rows()
        super().__init__(shape)
        self._N = shape[1]
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


class LocationGroupProperties(PropertyTensor):

    @dataclass(frozen=True)
    class Rows:

        sensory_input_type: int = 0
        b_thalamic_input: int = 1
        thalamic_inh_input_current: int = 2
        thalamic_exc_input_current: int = 3
        b_sensory_group: int = 4
        b_output_group: int = 5
        output_type: int = 6
        b_sensory_input: int = 7
        sensory_input_current0: int = 8
        sensory_input_current1: int = 9

        def __len__(self):
            return 10

    def __init__(self, shape, device, config, select_ibo):
        self._rows = self.Rows()
        super().__init__(shape)
        self._G = shape[1]
        nbytes = 4
        self.selected_array = RegisteredGPUArray.from_buffer(
            select_ibo, config=GPUArrayConfig(shape=(self._G, 1),
                                              strides=(2 * nbytes, nbytes),
                                              dtype=np.int32, device=device))

        self.input_face_colors: Optional[torch.Tensor] = None
        self.output_face_colors: Optional[torch.Tensor] = None
        self.group_numbers_gpu: Optional[torch.Tensor] = None

        self.set_tensor(shape, device, config)
        self.selection_property = None

        self.spin_box_sliders = Sliders(self.Rows())

    def set_tensor(self, shape, device, config: NetworkConfig):
        self.tensor = torch.zeros(shape, dtype=torch.float32, device=device)
        thalamic_input_arr = torch.zeros(self._G)
        thalamic_input_arr[: int(self._G/2)] = 1
        # self.b_thalamic_input = thalamic_input_arr
        self.b_thalamic_input = 0
        # self.thalamic_input = 1
        self.thalamic_inh_input_current = config.InitValues.ThalamicInput.inh_current
        self.thalamic_exc_input_current = config.InitValues.ThalamicInput.exc_current
        self.sensory_input_current0 = config.InitValues.SensoryInput.input_current0
        self.sensory_input_current1 = config.InitValues.SensoryInput.input_current1

        self.b_sensory_group = torch.from_numpy(config.sensory_group_mask).to(device)
        self.sensory_input_type = -1.

        self.group_numbers_gpu = (torch.arange(self._G).to(device=device)
                                  .reshape((self._G, 1)))

    @property
    def selected(self):
        # noinspection PyUnresolvedReferences
        return (self.selected_array.tensor != self._G).flatten()

    @selected.setter
    def selected(self, mask):
        self.selected_array.tensor[:] = torch.where(mask.reshape((self._G, 1)),
                                                    self.group_numbers_gpu, self._G)
        if self.selection_property is not None:
            setattr(self, self.selection_property, mask)

    @property
    def sensory_input_type(self):
        return self._tensor[self._rows.sensory_input_type, :]

    @sensory_input_type.setter
    def sensory_input_type(self, v):
        self._tensor[self._rows.sensory_input_type, :] = v

    @property
    def b_thalamic_input(self):
        return self._tensor[self._rows.b_thalamic_input, :]

    @b_thalamic_input.setter
    def b_thalamic_input(self, v):
        self._tensor[self._rows.b_thalamic_input, :] = v

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
    def b_sensory_group(self):
        return self._tensor[self._rows.b_sensory_group, :]

    @b_sensory_group.setter
    def b_sensory_group(self, v):
        self._tensor[self._rows.b_sensory_group, :] = v

    @property
    def b_output_group(self):
        return self._tensor[self._rows.b_output_group, :]

    @b_output_group.setter
    def b_output_group(self, v):
        self._tensor[self._rows.b_output_group, :] = v

    @property
    def output_type(self):
        return self._tensor[self._rows.output_type, :]

    @output_type.setter
    def output_type(self, v):
        self._tensor[self._rows.output_type, :] = v

    @property
    def b_sensory_input(self):
        return self._tensor[self._rows.b_sensory_input, :]

    @b_sensory_input.setter
    def b_sensory_input(self, v):
        self._tensor[self._rows.b_sensory_input, :] = v

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
