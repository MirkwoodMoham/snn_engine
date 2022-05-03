from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
import torch
from typing import Optional

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

    def __init__(self, tensor: Optional[torch.Tensor] = None):
        self._tensor: Optional[torch.Tensor] = None
        if tensor is not None:
            self.tensor: Optional[torch.Tensor] = tensor

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

    def _row(self, row):
        return self._tensor[row, :]

    def _set_row(self, row, v):
        self._tensor[row, :] = v

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.tensor.cpu().numpy())


@dataclass(frozen=True)
class NeuronStatesRows:

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
class IzhikevichModel(PropertyTensor):

    @dataclass(frozen=True)
    class Rows(NeuronStatesRows):

        pt: int = 0
        u: int = 1
        v: int = 2
        a: int = 3
        b: int = 4
        c: int = 5
        d: int = 6
        i: int = 7

    def __init__(self, shape, device, types_tensor):
        super().__init__()
        self._N = shape[1]
        self.set_tensor(shape, device, types_tensor)

    def set_tensor(self, shape, device, types_tensor):

        self.tensor = torch.zeros(shape, dtype=torch.float32, device=device)

        mask_inh = types_tensor == NeuronTypes.INHIBITORY.value
        mask_exc = types_tensor == NeuronTypes.EXCITATORY.value
        r = torch.rand(self.N, dtype=torch.float32, device=device)

        self.pt = torch.rand(self.N, dtype=torch.float32, device=device)
        self.v = -65
        self.a = .02 + .08 * r * mask_inh
        self.b = .2 + .05 * (1. - r) * mask_inh
        self.c = -65 + 15 * (r ** 2) * mask_exc
        self.d = 2 * mask_inh + (8 - 6 * (r ** 2)) * mask_exc
        self.u = self.b * self.v

    @property
    def N(self):
        return self._N

    @property
    def pt(self):
        return self._row(self.rows.pt)

    @pt.setter
    def pt(self, v):
        self._set_row(self.rows.pt, v)

    @property
    def u(self):
        return self._row(self.rows.u)

    @u.setter
    def u(self, v):
        self._set_row(self.rows.u, v)

    @property
    def v(self):
        return self._row(self.rows.v)

    @v.setter
    def v(self, v):
        self._set_row(self.rows.v, v)

    @property
    def a(self):
        return self._row(self.rows.a)

    @a.setter
    def a(self, v):
        self._set_row(self.rows.a, v)

    @property
    def b(self):
        return self._row(self.rows.b)

    @b.setter
    def b(self, v):
        self._set_row(self.rows.b, v)

    @property
    def c(self):
        return self._row(self.rows.c)

    @c.setter
    def c(self, v):
        self._set_row(self.rows.c, v)

    @property
    def d(self):
        return self._row(self.rows.d)

    @d.setter
    def d(self, v):
        self._set_row(self.rows.d, v)

    @property
    def i(self):
        return self._row(self.rows.i)

    @i.setter
    def i(self, v):
        self._set_row(self.rows.i, v)


class Sliders:
    def __init__(self, rows):
        for k in asdict(rows):
            setattr(self, k, None)


class LocationGroupProperties(PropertyTensor):

    @dataclass(frozen=True)
    class Rows:

        sensory_input_type: int = 0
        thalamic_input: int = 1
        thalamic_inh_input_current: int = 2
        thalamic_exc_input_current: int = 3
        sensory_group: int = 4
        prop5: int = 5
        prop6: int = 6
        prop7: int = 7
        prop8: int = 8
        prop9: int = 9

        def __len__(self):
            return 10

    def __init__(self, shape, device, config, select_ibo):
        super().__init__()
        self._G = shape[1]
        nbytes = 4
        self.selected_array = RegisteredGPUArray.from_buffer(
            select_ibo, config=GPUArrayConfig(shape=(self._G, 1),
                                              strides=(2 * nbytes, nbytes),
                                              # strides=(nbytes, nbytes * (self._G + 1)),
                                              dtype=np.int32, device=device))
        # pd.options.display.max_columns = 29
        # print(self.selected_array.to_dataframe)
        # self.selected[:, :] = self._G
        # self.selected[0] = 2
        # self.selected.tensor[0, 1] = 0
        self.set_tensor(shape, device, config)
        self.selection_property = 'thalamic_input'

        self.spin_box_sliders = Sliders(self.Rows())

    def set_tensor(self, shape, device, config: NetworkConfig):
        self.tensor = torch.zeros(shape, dtype=torch.float32, device=device)
        thalamic_input_arr = torch.zeros(self._G)
        thalamic_input_arr[: int(self._G/2)] = 1
        self.thalamic_input = thalamic_input_arr
        # self.thalamic_input = 1
        self.thalamic_inh_input_current = config.DefaultValues.ThalamicInput.inh_current
        self.thalamic_exc_input_current = config.DefaultValues.ThalamicInput.exc_current

        self.sensory_group = torch.from_numpy(config.sensory_group_mask).to(device)

    @property
    def selected(self):
        return self.selected_array.tensor

    @selected.setter
    def selected(self, v):
        self.selected_array.tensor[:] = v
        if self.selection_property is not None:
            setattr(self, self.selection_property, v.flatten() != self._G)

    @property
    def sensory_input_type(self):
        return self._row(self.rows.sensory_input_type)

    @sensory_input_type.setter
    def sensory_input_type(self, v):
        self._set_row(self.rows.sensory_input_type, v)

    @property
    def thalamic_input(self):
        return self._row(self.rows.thalamic_input)

    @thalamic_input.setter
    def thalamic_input(self, v):
        self._set_row(self.rows.thalamic_input, v)

    @property
    def thalamic_inh_input_current(self):
        return self._row(self.rows.thalamic_inh_input_current)

    @thalamic_inh_input_current.setter
    def thalamic_inh_input_current(self, v):
        self._set_row(self.rows.thalamic_inh_input_current, v)

    @property
    def thalamic_exc_input_current(self):
        return self._row(self.rows.thalamic_exc_input_current)

    @thalamic_exc_input_current.setter
    def thalamic_exc_input_current(self, v):
        self._set_row(self.rows.thalamic_exc_input_current, v)

    @property
    def sensory_group(self):
        return self._row(self.rows.sensory_group)

    @sensory_group.setter
    def sensory_group(self, v):
        self._set_row(self.rows.sensory_group, v)
