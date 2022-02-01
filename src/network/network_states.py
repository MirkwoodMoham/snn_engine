from dataclasses import asdict, dataclass
import numpy as np
import torch
from typing import Optional

from .network_structures import NeuronTypes


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


class NeuronNetworkState:

    rows = NeuronStatesRows()

    def __init__(self, tensor: Optional[torch.Tensor] = None):

        self.has_tensor = False

        self._tensor: Optional[torch.Tensor] = None

        if tensor is not None:
            self.tensor: Optional[torch.Tensor] = tensor

    @classmethod
    def __len__(cls):
        return len(cls.rows)

    def __str__(self):
        return "NeuronNetworkState:\n" + str(self.tensor)

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, v):
        if self.has_tensor is False:
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


# noinspection PyPep8Naming
class IzhikevichModel(NeuronNetworkState):

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

    rows = Rows()

    def __init__(self, N, shape, device, types_tensor):
        super(IzhikevichModel, self).__init__()
        self._N = N
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

