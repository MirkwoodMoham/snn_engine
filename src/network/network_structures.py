from enum import Enum, unique

import numpy as np

from .gpu_arrays import (
    reshape_wrt_size,
    shape_size
)


@unique
class NeuronTypes(Enum):
    INHIBITORY = 1
    EXCITATORY = 2


class NetworkStructure:

    # noinspection PyPep8Naming
    def __init__(self, ID, struct_dict):
        if (ID in struct_dict) or ((len(struct_dict) > 0) and (ID < max(struct_dict))):
            raise AssertionError
        self.id = ID
        self._struct_dict = struct_dict

    def _post_init(self, verbose):
        self._struct_dict[self.id] = self
        if verbose is True:
            print('NEW:', self)


class NeuronTypeGroup(NetworkStructure):

    """
    Index-container for type-neuron-groups.
    """

    # noinspection PyPep8Naming
    def __init__(self, ID, start_idx, end_idx, S, neuron_type: NeuronTypes, group_dct, verbose=True):
        super().__init__(ID=ID, struct_dict=group_dct)
        self.ntype = neuron_type if isinstance(neuron_type, NeuronTypes) else NeuronTypes(neuron_type)
        self.start_idx = start_idx  # index of the first neuron of this group
        self.end_idx = end_idx  # index of the last neuron of this group
        self.S = S
        super()._post_init(verbose=verbose)

    @property
    def size(self):
        return self.end_idx - self.start_idx + 1

    def __len__(self):
        return self.size

    def __str__(self):
        return f'NeuronTypeGroup(type={self.ntype.name}, [{self.start_idx}, {self.end_idx}], id={self.id})'

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

    @staticmethod
    def validate(group_dct: dict, N):
        count = 0
        for g in group_dct.values():
            count += len(g)
        if count != N:
            raise ValueError


class NeuronTypeGroupConnection(NetworkStructure):

    # noinspection PyPep8Naming
    def __init__(self,
                 src: NeuronTypeGroup,
                 snk: NeuronTypeGroup,
                 w0: float,
                 S: int,
                 exp_syn_counts,
                 max_batch_size_mb: int,
                 conn_dict: dict,
                 verbose: bool = True):

        super().__init__(ID=(src.id, snk.id), struct_dict=conn_dict)

        self.src = src
        self.snk = snk
        self.w0 = w0
        self.S = S
        self.max_batch_size_mb = max_batch_size_mb

        self.conn_shape = (len(src), exp_syn_counts)

        if ((not isinstance(exp_syn_counts, int))
                or (self.conn_shape[0] < 1) or (self.conn_shape[1] < 1)):
            raise AssertionError

        self.batch_shape = reshape_wrt_size(self.conn_shape, max_batch_size_mb)

        if (not isinstance(self.batch_shape, tuple)
                or (len(self.batch_shape) != 2)
                or (self.batch_size('mb') > max_batch_size_mb)):
            raise ValueError

        self._col = self._set_col(conn_dict)

        super()._post_init(verbose=verbose)

    def __str__(self):
        types = self.type_name
        return f'NeuronTypeGroupConnection(types=({types[0]}, {types[1]}), id={self.id}, ' \
               f'shape={self.conn_shape})'

    def __len__(self):
        return self.conn_shape[1]

    def _set_col(self, conn_dict: dict):
        col = 0
        for gc in conn_dict.values():
            if gc.src_type_value == self.src_type_value:
                if gc.snk_type_value >= self.snk_type_value:
                    raise AssertionError
                col += len(gc)
        return col

    def batch_size(self, unit=None):
        return shape_size(self.batch_shape, unit=unit)

    @property
    def type_name(self):
        return self.src.ntype.name, self.snk.ntype.name

    @property
    def src_type_value(self):
        return self.src.ntype.value

    @property
    def snk_type_value(self):
        return self.snk.ntype.value

    @property
    def _row(self):
        return self.src.start_idx

    # noinspection PyPep8Naming
    @property
    def location(self):
        return self._row, self._col

    # noinspection PyPep8Naming,PyProtectedMember
    @staticmethod
    def validate(group_conn_dct: dict, S):
        syn_counts = {}
        cols = {}
        rows = {}
        for gc in group_conn_dct.values():
            src_id = gc.src.id
            if src_id not in syn_counts:
                syn_counts[src_id] = 0
                cols[src_id] = 0
                rows[src_id] = gc._row
            syn_counts[src_id] += len(gc)
            if not gc._col == cols[src_id]:
                raise AssertionError
            cols[src_id] += len(gc)
            if gc._row != rows[src_id]:
                raise AssertionError
        for i in syn_counts.values():
            if i != S:
                raise ValueError
