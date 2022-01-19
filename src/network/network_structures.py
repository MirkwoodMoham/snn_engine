from enum import Enum, unique


@unique
class NeuronTypes(Enum):
    INHIBITORY = 1
    EXCITATORY = 2


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
