from dataclasses import dataclass


@dataclass
class XYZ:
    x: object = None
    y: object = None
    z: object = None

    _allowed_keys = ('x', 'y', 'z', '_tuple')

    def __post_init__(self):
        self._tuple = (self.x, self.y, self.z)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._tuple[item]
        else:
            return getattr(self, item)

    def __setattr__(self, key, value):
        if key not in self._allowed_keys:
            raise KeyError(f"'{key}' not in {self._allowed_keys}")
        super().__setattr__(key, value)
        if key != '_tuple':
            super().__setattr__('_tuple', (self.x, self.y, self.z))
