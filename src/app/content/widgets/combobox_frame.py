from typing import Tuple

from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel


class CustomComboBox(QComboBox):

    def __init__(self, item_list=None, set_current=0):

        super().__init__()

        if item_list is not None:
            self.add_items(item_list, set_current)

        self.setFixedHeight(28)

    def add_items(self, item_list, set_current=1):
        for item in item_list:
            self.addItem(item)
        if set_current is not None:
            self.setCurrentIndex(set_current)


class ComboBoxFrame(QFrame):

    def __init__(self, name, actualize_button=None, parent=None, height=32, max_width=None, item_list=None,
                 set_current=0):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.combo_box = CustomComboBox(item_list, set_current)
        self.label = QLabel(name)
        self.label.setMaximumWidth(80)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.combo_box)
        if actualize_button is not None:
            self.actualize_button = actualize_button
            self.layout().addWidget(self.actualize_button)
        else:
            self.actualize_button = None

        self.setFixedHeight(height)

        if max_width is not None:
            self.setMaximumWidth(max_width)

    def __call__(self):
        return self.combo_box

    def connect(self, func):
        # noinspection PyUnresolvedReferences
        self.combo_box.currentTextChanged.connect(func)
        if self.actualize_button is not None:
            self.actualize_button.clicked.connect(func)


class DualComboBoxFrame(QFrame):

    def __init__(self, name, actualize_button=None, parent=None):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.combo_box0 = CustomComboBox()
        self.combo_box0.setMaximumWidth(50)
        self.combo_box0.setMaxVisibleItems(10)
        self.combo_box0.setEditable(True)
        self.combo_box1 = CustomComboBox()
        self.label = QLabel(name)
        self.label.setMaximumWidth(80)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.combo_box0)
        self.layout().addWidget(self.combo_box1)
        if actualize_button is not None:
            self.actualize_button = actualize_button
            self.layout().addWidget(self.actualize_button)
        else:
            self.actualize_button = None

        self.setFixedHeight(32)

    def connect(self, func):
        # noinspection PyUnresolvedReferences
        self.combo_box0.currentTextChanged.connect(func)
        # noinspection PyUnresolvedReferences
        self.combo_box1.currentTextChanged.connect(func)
        if self.actualize_button is not None:
            self.actualize_button.clicked.connect(func)


class GroupValidator(QValidator):

    def __init__(self, parent, group_ids):
        self.group_ids = group_ids
        super().__init__(parent)

    def validate(self, a0: str, a1: int) -> Tuple['QValidator.State', str, int]:
        if a0 in self.group_ids:
            state = QValidator.State.Acceptable
        else:
            state = QValidator.State.Invalid
        return state, a0, a1


class G2GInfoDualComboBoxFrame(DualComboBoxFrame):

    def __call__(self):
        return self.combo_box1

    def set_src_group_validator(self, groups_ids):
        v = GroupValidator(self, groups_ids)
        self.combo_box0.setValidator(v)

    def init_src_group_combo_box(self, group_ids):
        self.combo_box0.add_items(group_ids, 0)
        self.set_src_group_validator(group_ids)
