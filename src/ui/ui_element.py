from dataclasses import dataclass
import enum
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QIcon, QAction

from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget
)


@dataclass
class UIElement:
    name: Optional[str] = None
    icon_name: Optional[str] = None
    status_tip: Optional[str] = None
    checkable: bool = False
    parent: Optional[QtWidgets.QMainWindow] = None
    connects: Optional[Union[Callable, list[Callable]]] = None
    disabled: bool = False

    def _set_checkable(self, obj: Union[QAction, QPushButton]):
        if self.checkable is True:
            obj.setCheckable(True)

    def _set_disabled(self, obj: Union[QAction, QPushButton]):
        if self.disabled is True:
            obj.setDisabled(True)

    def _set_png_icon(self, obj: Union[QAction, QPushButton]):
        if self.icon_name is not None:
            name = self.icon_name
            if (not name.endswith('.png')) and (not name.endswith('.PNG')):
                name += '.png'
            path = str(Path(__file__).parent) + f'/icons/{name}'
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            obj.setIcon(QIcon(path))

    def _set_status_tip(self, obj: Union[QAction, QPushButton]):
        if self.status_tip is not None:
            obj.setStatusTip(self.status_tip)

    def set_std_icon(self, obj: Union[QAction, QPushButton]):
        pixmapi = getattr(QStyle.StandardPixmap, self.icon_name)
        icon = self.parent.style().standardIcon(pixmapi)
        obj.setIcon(icon)

    def _init(self, obj):
        self._set_png_icon(obj)
        self._set_status_tip(obj)
        self._set_checkable(obj)
        self._set_disabled(obj)


@dataclass
class ButtonMenuAction(UIElement):
    menu_name: Optional[str] = None
    menu_short_cut: Optional[str] = None
    _action = None
    _button = None

    def __post_init__(self):
        if (self.connects is not None) and (not isinstance(self.connects, list)):
            self.connects = [self.connects]

    def _set_menu_short_cut(self, obj: Union[QAction, QPushButton]):
        if self.menu_short_cut is not None:
            obj.setShortcut(self.menu_short_cut)

    def action(self):
        if self._action is None:
            self._action = QAction(self.menu_name, self.parent)
            self._set_menu_short_cut(self._action)
            self._init(self._action)
            if self.connects is not None:
                for callable_ in self.connects:
                    self._action.triggered.connect(callable_)
        return self._action

    def button(self):
        if self._button is None:
            self._button = QPushButton(self.name, self.parent)
            self._init(self._button)
            if self.connects is not None:
                for callable_ in self.connects:
                    self._button.clicked.connect(callable_)
        return self._button


# noinspection PyAttributeOutsideInit
@dataclass
class Slider(UIElement):
    prop_id: str = None

    min_value: Optional[int] = None
    max_value: Optional[int] = 10000
    type: Callable = float
    func_: Optional[Callable] = lambda x: float(x)/100.
    func_inv_: Optional[Callable] = lambda x: int(x * 100)
    orientation: enum.IntFlag = QtCore.Qt.Orientation.Horizontal

    def __post_init__(self):
        self.prop_tensor = None

        self.widget = QWidget()
        self.vbox = QVBoxLayout(self.widget)

        self.label = QLabel(self.name)
        self.line_edit = QLineEdit()

        self.slider = QSlider(self.orientation)

        if self.min_value is not None:
            self.slider.setMinimum(self.min_value)
        if self.max_value is not None:
            self.slider.setMaximum(self.max_value)

        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.line_edit)
        self.vbox.addWidget(self.slider)

        self.change_from_text = False
        self.change_from_slider = False

        self.previous_applied_value = None

    def func(self, v, *args, **kwargs):
        if self.func_ is None:
            return v
        return self.func_(v, *args, **kwargs)

    def func_inv(self, v, *args, **kwargs):
        if self.func_inv_ is None:
            return v
        return self.func_inv_(v, *args, **kwargs)

    @property
    def text_value(self):
        if self.line_edit.text() == '':
            return ''
        return self.type(self.line_edit.text())

    @text_value.setter
    def text_value(self, v):
        self.line_edit.setText(str(self.validate_line_edit_value(v)))

    def set_slider_value(self, v):
        if isinstance(v, str):
            if v == '':
                return
            v = self.validate_line_edit_value(v)
        self.slider.setValue(self.func_inv(v))

    @property
    def value(self):
        return self.func(self.slider.value())

    @value.setter
    def value(self, v):
        self.change_from_text = True
        self.change_from_slider = True
        self.text_value = v
        self.set_slider_value(v)

    def validate_line_edit_value(self, v):
        v = min(max(self.type(v), self.func(self.slider.minimum())), self.func(self.slider.maximum()))
        print('val:', v)
        return v

    def changed_slider(self):
        if self.change_from_text is False:
            value = self.value
            print('slider:', value)
            self.change_from_slider = True
            self.text_value = value
            self.previous_applied_value = value
            setattr(self.prop_tensor, self.prop_id, value)
        else:
            self.change_from_text = False

    def changed_text(self):
        value = self.text_value
        print('text:', value)
        if (self.change_from_slider is False) and (value != ''):
            self.text_value = value
            self.change_from_text = True
            self.set_slider_value(value)
            self.previous_applied_value = value
            setattr(self.prop_tensor, self.prop_id, value)
        elif value == '':
            self.text_value = self.previous_applied_value
        self.change_from_slider = False

    def connect(self, prop_tensor, value):
        self.prop_tensor = prop_tensor
        self.value = value
        self.previous_applied_value = value
        self.change_from_text = False
        self.change_from_slider = False
        self.slider.sliderReleased.connect(self.changed_slider)
        self.line_edit.returnPressed.connect(self.changed_text)

