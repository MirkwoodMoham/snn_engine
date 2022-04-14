from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget
)

from .collapsible_widget.collapsible_widget import CollapsibleWidget
from rendering import RenderedObject


@dataclass
class GUIElement:
    name: Optional[str] = None
    icon_name: Optional[str] = None
    status_tip: Optional[str] = None
    checkable: bool = False
    # parent: Optional[QtWidgets.QMainWindow] = None
    connects: Optional[Union[Callable, list[Callable]]] = None
    disabled: bool = False
    window: Optional[QtWidgets.QMainWindow] = None

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
        icon = self.window.style().standardIcon(pixmapi)
        obj.setIcon(icon)

    def _init(self, obj):
        self._set_png_icon(obj)
        self._set_status_tip(obj)
        self._set_checkable(obj)
        self._set_disabled(obj)


@dataclass
class ButtonMenuAction(GUIElement):
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
            self._action = QAction(self.menu_name, self.window)
            self._set_menu_short_cut(self._action)
            self._init(self._action)
            if self.connects is not None:
                for callable_ in self.connects:
                    self._action.triggered.connect(callable_)
        return self._action

    def button(self):
        if self._button is None:
            self._button = QPushButton(self.name, self.window)
            self._button.setMinimumHeight(28)
            self._init(self._button)
            if self.connects is not None:
                for callable_ in self.connects:
                    self._button.clicked.connect(callable_)
            self._button.clicked.connect(self.button_clicked)
        return self._button

    def button_clicked(self):
        sender = self.window.sender()
        # noinspection PyUnresolvedReferences
        msg = f'Clicked: {sender.text()}'
        self.window.statusBar().showMessage(msg)


class CustomQSlider(QSlider):
    def __init__(self, *arg,
                 ui_element: Optional[GUIElement] = None,
                 minimum: Optional[int] = None,
                 maximum: Optional[int] = None,
                 single_step: Optional[int] = 100):
        super().__init__(*arg)
        self.ui_element = ui_element
        self.wheel_func = None
        self.scroll_step = single_step
        self.mouse_function = None

        if minimum is not None:
            self.setMinimum(minimum)
        if maximum is not None:
            self.setMaximum(maximum)

    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:
        if self.wheel_func is not None:
            new_value = self.value() + (1 if e.angleDelta().y() > 0 else -1) * self.scroll_step
            # self.ui_element.change_from_scroll = True
            self.setValue(new_value)
            self.wheel_func(new_value, from_scroll=True)

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.ui_element.change_from_keypress = True
        super().keyPressEvent(a0)


class CustomQDoubleSpinBox(QDoubleSpinBox):

    def __init__(self,
                 parent=None,
                 ui_element: Optional[GUIElement] = None,
                 precision=2):
        super().__init__(parent)
        self.ui_element = ui_element
        self.wheel_func = None
        self.precision = precision

    def set_value(self):
        if self.wheel_func is not None:
            self.setValue(round(self.value(), self.precision))
            self.wheel_func(self.value(), from_scroll=True)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(e)
        opt = QtWidgets.QStyleOptionSpinBox()
        self.initStyleOption(opt)
        rect_up = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_SpinBox,
            opt,
            QtWidgets.QStyle.SubControl.SC_SpinBoxUp)
        if rect_up.contains(e.pos()):
            # print('UP')
            self.set_value()
        else:
            rect_down = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_SpinBox,
                opt,
                QtWidgets.QStyle.SubControl.SC_SpinBoxDown)
            if rect_down.contains(e.pos()):
                # print('DOWN')
                self.set_value()

    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:
        super().wheelEvent(e)
        self.set_value()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(a0)
        if a0.key() == QtCore.Qt.Key.Key_Enter:
            self.set_value()


# noinspection PyAttributeOutsideInit
@dataclass
class SpinBoxSlider(GUIElement):

    prop_id: str = None
    property_container: Any = None
    min_value: Optional[int] = None
    max_value: Optional[int] = 10000
    # type Callable = float
    func_: Optional[Callable] = lambda x: float(x)/100.
    func_inv_: Optional[Callable] = lambda x: int(x * 100)
    orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Horizontal
    boxlayout_orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Vertical
    maximum_width: Optional[int] = None
    fixed_width: Optional[int] = None
    single_step_spin_box: float = 0.1
    single_step_slider: Optional[int] = 100

    def __post_init__(self):
        self.property_container = None

        self.widget = QWidget()
        if self.status_tip is not None:
            self.widget.setStatusTip(self.status_tip)
        if self.maximum_width is not None:
            self.widget.setMaximumWidth(self.maximum_width)
        if self.fixed_width is not None:
            self.widget.setFixedWidth(self.fixed_width)

        if self.boxlayout_orientation == QtCore.Qt.Orientation.Vertical:
            self.layout_box = QVBoxLayout(self.widget)
            self.widget.setFixedHeight(84)
        else:
            self.layout_box = QHBoxLayout(self.widget)
            self.widget.setFixedHeight(28)

        self.label = QLabel(self.name)
        self.spin_box = CustomQDoubleSpinBox(ui_element=self)
        self.spin_box.setSingleStep(self.single_step_spin_box)

        # noinspection PyTypeChecker
        self.slider = CustomQSlider(self.orientation, ui_element=self,
                                    minimum=self.min_value, maximum=self.max_value,
                                    single_step=self.single_step_slider)

        self.layout_box.addWidget(self.label)
        self.layout_box.addWidget(self.spin_box)
        self.layout_box.addWidget(self.slider)

        self.change_from_text = False
        self.change_from_slider = False
        self.change_from_scroll = False
        self.change_from_key_press = False

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
        try:
            # return self.type(self.spin_box.value())
            return self.spin_box.value()
        except ValueError as err:
            self.window.statusBar().showMessage(str(err))
            return ''

    @text_value.setter
    def text_value(self, v):
        # self.line_edit.setText(str(self.validate_line_edit_value(v)))
        self.spin_box.setValue(self.validate_line_edit_value(v))

    def set_slider_value(self, v):
        if isinstance(v, str):
            v = self.validate_line_edit_value(v)
        self.slider.setValue(self.func_inv(v))
        self.change_from_scroll = False

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
        # return min(max(self.type(v), self.func(self.slider.minimum())), self.func(self.slider.maximum()))
        return min(max(v, self.func(self.slider.minimum())), self.func(self.slider.maximum()))

    def changed_slider(self, value_=None, from_scroll=False):
        if (self.change_from_text is False) or from_scroll:
            value = self.value if value_ is None else self.func(value_)

            self.change_from_slider = True
            self.change_from_scroll = from_scroll
            self.text_value = value
            self.previous_applied_value = value

            setattr(self.property_container, self.prop_id, value)
            self.change_from_slider = False

        self.change_from_text = False
        self.change_from_scroll = False

    def changed_spinbox(self, value_=None, from_scroll=False):
        # print('changed_line_edit', 'slider', self.change_from_slider, 'scroll', from_scroll)
        value = self.text_value if value_ is None else value_
        # print('line_edit:', value)
        if ((self.change_from_slider is False) or from_scroll) and (value != ''):
            self.text_value = value
            self.change_from_text = True
            self.change_from_scroll = from_scroll
            self.set_slider_value(value)
            self.previous_applied_value = value
            print(value)
            setattr(self.property_container, self.prop_id, value)
            self.change_from_text = False
        elif value == '':
            self.text_value = self.previous_applied_value
        self.change_from_slider = False
        self.change_from_scroll = False

    def new_spin_box_value(self, value):
        # print('next_line_edit', 'text', self.change_from_text, 'scroll', self.change_from_scroll)
        if ((self.change_from_scroll is False) and (self.change_from_text is False)
                and self.slider.isSliderDown() and (self.change_from_key_press is False)):
            # self.line_edit.setText(f"{self.previous_applied_value} -> {self.func(value)}")
            self.window.statusBar().showMessage(
                f"{self.status_tip}: "
                f"{self.previous_applied_value} -> {self.func(value)}")
            self.spin_box.setValue(self.func(value))
        elif not self.slider.isSliderDown() or self.change_from_key_press:
            # self.line_edit.setText(f"{self.func(value)}")
            self.spin_box.setValue(self.func(value))
            self.change_from_key_press = False

    # noinspection PyUnresolvedReferences
    def connect_property(self, property_container, value):
        self.property_container = property_container
        self.value = value
        self.previous_applied_value = value
        self.change_from_text = False
        self.change_from_slider = False
        self.slider.sliderReleased.connect(self.changed_slider)
        # self.slider.sliderPressed()
        self.slider.valueChanged[int].connect(self.new_spin_box_value)
        self.slider.wheel_func = self.changed_slider
        self.spin_box.wheel_func = self.changed_spinbox
        # self.line_edit.returnPressed.connect(self.changed_line_edit)
        self.spin_box.lineEdit().returnPressed.connect(self.changed_spinbox)


class RenderedObjectSliders(CollapsibleWidget):

    @dataclass
    class ScaleSliders:
        x: SpinBoxSlider = None
        y: SpinBoxSlider = None
        z: SpinBoxSlider = None
        a: SpinBoxSlider = None

    def __init__(self, obj: RenderedObject, window, parent=None, scale_sliders=None):

        super().__init__(parent=parent, title=obj.name)

        self.scale = self.ScaleSliders()

        scale_frame = QFrame(self)
        # scale_frame.setFixedHeight(450)
        scale_frame.setFixedWidth(450)
        # scale_frame.setMinimumHeight(135)
        # scale_frame.setFixedWidth(True)
        scale_layout = QHBoxLayout(scale_frame)
        # scale_layout.setSpacing(0)
        scale_layout.setContentsMargins(15, 0, 0, 0)
        scale_sliders_widget = QWidget()
        # scale_sliders_widget.setMinimumHeight(135)
        scale_sliders_widget.setMaximumHeight(135)
        scale_sliders_layout = QVBoxLayout(scale_sliders_widget)
        scale_sliders_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(QLabel('Scale'))

        sliders = []
        for i in ('x', 'y', 'z'):

            sbs = SpinBoxSlider(name=i + ':',
                                window=window,
                                boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                                status_tip=f"{obj.name}.scale.{i}",
                                prop_id=i,
                                single_step_spin_box=0.01,
                                single_step_slider=10)
            setattr(self.scale, i, sbs)
            sbs.widget.setFixedHeight(35)
            scale_sliders_layout.addWidget(sbs.widget)
            sliders.append(sbs)

            sbs.connect_property(
                obj.scale,
                getattr(obj.scale, i))

        # scale_frame.setMinimumHeight(25 + 35 * len(sliders))
        scale_frame.setFixedHeight(25 + 35 * len(sliders))

        scale_layout.addWidget(scale_sliders_widget)
        self.add(scale_frame)

    def connect(self):
        pass