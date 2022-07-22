from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout

from app.content.widgets.collapsible_widget.collapsible_widget import CollapsibleWidget
from app.content.widgets.spin_box_sliders import SpinBoxSlider, SubCollapsibleFrame
from rendering import RenderedObjectNode


class RenderedObjectPropertyFrame(SubCollapsibleFrame):

    class Sliders:
        x: SpinBoxSlider = None
        y: SpinBoxSlider = None
        z: SpinBoxSlider = None

    def __init__(self, parent, window, obj: RenderedObjectNode,
                 prop_id: str, label=None,
                 min_value: Optional[int] = None,
                 max_value=10):

        super().__init__(parent)

        if label is None:
            label = prop_id
            label = label[0].upper() + label[1:]
        self.layout().addWidget(QLabel(label))

        self.sliders = self.Sliders()

        sliders_widget = QWidget()

        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)
        slider_names = ('x', 'y', 'z')
        for i in slider_names:

            sbs = SpinBoxSlider(name=i + ':',
                                window=window,
                                _min_value=min_value,
                                _max_value=max_value,
                                boxlayout_orientation=QtCore.Qt.Orientation.Horizontal,
                                status_tip=f"{obj.name}.{prop_id}.{i}",
                                prop_id=i,
                                single_step_spin_box=0.01,
                                single_step_slider=10)
            setattr(self.sliders, i, sbs)
            # sbs.widget.setFixedHeight(35)
            sliders_layout.addWidget(sbs.widget)
            sbs.connect_property(getattr(obj, prop_id))

        max_height = 25 + 35 * len(slider_names)
        self.setFixedHeight(max_height)
        sliders_widget.setMaximumHeight(max_height-5)

        self.layout().addWidget(sliders_widget)


class RenderedObjectCollapsible(CollapsibleWidget):

    def __init__(self, obj: RenderedObjectNode, window, parent=None):

        super().__init__(parent=parent, title=obj.name)

        self.scale = RenderedObjectPropertyFrame(parent=self, window=window, obj=obj, prop_id='scale')
        self.translate = RenderedObjectPropertyFrame(parent=self, window=window, obj=obj, prop_id='translate')
        self.add(self.scale)
        self.add(self.translate)
