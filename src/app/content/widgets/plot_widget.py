from dataclasses import dataclass, asdict
from typing import Optional, Union

import numpy as np
from vispy import scene
from vispy.scene import Widget, PanZoomCamera


@dataclass
class AxisVisualConfig:

    scale: int = 1
    offset: int = 0
    pos: Optional[np.array] = None
    domain: tuple = (0., 1.)
    scale_type: str = "linear"
    axis_color: tuple = (1, 1, 1)
    tick_color: tuple = (0.7, 0.7, 0.7)
    text_color: str = 'w',
    minor_tick_length: int = 5
    major_tick_length: int = 10
    tick_width: int = 2
    tick_label_margin: int = 12
    tick_font_size: int = 8
    axis_width: int = 3
    axis_label: Optional[str] = None
    axis_label_margin: int = 35
    axis_font_size: int = 10
    font_size: Optional[int] = None
    anchors: Optional[Union[tuple, list]] = None


class CustomLinkedAxisWidget(scene.AxisWidget):

    def __init__(self, scale=1, offset=0, orientation='left', **kwargs):
        super().__init__(orientation, **kwargs)
        self.unfreeze()
        self.scale = scale
        self.offset = offset
        self.freeze()

    def _view_changed(self, event=None):
        tr = self.node_transform(self._linked_view.scene)
        p1, p2 = tr.map(self._axis_ends())
        if self.orientation in ('left', 'right'):
            self.axis.domain = (self.scale * p1[1] + self.offset, self.scale * p2[1] + self.offset)
        else:
            self.axis.domain = (self.scale * p1[0] + self.offset, self.scale * p2[0] + self.offset)


class PlotWidget(Widget):

    # noinspection PyTypeChecker
    def __init__(self,
                 title: Optional[str], plot_height: int, plot_length: int, cam_yscale: int = 1,
                 width_min=100, width_max=None,
                 height_min=100, height_max=None,
                 title_color='white',
                 title_height_min=30,
                 title_height_max=30,
                 x_axis_height_min=30,
                 x_axis_height_max=30,
                 y_axis_width_min=50,
                 y_axis_width_max=50,
                 x_axis_config=None,
                 y_axis_config=None,
                 ):

        view_row_span, view_col_span = 1, 1

        super().__init__()

        self.unfreeze()

        row, col = 0, 0
        self.grid = self.add_grid()

        if x_axis_config is None:
            x_axis_config = {}
        if y_axis_config is None:
            y_axis_config = {}
        if isinstance(x_axis_config, AxisVisualConfig):
            x_axis_config = asdict(x_axis_config)
        if isinstance(y_axis_config, AxisVisualConfig):
            y_axis_config = asdict(y_axis_config)

        if title is not None:
            self.title = scene.Label(title, color=title_color)
            self.title.height_min = title_height_min
            self.title.height_max = title_height_max
            self.grid.add_widget(self.title, row=row, col=col + 1, col_span=view_col_span)
            plot_row = 1
        else:
            plot_row = 0

        yoffset = 0.05 * plot_height
        self.y_axis = CustomLinkedAxisWidget(orientation='left', **y_axis_config)
        self.y_axis.stretch = (0.12, 1)
        self.y_axis.width_min = y_axis_width_min
        self.y_axis.width_max = y_axis_width_max
        self.grid.add_widget(self.y_axis, row=plot_row, col=col, row_span=view_row_span)

        xoffset = 0.05 * plot_length
        self.x_axis = CustomLinkedAxisWidget(orientation='bottom', **x_axis_config)
        self.x_axis.stretch = (1, 0.15)
        self.x_axis.height_min = x_axis_height_min
        self.x_axis.height_max = x_axis_height_max
        self.grid.add_widget(self.x_axis, row=plot_row + view_row_span, col=col + 1, row_span=1, col_span=view_col_span)

        self.view = self.grid.add_view(row=plot_row, col=col + 1, border_color='w', row_span=view_row_span,
                                       col_span=view_col_span)
        if width_min is not None:
            self.width_min = width_min
        if width_max is not None:
            self.width_max = width_max
        if height_min is not None:
            self.height_min = height_min
        if height_max is not None:
            self.height_max = height_max

        scene.visuals.GridLines(parent=self.view.scene)

        self.view.camera = PanZoomCamera((-xoffset,
                                          -yoffset * cam_yscale,
                                          plot_length + xoffset + xoffset,
                                          (plot_height + yoffset + yoffset) * cam_yscale))
        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        self.freeze()

    @property
    def visible(self):
        return self.view.visible

    @visible.setter
    def visible(self, value):
        self.view.visible = value
        self.x_axis.visible = value
        self.y_axis.visible = value
        self.title.visible = value

    def add(self, widget):
        self.view.add(widget)

    def cam_reset(self):
        self.view.camera.zoom(1)
        self.view.camera.reset()
        self.y_axis._view_changed()
        self.update()
