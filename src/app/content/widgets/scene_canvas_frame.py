from dataclasses import dataclass, asdict
from typing import Optional, Union

from PyQt6.QtWidgets import QFrame, QVBoxLayout
from vispy import scene
from vispy.app import Canvas
from vispy.color import Color
from vispy.gloo import GLContext

from vispy.scene import SceneCanvas, Widget


class SceneCanvasFrame(QFrame):

    def __init__(self, parent, canvas: SceneCanvas):
        super().__init__(parent)
        self.canvas = canvas
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLayout(QVBoxLayout(self))
        self.layout().addWidget(canvas.native)


@dataclass
class CanvasConfig:
    title: str = 'VisPy canvas'
    size: tuple = (1600, 1200)
    position: Optional[tuple] = None
    show: bool = False
    autoswap: bool = True

    create_native: bool = True
    vsync: bool = False
    resizable: bool = True
    decorate: bool = True
    fullscreen: bool = False
    config: Optional[dict] = None
    shared = Optional[Union[Canvas, GLContext]]
    keys: Optional[Union[str, dict]] = 'interactive'
    parent: Optional = None
    dpi: Optional[float] = None
    always_on_top: bool = False
    px_scale: int = 1
    bgcolor: Union[str, Color] = 'black'


class TextTableWidget(Widget):

    def __init__(self, labels: list[str], heights_min=None, heights_max=None,
                 height_min_global=None, height_max_global=None):

        super().__init__()

        self.unfreeze()
        self.item_count = 0
        self.grid = self.add_grid()
        width = 130
        self.width_min = width
        self.width_max = width

        if height_min_global is None:
            generate_height_min_global = True
            height_min_global = 0
            if height_max_global is None:
                height_min_default = 25
            else:
                height_min_default = int(height_max_global / len(labels))
        else:
            generate_height_min_global = False
            height_min_default = int(height_min_global / len(labels))

        if height_max_global is None:
            generate_height_max_global = True
            height_max_global = 0
            height_max_default = 25
        else:
            generate_height_max_global = False
            height_max_default = int(height_max_global / len(labels)) + 1

        for i, label in enumerate(labels):
            height_min = heights_min[i] if heights_min is not None else height_min_default
            f = label.count('_')
            height_min += f * height_min_default
            height_max = heights_max[i] if heights_max is not None else height_max_default
            height_max += f * height_max_default
            if generate_height_min_global is True:
                height_min_global += height_min
            if generate_height_max_global is True:
                height_max_global += height_max
            self.add_label(label, height_min=height_min, height_max=height_max)

        self.grid.height_min = (min(height_min_global, height_max_global)
                                if generate_height_min_global is True else height_min_global)
        self.grid.height_max = height_max_global

        # self.height_max = height_max_global
        # if width_min_global is not None:
        #     self.grid.width_min = width_min_global

        self.freeze()

    # noinspection PyTypeChecker
    def add_label(self, label_name, initial_value='0', height_min=28, height_max=28):
        font_size = 9
        label = scene.Label(label_name.replace('_', '\n'), color='white', font_size=font_size)
        label.border_color = 'w'
        label_value = scene.Label(initial_value, color='white', font_size=font_size)
        label_value.border_color = 'w'
        label.height_min = height_min
        label.height_max = height_max
        label_value.height_max = height_max
        self.grid.add_widget(label, row=self.item_count, col=0)
        self.grid.add_widget(label_value, row=self.item_count, col=1)
        self.item_count += 1
        setattr(self, label_name.replace('\n', '_'), label_value)


class BaseEngineSceneCanvas(scene.SceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app):  # Optional[Application]):

        conf = conf or CanvasConfig()
        super().__init__(**asdict(conf), app=app)

        self.central_widget.margin = 5

    # def create_native(self):
    #     """Create the native widget if not already done so. If the widget
    #     is already created, this function does nothing.
    #     """
    #     if self._backend is not None:
    #         return
    #     # Make sure that the app is active
    #     assert self._app.native
    #     # Instantiate the backend with the right class
    #     self.canvas_backend = self._app.backend_module.CanvasBackend(self, **self._backend_kwargs)
    #     # self._backend = set by BaseCanvasBackend
    #     self._backend_kwargs = None  # Clean up
    #
    #     # Connect to draw event (append to the end)
    #     # Process GLIR commands at each paint event
    #     self.events.draw.connect(self.context.flush_commands, position='last')
    #     if self._autoswap:
    #         self.events.draw.connect((self, 'swap_buffers'),
    #                                  ref=True, position='last')
