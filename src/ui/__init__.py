from .window import EngineWindow
from PyQt6 import QtWidgets
import qdarktheme
from vispy.app import Application


class BackendApp:
    def __init__(self):
        self.qt = QtWidgets.QApplication([])
        self.qt.setStyleSheet(qdarktheme.load_stylesheet())

        self.vs = Application(backend_name='pyqt6')
