__author__ = 'Caroline Beyne'

from PyQt6 import QtGui, QtCore, QtWidgets
from vispy.scene import SceneCanvas


class CollapsibleWidget(QtWidgets.QWidget):

    class Arrow(QtWidgets.QFrame):
        def __init__(self, parent=None, collapsed=False):
            super().__init__(parent=parent)
            self.setFixedSize(24, 24)
            # self.setMaximumSize(24, 24)

            # horizontal == 0
            self._arrow_horizontal = (QtCore.QPointF(7.0, 8.0),
                                      QtCore.QPointF(17.0, 8.0),
                                      QtCore.QPointF(12.0, 13.0))
            # vertical == 1
            self._arrow_vertical = (QtCore.QPointF(8.0, 7.0),
                                    QtCore.QPointF(13.0, 12.0),
                                    QtCore.QPointF(8.0, 17.0))
            # arrow
            self._arrow = None
            self.set_arrow(int(collapsed))

        def set_arrow(self, arrow_dir):
            if arrow_dir:
                self._arrow = self._arrow_vertical
            else:
                self._arrow = self._arrow_horizontal

        def paintEvent(self, event):
            painter = QtGui.QPainter()
            painter.begin(self)
            painter.setBrush(QtGui.QColor(192, 192, 192))
            painter.setPen(QtGui.QColor(64, 64, 64))
            painter.drawPolygon(*self._arrow)
            painter.end()

    class TitleFrame(QtWidgets.QFrame):
        def __init__(self, body, parent=None, title="", collapsed=False, min_collapsed_height=28):
            super().__init__(parent=parent)

            self.setFixedHeight(min_collapsed_height)
            self.move(QtCore.QPoint(20, 0))
            self.setStyleSheet("border:1px solid rgb(71, 71, 71); ")

            self._hlayout = QtWidgets.QHBoxLayout(self)
            self._hlayout.setContentsMargins(0, 0, 0, 0)
            # self._hlayout.setSpacing(0)

            self._arrow = CollapsibleWidget.Arrow(collapsed=collapsed)
            self._arrow.setStyleSheet("border:0px")

            self._title = QtWidgets.QLabel(title)
            self._title.setFixedHeight(24)
            # self._title.setMaximumHeight(24)
            self._title.move(QtCore.QPoint(20, 0))
            self._title.setMinimumWidth(100)
            self._title.setStyleSheet("border:0px")

            self._hlayout.addWidget(self._arrow)
            self._hlayout.addWidget(self._title)

            self.body = body

        def mousePressEvent(self, event: QtGui.QMouseEvent):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.body.toggle_collapsed()

            # return super(FrameLayout.TitleFrame, self).mousePressEvent(event)

    def __init__(self, parent=None, title=None, spacing=0):
        super().__init__(parent=parent)

        self._is_collapsed = True
        self._title_frame = self.TitleFrame(self, title=title, collapsed=self._is_collapsed)

        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout()
        self._content_layout.setContentsMargins(5, 0, 0, 0)
        self._content.setFixedHeight(0)
        self._content_layout.setSpacing(spacing)
        self._content.setLayout(self._content_layout)
        self._content.setVisible(not self._is_collapsed)
        self._main_v_layout = QtWidgets.QVBoxLayout(self)
        self._main_v_layout.addWidget(self._title_frame)
        self._main_v_layout.addWidget(self._content)
        self._main_v_layout.setContentsMargins(0, 0, 0, 0)
        self._main_v_layout.setSpacing(spacing)
        self.min_collapsed_height = 28
        self.setFixedHeight(self.min_collapsed_height)
        self.parent_collapsible = None
        self.children_collapsibles = []
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.setSizePolicy(sp)

    def children_height(self):
        height = 0
        for child in self.children_collapsibles:
            height += child.height()
        return height

    def _resize_parent_collapsible(self, expanding, content_height, height_additive):
        if expanding:
            factor = 1
        else:
            factor = -1
        self.parent_collapsible.setFixedHeight(
            self.parent_collapsible.height() + factor * (content_height - height_additive))
        self.parent_collapsible._content.setFixedHeight(
            self.parent_collapsible._content.height() + factor * (content_height - height_additive))
        if self.parent_collapsible.parent_collapsible is not None:
            self.parent_collapsible._resize_parent_collapsible(
                self._is_collapsed, content_height=content_height, height_additive=height_additive)

    def toggle_collapsed(self):
        self._content.setVisible(self._is_collapsed)

        height_additive = - 2
        content_height = self._content.height()

        if self._is_collapsed:
            self.setFixedHeight(self.min_collapsed_height + content_height - height_additive)

        if self.parent_collapsible is not None:
            self._resize_parent_collapsible(
                self._is_collapsed, content_height=content_height, height_additive=height_additive)

        if not self._is_collapsed:
            self.setFixedHeight(self.min_collapsed_height)

        self._is_collapsed = not self._is_collapsed
        self._title_frame._arrow.set_arrow(int(self._is_collapsed))

    def add(self, o):
        add = 0
        self._content.setFixedHeight(self._content.height() + o.height() + add)
        # print('content-height:', self._content.height())
        self._content_layout.addWidget(o)
        if isinstance(o, CollapsibleWidget):
            # self._content.setFixedHeight(self._content.height() + 3)
            o.parent_collapsible = self
            o.min_collapsed_height += add
            o.setFixedHeight(o.min_collapsed_height)
            self.children_collapsibles.append(o)
            o.toggle_collapsed()
            self.toggle_collapsed()
            self.toggle_collapsed()

    def _canvas_frame(self, canvas: SceneCanvas):
        frame = QtWidgets.QFrame(self)
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.addWidget(canvas.native)
        return frame


if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)

    win = QtWidgets.QMainWindow()
    w = QtWidgets.QWidget()
    w.setMinimumWidth(350)
    win.setCentralWidget(w)
    layout_ = QtWidgets.QVBoxLayout()
    layout_.setSpacing(0)
    layout_.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    w.setLayout(layout_)

    t = CollapsibleWidget(title="Buttons")
    t.add(QtWidgets.QPushButton('a'))
    t.add(QtWidgets.QPushButton('b'))
    t.add(QtWidgets.QPushButton('c'))

    f = CollapsibleWidget(title="TableWidget")
    rows, cols = (6, 3)
    data = {'col1': ['1', '2', '3', '4', '5', '6'],
            'col2': ['7', '8', '9', '10', '11', '12'],
            'col3': ['13', '14', '15', '16', '17', '18']}
    table = QtWidgets.QTableWidget(rows, cols)
    headers = []
    for n, key in enumerate(sorted(data.keys())):
        headers.append(key)
        for m, item in enumerate(data[key]):
            new_item = QtWidgets.QTableWidgetItem(item)
            table.setItem(m, n, new_item)
    table.setHorizontalHeaderLabels(headers)
    f.add(table)

    layout_.addWidget(t)
    layout_.addWidget(f)

    win.show()
    win.raise_()
    print("Finish")
    sys.exit(app.exec())
