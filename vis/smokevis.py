import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import Qt

class SmokeItem:
    def __init__(self, area, panel_center, color=None):
        # 表示烟雾的数据结构为tuple ([x_smoke,y_smoke],timelast,radius,desity)
        self.smoke_pos = area[0]
        self.smoke_time_last = area[1]
        self.smoke_area_radius = area[2]
        self.panel_center = panel_center
        self.color = color

    def draw(self, painter: QtGui.QPainter):
        x = int(self.smoke_pos[0] + self.panel_center[0])
        y = int(self.smoke_pos[1] + self.panel_center[1])
        color_with_alpha = QtGui.QColor(200,200,200, alpha= 64)
        painter.setBrush(QtGui.QBrush(color_with_alpha))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QtCore.QRectF(x - self.smoke_area_radius/2, y - self.smoke_area_radius/2, self.smoke_area_radius, self.smoke_area_radius))
