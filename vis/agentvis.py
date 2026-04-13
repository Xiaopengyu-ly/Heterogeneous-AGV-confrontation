import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import Qt

class AgentItem:
    
    def __init__(self, agent_data, panel_center, color=None):
        self.agent_data = agent_data  # 来自 sim.get_render_data()
        self.panel_center = panel_center
        self.agent_size = 8
        self.attk_pos_size = 4
        self.color = color

    def draw(self, painter: QtGui.QPainter):
        # 1. 获取位置，如果为空则跳过
        pos = self.agent_data.get('position')
        if pos is None:
            return
        x = int(pos[0] + self.panel_center[0])
        y = int(pos[1] + self.panel_center[1])
        livestate = self.agent_data.get('disabled')
        if livestate is True:
            # === 4. 绘制本体 ===
            painter.setBrush(QtGui.QColor(QtGui.QColor(0,0,0, alpha= 64)))
            painter.setPen(QtGui.QPen(Qt.black, 1))
            painter.drawEllipse(QtCore.QRectF(x - self.agent_size/2, y - self.agent_size/2, self.agent_size, self.agent_size))
            # === 5. 绘制信息 ===
            self.draw_id(painter, x, y)
            return
        # === 2. 绘制航路点==
        # 获取航路点数据
        r_point = self.agent_data.get('rpoint')
        # 只有当航路点存在，且距离当前位置有一定距离时才绘制（避免原地乱画）
        if r_point is not None:
            dist = np.linalg.norm(r_point[0:1])
            if dist > 2.0: 
                rx = int(pos[0] + r_point[0] * np.cos(self.agent_data.get('angle')) - r_point[1] * np.sin(self.agent_data.get('angle')) + self.panel_center[0])
                ry = int(pos[1] + r_point[0] * np.sin(self.agent_data.get('angle')) + r_point[1] * np.cos(self.agent_data.get('angle')) + self.panel_center[1])
                # 绘制连线 (灰色虚线)
                # 这代表智能体当前 "想去" 的地方
                pen_line = QtGui.QPen(QtGui.QColor(150, 150, 150), 1, Qt.DashLine)                       
                painter.setPen(pen_line)
                painter.drawLine(x, y, rx, ry)
                # 绘制航路点标记 (小十字)
                pen_marker = QtGui.QPen(QtGui.QColor(self.color), 2)
                painter.setPen(pen_marker)
                ms = 4 # 标记大小
                painter.drawLine(rx - ms, ry, rx + ms, ry)
                painter.drawLine(rx, ry - ms, rx, ry + ms)
        # === 【新增】绘制编队占位点 (p_pos) ===
        p_pos = self.agent_data.get('p_pos')
        if p_pos is not None:
            # 计算屏幕坐标
            px = int(p_pos[0] + self.panel_center[0])
            py = int(p_pos[1] + self.panel_center[1])
            # 绘制占位点连线
            pen_line = QtGui.QPen(QtGui.QColor(100, 100, 100), 1, Qt.DashLine)                       
            painter.setPen(pen_line)
            painter.drawLine(x, y, px, py)
            # 绘制占位点十字标记
            pen_formation = QtGui.QPen(QtGui.QColor(0, 100, 255, 180), 1, Qt.DashLine)
            painter.setPen(pen_formation)
            ms = 6 # 标记大小
            painter.drawLine(px - ms, py, px + ms, py)
            painter.drawLine(px, py - ms, px, py + ms)
        # === 3. 绘制火炮和侦察朝向 ===
        painter.setPen(QtGui.QPen(Qt.black, 3))   # 火炮朝向
        WP_theta = self.agent_data.get('WPangle')
        painter.drawLine(x, y, x + int(20 * np.cos(WP_theta)), y + int(20 * np.sin(WP_theta)))
        # 扇形单位：1/16 度
        theta = int(self.agent_data.get('angle')* 180/np.pi)
        new_radius = 30  # 此半径不等于侦测半径，仅为可视化方便
        theta_span = int(self.agent_data.get('sense_angle') * 180 / np.pi)
        # 定义扇形的角度（单位：1/16 度）
        start_angle = (- theta - theta_span ) * 16 # 偏移与侦测角度范围保持一致
        span_angle  = (theta_span * 2) * 16    # 跨度为侦测角度范围2倍
        # 扇形所在的外接矩形（确保其几何中心位于 (x, y)，并且根据新的半径调整尺寸）
        rect = QtCore.QRectF(x - new_radius, y - new_radius, 2*new_radius, 2*new_radius)
        color_with_alpha = QtGui.QColor(0,255,0, alpha= 64)
        painter.setBrush(QtGui.QBrush(color_with_alpha))
        painter.setPen(Qt.NoPen)
        # 绘制扇形
        painter.drawPie(rect, start_angle, span_angle)
        # === 4. 绘制本体 ===
        painter.setBrush(QtGui.QColor(self.color))
        painter.setPen(QtGui.QPen(Qt.black, 1))
        painter.drawEllipse(QtCore.QRectF(x - self.agent_size/2, y - self.agent_size/2, self.agent_size, self.agent_size))
        # === 5. 绘制信息 ===
        self.draw_id(painter, x, y)

        # === 6. 绘制弹着点==
        attk_pos = self.agent_data.get('ATTKpos')
        if attk_pos is not None:
            attk_x = int(attk_pos[0] + self.panel_center[0])
            attk_y = int(attk_pos[1] + self.panel_center[1])
            painter.setBrush(Qt.black)
            painter.setPen(QtGui.QPen(Qt.black, 1))
            painter.drawEllipse(QtCore.QRectF(attk_x - self.attk_pos_size/2, attk_y - self.attk_pos_size/2, self.attk_pos_size, self.attk_pos_size))

    def draw_id(self, painter: QtGui.QPainter, x: int, y: int):
        painter.setPen(QtGui.QPen(Qt.black))
        font = QtGui.QFont("Arial", 12)
        font.setBold(True)
        painter.setFont(font)
        # 获取ID
        display_id = self.agent_data['id']
        painter.drawText(x + 8, y + 5, str(display_id))
