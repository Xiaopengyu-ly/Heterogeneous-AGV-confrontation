# vis/base_vis.py
import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt

from vis.agentvis import AgentItem
from vis.smokevis import SmokeItem

class VisualizationWindow(QtWidgets.QWidget):
    def __init__(self, simulation):
        super().__init__()
        # 兼容处理：如果是 Adapter，取出内部 Engine；如果是 Engine，直接用
        if hasattr(simulation, 'engine'):
            self.engine = simulation.engine
            self.env_adapter = simulation # 保留 Adapter 引用以便 step
        else:
            self.engine = simulation
            self.env_adapter = None

        self.grid_map = self.engine.grid_map
        self.grid_size = self.engine.grid_size

        height, width = self.grid_map.shape
        canvas_w = width * self.grid_size
        canvas_h = height * self.grid_size
        self.panel_center = np.array([canvas_w / 2, canvas_h / 2])

        self.setWindowTitle("Agent Simulation - Visualization")
        self.setGeometry(100, 100, canvas_w + 300, canvas_h)

        self.background_pixmap = None
        self.update_background()

        self.group_color = ["red", "blue"]
        self.smoke_color = str("gray")

        # 启动定时器（与 engine.dT 同步）
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(int(self.engine.dT * 1000))

    def update_background(self):
        height, width = self.grid_map.shape
        canvas_w = width * self.grid_size
        canvas_h = height * self.grid_size
        self.background_pixmap = QtGui.QPixmap(canvas_w, canvas_h)
        
        painter = QtGui.QPainter(self.background_pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing) 
        
        
        self.background_pixmap.fill(QtGui.QColor(255, 255, 255))

        # 1. 绘制海洋底色 (浅蓝色)
        self.background_pixmap.fill(QtGui.QColor(136, 207, 240))
        
        # 2. 生成蜿蜒的海岸线路径 (上方是沙滩陆地，下方是海)
        land_path = QtGui.QPainterPath()
        land_path.moveTo(0, 0)  # 陆地起点：左上角
        
        # 设定海岸线的基准高度 (这里设为画布的 40% 高度处)
        base_y = canvas_h * 0.4
        land_path.lineTo(0, base_y)
        import random
        import math
        # 用三角函数叠加来模拟自然蜿蜒的曲线
        # 同时保存曲线的 y 坐标，方便后面判断植物是否长在陆地上
        curve_y_coords = {} 
        for x in range(0, canvas_w + 5, 5):
            # 两个不同频率和振幅的正弦/余弦波叠加，显得更随机自然
            y = base_y + 20 * math.sin(x / 40.0) + 15 * math.cos(x / 90.0)
            land_path.lineTo(x, y)
            curve_y_coords[x] = y
            
        land_path.lineTo(canvas_w, 0)  # 连到右上角
        land_path.closeSubpath()       # 闭合路径 (回到左上角)，形成完整的陆地区域
        
        # 填充沙滩颜色
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(238, 214, 175))
        painter.drawPath(land_path)
        
        # (视效增强) 沿着海岸线画一条半透明白色的“海浪泡沫”
        foam_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 150), 3)
        painter.setPen(foam_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(land_path)
        painter.setPen(Qt.PenStyle.NoPen) # 恢复无笔触状态

        # 4. 最后绘制障碍物 (此时礁石会直接覆盖在海洋或沙滩的背景上)
        obstacle_coords = np.argwhere(self.grid_map == 1)
        if len(obstacle_coords) > 0:
            painter.setBrush(QtGui.QColor(74, 80, 84))
            for y, x in obstacle_coords:
                padding = 2
                rect_size = self.grid_size - padding * 2
                
                if rect_size <= 0:
                    painter.drawRect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size)
                else:
                    rect_x = x * self.grid_size + padding
                    rect_y = y * self.grid_size + padding
                    painter.drawRoundedRect(rect_x, rect_y, rect_size, rect_size, 4.0, 4.0)
                    
        painter.end()
        self.update()
        
    def update_simulation(self):
        # 简单定时刷新模式：
        # 如果有 adapter，调用 adapter.step (带空动作或默认动作)
        # 如果只有 engine，调用 engine.step_physics
        if self.env_adapter:
            # 这里的 action 需要是符合 Space 的，为了不报错，先给空
            # 注意：这只是为了 VisualizationWindow 独立运行时不崩溃
            # 实际上 ControlledWindow 会接管 Timer，这里主要是个兜底
            dummy_action = np.zeros(5, dtype=np.float32)
            self.env_adapter.step(dummy_action)
        else:
            self.engine.step_physics()
            
        self.update()  # 触发重绘

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if self.background_pixmap:
            painter.drawPixmap(0, 0, self.background_pixmap)

        # 获取最新渲染数据 (调用 Phase 1 物理引擎的新接口)
        render_data = self.engine.get_render_data()

        # 绘制智能体
        for agent_data in render_data['agents']:
            side = agent_data.get('side', 0)
            color = self.group_color[side] if side < len(self.group_color) else "gray"
            item = AgentItem(agent_data, self.panel_center, color=color)
            item.draw(painter)
        
        # 绘制烟雾区域
        env_data = render_data['env']
        smoke_areas = env_data.get('SmokeArea', 0)
        # 容错处理，防止 smoke 为 None 或格式不对
        if isinstance(smoke_areas, list):
            for area in smoke_areas:
                color = self.smoke_color
                item = SmokeItem(area, self.panel_center, color=color)
                item.draw(painter)