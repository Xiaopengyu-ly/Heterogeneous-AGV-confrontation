from PyQt5.QtWidgets import QMainWindow, QDockWidget
from PyQt5.QtCore import Qt, QTimer

from vis.base_vis import VisualizationWindow
from vis.info_panel import InfoPanelWidget
from sim.sim_controller import SimulationController

class ControlledVisWindow(QMainWindow):
    def __init__(self, simulation, config=None):
        super().__init__()
        self.sim = simulation
        self.config = config or {}

        # 创建控制器（逻辑核心）
        self.controller = SimulationController(simulation, config)

        # 创建可视化主窗口
        self.vis_widget = VisualizationWindow(simulation)
        self.setCentralWidget(self.vis_widget)

        # 创建信息面板
        self.info_panel = InfoPanelWidget()
        self.info_dock = QDockWidget("Simulation Info", self)
        self.info_dock.setWidget(self.info_panel)
        self.info_dock.setFixedWidth(500)
        self.info_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)

        # 停止原定时器，用自己的
        self.vis_widget.timer.stop()

        # 启动新定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(int(self.controller.engine.dT * 1000))

        self.setWindowTitle("Agent Simulation - Modular")
        self.resize(self.vis_widget.width() + 230, self.vis_widget.height())

    def on_timer_tick(self):
        # 执行一步仿真
        continue_flag = self.controller.step()
        if not continue_flag:
            self.timer.stop()
            print("仿真结束")
            return

        # 更新 UI
        self.vis_widget.update()
        info = self.controller.get_info()
        self.info_panel.update_info(info)