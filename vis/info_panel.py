from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QComboBox
from PyQt5.QtCore import Qt
import pyqtgraph as pg

class InfoPanelWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # === 1. 固定显示的静态标签 (Step 进度条) ===
        self.step_container = QWidget()
        self.step_layout = QHBoxLayout(self.step_container)
        self.step_layout.setContentsMargins(0, 0, 0, 0)
        
        self.step_label = QLabel("Step:")
        self.step_bar = QProgressBar()
        self.step_bar.setRange(0, 500)
        self.step_bar.setValue(0)
        self.step_bar.setFormat("%v / 500")
        
        self.step_layout.addWidget(self.step_label)
        self.step_layout.addWidget(self.step_bar)
        self.layout.addWidget(self.step_container)

        self.layout.addStretch()

        # === 2. 微调2：新增下拉选择框 ===
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(["agent.v", "agent.w"])
        # 绑定下拉框切换事件
        self.plot_selector.currentTextChanged.connect(self._on_target_changed)
        self.layout.addWidget(self.plot_selector)

        # === 3. 微调3：纯白背景 (必须在生成 PlotWidget 前设置) ===
        pg.setConfigOptions(antialias=True, background='w', foreground='k')

        # === 4. 微调1：绘图窗口与高度限制 ===
        self.plot_widget = pg.PlotWidget(title="Agent Velocity (v)")
        self.plot_widget.setLabel('left', 'Velocity (v)')
        self.plot_widget.setLabel('bottom', 'Step')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        # 将图窗纵向挤扁，高度可以根据你的喜好修改 (例如 150 到 250 之间)
        self.plot_widget.setFixedHeight(200) 
        self.layout.addWidget(self.plot_widget)

        self.agent_ui_map = {}
        # 数据字典结构增加 'w'
        self.plot_data_map = {}  

    def _on_target_changed(self, text):
        """当下拉框切换时，动态修改图表的标题和 Y 轴标签"""
        if text == "agent.v":
            self.plot_widget.setTitle("Agent Velocity (v)")
            self.plot_widget.setLabel('left', 'Velocity (v)')
        else:
            self.plot_widget.setTitle("Agent Angular Velocity (w)")
            self.plot_widget.setLabel('left', 'Velocity (w)')
        # 注意：这里不需要手动刷新图表，下一次 step tick 过来时会自动绘制新数据

    def update_info(self, info_dict):
        current_step = info_dict['step']
        self.step_bar.setValue(current_step)

        max_x = max(20, current_step)
        min_x = max_x - 20
        self.plot_widget.setXRange(min_x, max_x, padding=0)

        # 获取当前下拉框选中的是要画哪个数据
        current_plot_target = self.plot_selector.currentText()
        current_ids = set()

        for agent in info_dict['agents']:
            a_id = agent['id']
            current_ids.add(a_id)
            
            # --- 处理 UI 面板 (Label + attk进度条) ---
            if a_id not in self.agent_ui_map:
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(0, 2, 0, 2)
                
                lbl = QLabel()
                lbl.setFixedWidth(120)
                
                bar = QProgressBar()
                bar.setRange(0, 10)
                bar.setFormat("%v")
                bar.setFixedHeight(15)
                
                h_layout.addWidget(lbl)
                h_layout.addWidget(bar)
                
                # 插入位置：目前倒数第1是plot_widget，倒数第2是QComboBox，倒数第3是弹簧
                # 为了插在弹簧前面，用 count() - 3
                self.layout.insertWidget(self.layout.count() - 3, container)
                self.agent_ui_map[a_id] = {'widget': container, 'label': lbl, 'bar': bar}

            ui = self.agent_ui_map[a_id]
            
            if agent['alive']:
                ui['label'].setText(f"Agent {a_id}")
                ui['bar'].setValue(agent['attk'])
                ui['label'].setStyleSheet("color: black;")
                ui['bar'].setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            else:
                ui['label'].setText(f"Agent {a_id} [DISABLED]")
                ui['bar'].setValue(agent['attk'])
                ui['label'].setStyleSheet("color: gray;")
                ui['bar'].setStyleSheet("QProgressBar::chunk { background-color: gray; }")

            # --- 处理画图逻辑 ---
            if a_id not in self.plot_data_map:
                color = pg.intColor(a_id, hues=9)
                curve = self.plot_widget.plot(pen=pg.mkPen(color, width=2))
                text_item = pg.TextItem(f"ID:{a_id}", color=color)
                self.plot_widget.addItem(text_item)
                
                self.plot_data_map[a_id] = {
                    'steps': [], 
                    'v': [], 
                    'w': [],  # 新增 w 的缓存
                    'curve': curve, 
                    'text': text_item
                }

            p_data = self.plot_data_map[a_id]
            p_data['steps'].append(current_step)
            # 后台同时记录 v 和 w
            p_data['v'].append(agent['v'])
            p_data['w'].append(agent['w'])

            if len(p_data['steps']) > 20:
                p_data['steps'].pop(0)
                p_data['v'].pop(0)
                p_data['w'].pop(0)

            # 根据下拉框的选择，决定提取哪组数据去画图
            y_data = p_data['v'] if current_plot_target == "agent.v" else p_data['w']

            # 更新曲线和末端标注
            p_data['curve'].setData(p_data['steps'], y_data)
            p_data['text'].setPos(p_data['steps'][-1], y_data[-1])

        # 清理已消失的 Agent
        existing_ids = list(self.agent_ui_map.keys())
        for old_id in existing_ids:
            if old_id not in current_ids:
                ui = self.agent_ui_map.pop(old_id)
                self.layout.removeWidget(ui['widget'])
                ui['widget'].deleteLater()
                
                if old_id in self.plot_data_map:
                    p_data = self.plot_data_map.pop(old_id)
                    self.plot_widget.removeItem(p_data['curve'])
                    self.plot_widget.removeItem(p_data['text'])