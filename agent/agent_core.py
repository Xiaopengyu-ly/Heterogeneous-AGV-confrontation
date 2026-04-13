# agent_core.py
import numpy as np
import math
from stable_baselines3 import PPO, SAC

# 导入所有新组件
from agent.agent_models import BehaviorSystem
from agent.agent_get import DataSystem
from agent.agent_check import CheckSystem
from agent.agent_comm import CommSystem
from agent.agent_loader import load_agent_config

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# 【重大修改】Agent 不再继承任何功能类
class Agent:
    def __init__(self, id: int, position: np.ndarray, velocity: np.ndarray, dT: float = 0.02, side=0,):
        # ... (配置加载部分保持不变) ...
        # === 从配置文件加载参数 ===
        config = load_agent_config(config_name = "default")

        # 任务参数
        self.TARGET_POS = np.array([0.0, 0.0])

        # 装备性能指标
        self.v_max = config["v_max"] 
        self.r_turn_min = config["r_turn_min"]
        self.s_max = config["s_max"]
        self.sense_field = config["sense_field"]
        self.sense_angle = config["sense_angle"]
        self.sense_variance = config["sense_variance"]
        self.attack_range = config["attack_range"]
        self.cannon_w_max = config["cannon_w_max"]
        self.launch_delay = int(config["launch_delay"])
        self.num_per_launch = config["num_per_launch"]
        self.attk_radius = config["attk_radius"]
        self.attk_variance = config["attk_variance"]
        self.cannon_capacity = config["cannon_capacity"]
        self.smoke_capacity = config["smoke_capacity"]
        self.reflective_surface = config["reflective_surface"]
        self.exposed_area = config["exposed_area"]
        self.decision_delay = config["decision_delay"]
        self.task_preference = config["task_preference"]
        self.task_assignment = config["task_assignment"] 
        self.weapon_assignment = config["weapon_assignment"]
        self.connect_dist = config["connect_dist"]
        # 装备性能参数
        self.L = 2.0
        self.R = 0.1
        self.sector_num = 36
        self.sector_center = [(np.pi / self.sector_num + 2* i * np.pi / self.sector_num) for i in range(self.sector_num)]
        self.v_min = 0.1
        self.miss_dist = 6
        self.send_period = int(2)
        self.sense_basic_P0 = 0.9
        self.sense_P_attenuation = 0.9
        self.smoke_attenuation = 0.1
        
        # 信息字典/数组初始化
        self.trajectory = []
        self.neighbors_id = []
        self.neighbors_info = {
            "weight": {}, "position": {}, "velo": {},
            "channelid": {}, "formdist": {}, 'obssector': {}
        }
        self.targets_id = 0
        self.targets_info = {
            "weight": {}, "position": {}, "velo": {}, "channelid": {}
        }
        self.cannon_targets_id = 0
        self.cannon_targets_info = {
            "weight": {}, "position": {}, "velo": {}, "channelid": {}
        }
        self.position_history = []
        self.local_obstacles = []
        self.obs_sector = []
        self.obs_v_sector = []
        self.smoke_zones = []
        
        # 状态信息
        self.id = id
        self.side = side
        self.t_pos = None
        self.attk_pos = None
        self.channel_id = None
        self.position = position.copy()
        dx = self.TARGET_POS[0] - self.position[0]
        dy = self.TARGET_POS[1] - self.position[1]
        self.theta = np.arctan2(dy, dx)
        self.cannon_theta = 0
        # 用于SAC/A*模式下局部导航的目标点，对接底层控制器
        self.r_point = None
        self.prev_r_point = None
        # 用于存储 HRL 模式下 MAPPO/Translator 分配的站位点
        self.p_pos = None
        self.velo = velocity.copy()
        self.v = 0
        self.dv = 0
        self.w = 0
        self.dw = 0
        self.cannon_w = 0
        self.cannon_remain = self.cannon_capacity
        self.smoke_remain = self.smoke_capacity
        self.v_left = None
        self.v_right = None
        self.planner = None
        self.grid_map = None
        self.down_sampled_map = None
        self.grid_size = 1
        
        # 策略模型
        # 原代码: self.SACmodel = SAC.load("sac_policy") 
        # 修改为: 不加载任何旧模型，直接设为 None
        self.SACmodel = None
        self.dT = dT
        
        # 计数标志位
        self.k = 0
        self.send_count = int(0)
        self.launch_delay_count = self.launch_delay
        self.rpoint_valid = False
        self.hit_rpoint = False
        self.rtPlanFlag = False
        self.disabled = False
        self.cannon_launched = False
        self.smoke_mission = True
        
        # === 【核心修改】实例化四大组件 ===
        self.comm_system = CommSystem(self)
        self.check_system = CheckSystem(self)
        self.data_system = DataSystem(self)
        self.behavior_system = BehaviorSystem(self)

    # ==========================================
    #               接口代理区 (Facade)
    # ==========================================
    
    # --- Check System ---
    def check_hit(self):
        self.check_system.check_hit()
    def check_env(self, env_feedback):
        self.check_system.check_env(env_feedback)
        
    # --- Data System ---
    def get_connect(self, pool):
        self.data_system.get_connect(pool)
    def get_init_parameters(self, channel_id, target_distance):
        self.data_system.get_init_parameters(channel_id, target_distance)
    def get_net_neighbors(self, neighbors_dict, channel_dict, formation_structure, targets_dict, cannon_targets_dict):
        self.data_system.get_net_neighbors(neighbors_dict, channel_dict, formation_structure, targets_dict, cannon_targets_dict)
    def get_trajectory(self):
        return self.data_system.get_trajectory()
    def get_grid_map(self, map_layers=None, grid_size=1):
        self.data_system.get_grid_map(map_layers, grid_size)
    def get_route_point(self, case="A_star", action=np.array([0,0,0,0,0])):
        return self.data_system.get_route_point(case, action)

    # --- Comm System ---
    def broadcast_msg(self, pool):
        self.comm_system.broadcast_msg(pool)
    def recieve_msg(self, pool):
        self.comm_system.recieve_msg(pool)
    def upload_toPanel(self, pool):
        self.comm_system.upload_toPanel(pool)

    # --- Behavior System ---
    def update_movement_model(self):
        # 逻辑：guidance -> kinematic -> integration
        self.behavior_system.move_logic_model()
        # self.behavior_system.Kinematic_model(self.v_left, self.v_right)
        
        # 积分逻辑保留在 Agent 本体，或者也可以移入 BehaviorSystem
        # 这里暂时保留在本体，因为涉及 self.position 的直接修改
        self.theta = self.theta + self.w * self.dT
        self.theta = normalize_angle(self.theta)
        self.velo = np.array([self.v * math.cos(self.theta), self.v * math.sin(self.theta)])
        self.position = self.position + self.velo * self.dT
        
        self.check_hit() # check 逻辑

    def update_sense_model(self):
        self.behavior_system.sense_model()
    
    def update_smoke_model(self):
        self.behavior_system.smoke_model()

    def update_task_allocate_model(self):
        self.behavior_system.task_allocate_model()

    def update_attack_model(self):
        self.behavior_system.attack_model()
        # 炮塔积分逻辑
        self.cannon_theta += self.cannon_w * self.dT
        self.cannon_theta = normalize_angle(self.cannon_theta)
        # self.cannon_theta = (self.cannon_theta + np.pi) % (2 * np.pi) - np.pi
        if self.launch_delay_count < self.launch_delay:
            self.launch_delay_count += 1

    def update_msg(self):
        if self.channel_id is not None:
            self.upload_toPanel(self.msg_pool)
        self.send_count += 1
        if self.send_count >= self.send_period:
            if self.msg_pool is not None:
                self.broadcast_msg(self.msg_pool)
                self.recieve_msg(self.msg_pool)
            self.send_count = 0

    def update_model(self):
        if self.disabled == True:
            return
        self.update_smoke_model()
        self.update_sense_model()
        self.update_task_allocate_model()
        self.update_attack_model()
        self.update_movement_model()

    def update(self, global_time: float = None):
        self.k = self.k + 1
        self.update_msg()
        self.update_model()