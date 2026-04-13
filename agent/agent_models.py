import numpy as np
import random
from agent.bot_controller import guidance_with_obstacle_avoidance
from agent.MapProcess import MapProcesser

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

'''
    [重构版] 行为动力学组件
    负责 Model 逻辑 (Move, Sense, Attack)
'''
class BehaviorSystem(MapProcesser):
    def __init__(self, agent):
        self.agent = agent
        # MapProcesser 可能没有任何初始化，但为了保险调用一下 super
        # super().__init__() 

    # ====================================================
    #  [核心技巧] 属性桥接 (Property Bridging)
    #  欺骗父类 MapProcesser，让它以为自己还是原来的 Agent
    # ====================================================
    @property
    def position(self): return self.agent.position
    @property
    def grid_map(self): return self.agent.grid_map
    @property
    def grid_size(self): return self.agent.grid_size
    @property
    def smoke_zones(self): return self.agent.smoke_zones
    @property
    def smoke_attenuation(self): return self.agent.smoke_attenuation
    @property
    def sense_field(self): return self.agent.sense_field
    @property
    def local_obstacles(self): return self.agent.local_obstacles
    @local_obstacles.setter
    def local_obstacles(self, value): self.agent.local_obstacles = value
    @property
    def obs_sector(self): return self.agent.obs_sector

    # ====================================================
    #  原 agent_models 逻辑 (self.xxx 替换为 self.agent.xxx)
    # ====================================================
    def move_logic_model(self):
        self.guidance_control()

    def guidance_control(self):
        if self.agent.disabled:
            self.agent.v_left = 0.0
            self.agent.v_right = 0.0
            return
        
        guide_state = self.agent.r_point if (self.agent.r_point is not None and np.all(np.isfinite(self.agent.r_point))) \
            else np.hstack((self.agent.t_pos - self.agent.position, np.array([self.agent.theta, 1e-2, 1e-2])))
        
        # 注意：这里 guidance_with_obstacle_avoidance 需要 agent 实例
        v_des, w_des = guidance_with_obstacle_avoidance(self.agent, guide_state)
        
        self.agent.v = v_des
        self.agent.w = w_des
        # self.agent.v_left  = (v - w * self.agent.L) / self.agent.R
        # self.agent.v_right = (v + w * self.agent.L) / self.agent.R
        # eps = self.agent.v_min
        
        # if self.agent.v_left < eps:
        #     self.agent.v_left = self.agent.v_left / (eps - self.agent.v_left)
        # if self.agent.v_right < eps:
        #     self.agent.v_right = self.agent.v_right / (eps - self.agent.v_right)
        
        # self.agent.v_left  = max(self.agent.v_left, 0.0)
        # self.agent.v_right = max(self.agent.v_right, 0.0)

    def Kinematic_model(self, v_left: float, v_right: float):
        self.agent.v = (v_left + v_right) * self.agent.R / 2
        self.agent.w = (v_right - v_left) * self.agent.R / (2 * self.agent.L)

    def task_allocate_model(self):
        return 0

    def smoke_model(self):
        can_smoke = (
            self.agent.smoke_mission and
            self.agent.smoke_remain > 0 
        )
        if can_smoke:
            self.agent.smoke_remain -= 1
        if self.agent.smoke_remain <= 0:
             self.agent.smoke_mission = False

    def _angle_diff(self, a, b):
        diff = a - b
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff
    
    def cannon_turning_control(self, error):
        us1 = np.abs(error) ** (0.2) 
        us2 = np.abs(error) ** (0.8)
        u = us1 + us2
        if np.abs(error) >= 0.1:
            self.agent.cannon_w  = 10 * u * np.sign(error)
        else:
            self.agent.cannon_w  = np.abs(error)/(0.1 - np.abs(error))* np.sign(error)
        max_w = self.agent.cannon_w_max
        self.agent.cannon_w = np.clip(self.agent.cannon_w, -max_w, max_w)
    
    def attack_model(self):
        if self.agent.attk_pos is not None:
            target_dir = -(self.agent.position - self.agent.attk_pos)
            target_angle = np.arctan2(target_dir[1], target_dir[0])
            error_eta = self._angle_diff(target_angle, self.agent.cannon_theta)
            target_dist = np.linalg.norm(target_dir)
            can_fire = (
                self.agent.cannon_remain > 0 and
                target_dist <= self.agent.attack_range and
                self.agent.launch_delay_count >= self.agent.launch_delay
            )
            target_outof_sight = (
                self.agent.cannon_remain > 0 and
                target_dist > self.agent.attack_range and
                self.agent.launch_delay_count >= self.agent.launch_delay  
            )
            if can_fire:
                if abs(error_eta) <= 0.01:
                    self.agent.cannon_remain -= 1
                    self.agent.cannon_launched = True
                    self.agent.launch_delay_count = 0
                    sigma = self.agent.attk_variance
                    noise = np.random.normal(loc=0.0, scale=sigma, size=2)
                    attk_pos = self.agent.attk_pos + noise
                    self.agent.attk_pos = attk_pos
                else:
                    self.cannon_turning_control(error_eta)
            elif target_outof_sight:
                self.agent.cannon_launched = False
                error_eta_normal =  (self.agent.theta - self.agent.cannon_theta)
                self.cannon_turning_control(error_eta_normal)
            else:
                self.agent.cannon_w = 0.0
                self.agent.cannon_launched = False
        else:
            error_eta_normal =  (self.agent.theta - self.agent.cannon_theta)
            self.cannon_turning_control(error_eta_normal)
            return None

    def sense_model(self):
        # 这里的 update_obstacles 是继承自 MapProcesser 的
        # 因为我们做了属性桥接，所以它能正常运行
        # self.update_obstacles(n_closest = 10, n_comp = 2, connectivity = 8)
        self.update_target()

    
    def update_target(self):
        if not self.agent.targets_id == 0:
            p_pos = self.agent.targets_info["position"][f"{self.agent.targets_id}"]
            max_sense_range = self.agent.sense_field
            max_sense_angle = self.agent.sense_angle
            # 新增判断：如果坐标还是初始的 [0, 0]，说明还没收到通信更新，跳过探测
            if not (p_pos[0] == 0.0 and p_pos[1] == 0.0):
                Tpos = self.target_detect_model(p_pos, max_sense_range, max_sense_angle)
                self.agent.p_pos = Tpos if Tpos is not None else self.agent.p_pos
        else:
            self.agent.p_pos = self.agent.TARGET_POS + 20 * self.agent.target_distance
            self.agent.t_pos = self.agent.TARGET_POS

        if not self.agent.cannon_targets_id == 0:
            attk_pos = self.agent.cannon_targets_info["position"][f"{self.agent.cannon_targets_id}"]
            max_sense_range = self.agent.sense_field
            max_sense_angle = self.agent.sense_angle
            Tpos = self.target_detect_model(attk_pos, max_sense_range, max_sense_angle)
            self.agent.attk_pos = Tpos if Tpos is not None else self.agent.attk_pos
        else:
            self.agent.attk_pos = None

    def target_detect_model(self, t_pos, max_sense_range, max_sense_angle):
        target_dir = -(self.agent.position - t_pos)
        t_angle = np.arctan2(target_dir[1], target_dir[0]) - self.agent.theta
        t_dist = np.linalg.norm(t_pos - self.agent.position)
        if np.abs(t_angle) < max_sense_angle:
            # block_and_smoke_check 也是继承自 MapProcesser
            delta = self.block_and_smoke_check(t_pos)
            if delta == 0:
                return None
            if t_dist < max_sense_range:
                sense_P = delta * self.agent.sense_basic_P0
            else:
                sense_P = delta * self.agent.sense_basic_P0 * np.exp(-self.agent.sense_P_attenuation * (t_dist - max_sense_range)**2)
            if random.random() < sense_P:
                sigma = self.agent.sense_variance
                noise = np.random.normal(loc=0.0, scale=sigma, size=2)
                observed_pos = t_pos + noise
                return observed_pos
            else:
                return None
        else:
            return None