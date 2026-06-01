import numpy as np
import random
import torch
import os
from agent.control.bot_controller import guidance_with_obstacle_avoidance
from agent.MapProcess import MapProcesser

# 引入 MPC 相关组件
from models.vqvae.VQVAE_skill_generate import SoftVQVAE
from models.predictors.agent_dyn_predictor import ForwardPredictor
from models.vae.action_vae import ActionVAE
from agent.planning.latent_mpc_search import LatentMPCPlanner
from agent.MapProcess import MapProcesser, GlobalNavField

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
    # ====================================================
    # 静态共享变量：确保所有智能体共享一个 MPC 规划器 / VQ-VAE，避免显存溢出
    # ====================================================
    _shared_mpc_planner = None
    _shared_vq_model = None      # VQ-VAE decoder，用于 MPC 动作解码
    _shared_skill_vecs = None    # skill 向量表 (16, 5)
    _mpc_seq_len = 10            # 从 checkpoint 推断
    _models_loaded = False
    def __init__(self, agent):
        self.agent = agent
        # MapProcesser 可能没有任何初始化，但为了保险调用一下 super
        # super().__init__() 
        # 从配置中读取是否启用 MPC (默认为 False)
        self.use_latent_mpc = getattr(self.agent, 'use_latent_mpc', False)
        # === 新增：为当前智能体初始化全局拓扑导航对象 ===
        self.global_nav = GlobalNavField()
        # 懒加载：只有在启用了 MPC 且尚未加载模型时才初始化
        if self.use_latent_mpc and not BehaviorSystem._models_loaded:
            BehaviorSystem._init_shared_mpc()

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

    @classmethod
    def _init_shared_mpc(cls):
        """初始化共享的 Latent MPC 模型与网络 (底层 SAC 观测对齐)"""
        print(">>> [BehaviorSystem] 正在初始化全局共享的 Latent MPC 组件...")
        device = torch.device('cpu')
        num_skills = 16
        POS_LIMIT = 500.0
        ANG_LIMIT = np.pi
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 1. 加载 VQ-VAE，从 checkpoint 推断 seq_len
        vqvae_path = os.path.join(base_dir, "models/vqvae/vqvae_skills.pth")
        if not os.path.exists(vqvae_path):
            print(f"  ❌ VQ-VAE 未找到: {vqvae_path}，无法启用 MPC")
            cls._models_loaded = True  # 标为已加载防重复尝试
            return

        # 从 checkpoint 推断 seq_len (enc.0.weight: Linear(seq_len*5, 256))
        ckpt = torch.load(vqvae_path, map_location=device)
        enc_in_features = ckpt['enc.0.weight'].shape[1]
        seq_len = enc_in_features // 5  # action_dim=5

        cls._mpc_seq_len = seq_len
        vq_model = SoftVQVAE(seq_len=seq_len, action_dim=5, latent_dim=4, num_skills=num_skills).to(device)
        vq_model.load_state_dict(ckpt)
        vq_model.eval()
        cls._shared_vq_model = vq_model
        print(f"  ✓ VQ-VAE 已加载 (seq_len={seq_len}): {vqvae_path}")

        # 构建 skill 向量 (codebook 潜变量 [0,1] + 归一化 skill ID)
        with torch.no_grad():
            codebook = vq_model.vq.embedding.weight.data
            codebook = (codebook + 1.0) / 2.0  # [-1,1] → [0,1]
            ids = torch.arange(num_skills).float().to(device).unsqueeze(1) / num_skills
            skill_vecs = torch.cat([codebook, ids], dim=1)  # (16, 5)
            cls._shared_skill_vecs = skill_vecs

        # 2. 加载 Forward Model (horizon == seq_len)
        fwd_path = os.path.join(base_dir, "models/predictors/forward_model.pth")
        if not os.path.exists(fwd_path):
            print(f"  ❌ Forward Model 未找到: {fwd_path}，无法启用 MPC")
            cls._models_loaded = True
            return

        # 从 checkpoint 推断 horizon (start_token shape → horizon)
        fwd_ckpt = torch.load(fwd_path, map_location=device)
        horizon = fwd_ckpt['start_token'].shape[1]
        # causal_mask provides backup
        if 'causal_mask' in fwd_ckpt:
            horizon = fwd_ckpt['causal_mask'].shape[0]

        forward_model = ForwardPredictor(horizon=horizon).to(device)
        forward_model.load_state_dict(fwd_ckpt)
        forward_model.eval()
        print(f"  ✓ Forward Model 已加载 (horizon={horizon}): {fwd_path}")

        # 2.5 加载 ActionVAE → 预计算每个 skill 的首帧动作嵌入 (供分析性 history_goal 更新)
        vae_path = os.path.join(base_dir, "models/vae/action_vae_pretrained.pt")
        if os.path.exists(vae_path):
            vae_ckpt = torch.load(vae_path, map_location=device)
            action_vae = ActionVAE().to(device)
            # 兼容两种 checkpoint 格式: 直接 state_dict 或包含 'model_state_dict' 的 dict
            if 'model_state_dict' in vae_ckpt:
                action_vae.load_state_dict(vae_ckpt['model_state_dict'])
            else:
                action_vae.load_state_dict(vae_ckpt)
            action_vae.eval()
            forward_model.register_skill_action_embeddings(vq_model, action_vae)
            print(f"  ✓ ActionVAE 已加载并注册 skill→action 嵌入: {vae_path}")
        else:
            print(f"  ⚠ ActionVAE 未找到: {vae_path}，分析性 history_goal 将使用零嵌入")

        # 3. 实例化 MPC Planner
        cls._shared_mpc_planner = LatentMPCPlanner(
            forward_model=forward_model,
            skill_vecs=skill_vecs,
            device=device,
            num_skills=num_skills,
            pos_limit=POS_LIMIT,
            ang_limit=ANG_LIMIT
        )
        cls._models_loaded = True
        print(f">>> [BehaviorSystem] Latent MPC 已激活 (seq_len={seq_len}, horizon={horizon})")

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
        v_des, w_des, self.agent.dv, self.agent.dw = guidance_with_obstacle_avoidance(self.agent, guide_state)
        
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

    def task_allocate_model(self, obs_d=None):
        """
        高层任务分配与规划模型 (MPC 选最优 skill)

        obs_d: 底层 SAC 4-key Dict {lidar_2d, goal_dir, history_goal, dynamics}
        返回: 最优 skill 向量 (5-dim)
        """
        if not getattr(self.agent, 'use_latent_mpc', False) or self._shared_mpc_planner is None or obs_d is None:
            # 无 MPC 时存储零 skill
            self.agent.mpc_skill = np.zeros(5, dtype=np.float32)
            return self.agent.mpc_skill

        # 只有当地图数据加载完毕后，才启动全局 MPC 寻优
        if self.agent.down_sampled_map is not None and self.agent.grid_map is not None:
            ds_h, ds_w = self.agent.down_sampled_map.shape
            orig_h, orig_w = self.agent.grid_map.shape

            nav_grid_size = self.agent.grid_size * (orig_w / ds_w)
            panel_center = np.array([orig_w * self.agent.grid_size * 0.5,
                                     orig_h * self.agent.grid_size * 0.5])

            self.global_nav.update_map(self.agent.down_sampled_map, nav_grid_size)
            target_pos = self.agent.t_pos if self.agent.t_pos is not None else self.agent.TARGET_POS
            self.global_nav.update_target(target_pos, panel_center)

            self._shared_mpc_planner.update_global_map(
                self.global_nav.nav_field, panel_center, nav_grid_size
            )

            agent_global_info = {
                'target_x': target_pos[0],
                'target_y': target_pos[1],
                'vehicle_heading': self.agent.theta
            }

            best_skill = self._shared_mpc_planner.search_best_skill(obs_d, agent_global_info)
        else:
            best_skill = self._shared_mpc_planner.search_best_skill(obs_d, None)

        # 存储 MPC 选中的 skill 到 agent
        self.agent.mpc_skill = best_skill
        self.agent.mpc_skill_idx = int(best_skill[4] * 16)  # norm_id → index
        return best_skill

    @classmethod
    def decode_skill(cls, skill_vec):
        """VQ-VAE decoder: skill 向量 → T 帧动作序列 (T, 5)"""
        if cls._shared_vq_model is None:
            return None
        with torch.no_grad():
            z = torch.FloatTensor(skill_vec[:4] * 2.0 - 1.0).unsqueeze(0)  # [0,1]→[-1,1]
            recon = cls._shared_vq_model.dec(z).view(cls._mpc_seq_len, 5)   # (T, 5)
        return recon.cpu().numpy()

    def get_mpc_action(self, obs_d):
        """
        MPC 规划 + 解码 → 15-dim action chunk (与 SAC 输出格式一致)。
        返回: (skill_idx, action_chunk_15)
        """
        best_skill = self.task_allocate_model(obs_d)
        skill_idx = self.agent.mpc_skill_idx

        # VQ-VAE decoder 解码 T 帧动作 → 取前 3 帧拼成 15-dim chunk
        decoded = self.decode_skill(best_skill)  # (T, 5)
        if decoded is not None and len(decoded) >= 3:
            action_chunk = decoded[:3].flatten()  # (15,)
        else:
            action_chunk = np.zeros(15, dtype=np.float32)

        # 简化输出：每 N 步打印一次
        if self.agent.k % 100 == 0:
            print(f"[MPC] Agent {self.agent.id:2d} → Skill {skill_idx:2d} | "
                  f"action[:5]=[{action_chunk[0]:+.2f} {action_chunk[1]:+.2f} "
                  f"{action_chunk[2]:+.2f} {action_chunk[3]:+.2f} {action_chunk[4]:+.2f}]")

        return skill_idx, action_chunk.astype(np.float32)

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