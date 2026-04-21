# RL_train/train_sim_core.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim.physics_engine import PhysicsEngine
from stable_baselines3 import SAC, PPO
import torch

class RLEnvAdapter(gym.Env):
    def __init__(self, engine: PhysicsEngine, agent_id : np.ndarray):
        super().__init__()
        self.engine = engine
        self.max_steps = 500

        self.horizon = 3  # 新增：动作块步数
        self.single_action_dim = 5 # 单步动作维度
        
        # === 动作空间修改 ===
        # 维度调整为 5 * 3 = 15
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.single_action_dim * self.horizon,), dtype=np.float32
        )

        self.history_len = 3  # 记忆过去 3 步
        self.observation_space = spaces.Dict({
            "rel_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "lidar": spaces.Box(0, 1, (36,), dtype=np.float32), 
            "semantic": spaces.Box(0, 1, (5,), dtype=np.float32),
            # 【新增】动作记忆：让网络知道自己上一秒在干嘛（比如正在后退）
            "prev_actions": spaces.Box(low=-1.0, high=1.0, shape=(self.history_len * 5,), dtype=np.float32),
            # 【新增】目标轨迹记忆：感知自身与目标的相对运动趋势
            "history_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_len * 3,), dtype=np.float32)
        })
        
    
        self.prev_potential = None
        self.prev_v = None
        self.prev_obs = None
        self.last_reward_terms = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.engine, name)

    def _angle_diff(self, a, b):
        diff = a - b
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def _check_termination(self, agent, obs):
        ex = agent.t_pos[0] - agent.position[0]
        ey = agent.t_pos[1] - agent.position[1]
        dist = np.hypot(ex, ey)
        terminated_success = bool(dist < 10)
        
        stuck = self._is_stuck(agent, window=5, pos_threshold=0.1)
        terminated_stuck = stuck and not terminated_success
        
        return terminated_success, terminated_stuck

    def _is_stuck(self, agent, window=5, pos_threshold=2):
        if not hasattr(agent, 'position_history'):
            agent.position_history = []
        agent.position_history.append(agent.position.copy())
        if len(agent.position_history) > window:
            agent.position_history.pop(0)
        if len(agent.position_history) < window:
            return False
        traj = np.array(agent.position_history)
        max_disp = np.max(np.linalg.norm(traj - traj[0], axis=1))
        return bool(max_disp < pos_threshold)

    def _action_post_process(self, raw_actions):
        """
        注意：此处接收的 raw_actions 必须是单步的 (5,) 向量
        """
        rou = 7.0 * (raw_actions[0] + 1) + 1.0
        # 【修改点 1】将 0.25 * np.pi 扩大为 0.6 * np.pi，提供更大的横向机动自由度
        phi = 0.25 * np.pi * raw_actions[1]
        e_theta = 0.25 * np.pi * raw_actions[2]
        v_r = (raw_actions[3] + 1)
        w_r = raw_actions[4]
        
        e_x = rou * np.cos(phi)
        e_y = rou * np.sin(phi)
        
        actions = np.array([e_x, e_y, e_theta, v_r, w_r], dtype=np.float32)
        return actions
    
    def get_agent_observation(self, agent):
        agent.p_pos = agent.attk_pos if agent.attk_pos is not None else agent.t_pos
        dx = agent.p_pos[0] - agent.position[0]
        dy = agent.p_pos[1] - agent.position[1]
        theta = agent.theta
        ex =  dx * np.cos(theta) + dy * np.sin(theta)
        ey = -dx * np.sin(theta) + dy * np.cos(theta)
        target_angle = np.arctan2(dy, dx)
        etheta = self._angle_diff(theta, target_angle)
        rel_goal = np.array([ex, ey, etheta], dtype=np.float32)
        
        obs_sector = self.engine.env_feedback['obs_sector_dict'].get(agent.id)
        if obs_sector is None: obs_sector = np.ones(36, dtype=np.float32) * 100.0
        lidar_data = 0.01 * np.array(obs_sector).astype(np.float32)

        # ★ 新增：读取当前分配的 Skill
        semantic_obs = getattr(agent, 'current_skill', np.array([0.9,0.9,0.9,0.9,1.0], dtype=np.float32))

        # ====== 【核心修改 3：维护目标位置的历史队列】 ======
        if not hasattr(agent, 'history_goal_buffer'):
            # 第一帧时，用当前的相对位置把整个队列填满
            agent.history_goal_buffer = [rel_goal.copy() for _ in range(self.history_len)]
        else:
            # 后续帧，把最新的位置推入，挤出最老的位置
            agent.history_goal_buffer.append(rel_goal.copy())
            agent.history_goal_buffer.pop(0)

        # ====== 【核心修改 4：组装新的观测字典】 ======
        return {
            "rel_goal": rel_goal,
            "lidar": lidar_data,
            "semantic": semantic_obs,
            # 将二维队列拉平为一维向量
            "prev_actions": np.concatenate(agent.history_action_buffer).astype(np.float32),
            "history_goal": np.concatenate(agent.history_goal_buffer).astype(np.float32)
        }

    def _compute_reward(self, agent, obs, action, terminated_success, terminated_stuck):
        # === 1. 定义物理边界参数 ===
        MAX_DIST = 200.0    
        MAX_V = 100.0        
        MAX_W = 5.0         
        dT = getattr(self, 'dT', 0.1) # 仿真步长
        
        ex, ey, etheta = obs["rel_goal"]
        dist = np.hypot(ex, ey)
        goal_potential = dist
        
        if self.prev_potential is None:
            delta_goal = 0.0
        else:
            delta_goal = self.prev_potential - goal_potential
        self.prev_potential = goal_potential

        # === 2. 核心修改：不对称进度奖励 (Asymmetric Progress) ===
        raw_progress = delta_goal / (MAX_V * dT)
        if raw_progress > 0:
            # 靠近目标时，给予全额奖励
            norm_progress = np.clip(raw_progress, 0.0, 1.0)
        else:
            # 远离目标（后退或绕行）时，将惩罚削弱至原来的 10%
            # 让网络敢于“无痛试错”地向后或侧向运动
            norm_progress = np.clip(raw_progress, -1.0, 0.0) * 0.1 

        # === 3. 核心修改：持续机动奖励 (Keep Moving Bonus) ===
        # 只要车速大于某个安全阈值（例如 2.0 m/s），就给恒定正奖励，逼迫它跑起来
        norm_moving = 1.0 if agent.v > 2.0 else 0.0

        # === 4. 其他归一化项 ===
        norm_dist = - np.clip(dist / MAX_DIST, 0.0, 1.0)
        norm_heading = np.cos(etheta)
        norm_w = - np.clip(abs(agent.w) / MAX_W, 0.0, 1.0)
        
        r_smooth_v = agent.v * (agent.v_max - agent.v)
        norm_v = np.clip(r_smooth_v / MAX_V ** 2, 0.0, 1.0)

        # 彻底信任底层 CBF 与 APF 的安全防撞托底，RL 层面不再因触发 CBF 而扣分
        norm_cbf = 0.0 

        # === 5. 动作一致性奖励 ===
        norm_consistency = 0.0
        if agent.prev_r_point is not None and agent.r_point is not None:
            prev_ex, prev_ey, prev_theta, prev_vr, prev_wr = agent.prev_r_point
            cur_ex, cur_ey, cur_theta, cur_vr, cur_wr = agent.r_point
            delta_e = np.array([cur_ex - prev_ex, cur_ey - prev_ey, self._angle_diff(cur_theta, prev_theta)])
            delta_e_pred = np.array([prev_vr * np.cos(prev_theta) * dT, prev_vr * np.sin(prev_theta) * dT, prev_wr * dT])
            
            eps = 1e-6
            norm_e = np.linalg.norm(delta_e)
            norm_e_pred = np.linalg.norm(delta_e_pred)
            if norm_e > eps and norm_e_pred > eps:
                cos_sim = np.dot(delta_e, delta_e_pred) / (norm_e * norm_e_pred)
            else:
                cos_sim = 0.0
                
            mag_ratio = norm_e / (norm_e_pred + eps)
            mag_penalty = abs(mag_ratio - 1.0)
            norm_consistency = np.clip(cos_sim - 0.5 * mag_penalty, -1.0, 1.0)

        # === 6. 权重分配矩阵 ===
        w_progress = 8.0      # 核心驱动：靠近目标
        w_moving = 2.0        # 核心驱动：保持运动
        w_dist = 0.2          # 削弱绝对距离的压迫感
        w_heading = 0.5       # 辅助航向
        w_v = 0.5
        w_w = 0.5             # 惩罚剧烈转向
        w_cbf = 0.0           # 权重为 0
        w_consistency = 1.0   # 动作平滑度
        
        step_reward = (w_progress * norm_progress + 
                       w_moving * norm_moving + 
                       w_dist * norm_dist + 
                       w_heading * norm_heading + 
                       w_v * norm_v +
                       w_w * norm_w + 
                       w_cbf * norm_cbf + 
                       w_consistency * norm_consistency - 
                       0.1) 
                       
        # === 7. 稀疏终止奖励 ===
        if terminated_success:
            step_reward += 100.0
        if terminated_stuck:
            step_reward -= 30.0 + 20.0 * (np.clip(goal_potential / MAX_DIST, 0.0, 1.0))
        if self.engine.steps >= self.max_steps:
            step_reward -= 20.0

        return float(step_reward)

    def reset(self, *, seed=None, options=None):
        self.engine.reset_engine()
        self.prev_potential = None
        self.prev_action = None
        self.last_reward_terms = None
        # ====== 【核心修改 2：回合重置时初始化队列】 ======
        for agent in self.engine.agents:
            # 动作缓存：初始化为全 0 动作
            agent.history_action_buffer = [np.zeros(self.single_action_dim, dtype=np.float32) for _ in range(self.history_len)]
            # 位置缓存：如果存在则删除，让 get_agent_observation 重新用初始位置填满
            if hasattr(agent, 'history_goal_buffer'):
                delattr(agent, 'history_goal_buffer')
            # ★ 新增：每次重置时，给智能体随机分配一个 5 维的伪 Skill，防止条件崩溃
            agent.current_skill = np.random.uniform(0, 1, size=(5,)).astype(np.float32)
            obs = self.get_agent_observation(agent)
        self.prev_obs = obs
        return obs, {}

    def step(self, action_input):
        """
        核心重构：支持 Action Chunking (H=3)
        """
        # ====== 模式 A: 并行仿真模式 (输入是字典) ======
        if isinstance(action_input, dict):
            controllers = {}
            # 1. 解析动作块并提取第一步
            for agent_id, raw_chunk in action_input.items():
                if raw_chunk is None: continue 
                # 将 (15,) reshape 为 (3, 5) 并取第 0 步
                chunk = raw_chunk[:5]
                action_step_0 = chunk 
                controllers[agent_id] = self._action_post_process(action_step_0)
                # ====== 【核心修改 5a：更新动作缓存队列 (并行模式)】 ======
                agent = next((a for a in self.engine.agents if a.id == agent_id), None)
                if agent and hasattr(agent, 'history_action_buffer'):
                    agent.history_action_buffer.append(chunk.copy()) # 保存未经缩放的 raw 动作
                    agent.history_action_buffer.pop(0)

            # 2. 物理步进
            self.engine.step_physics(controllers)

            # 3. 返回反馈
            obs_dict = {}
            reward_dict = {}
            done_dict = {}
            info_dict = {}
            truncated = bool(self.engine.steps >= self.max_steps)
            
            for agent in self.engine.agents:
                if agent.disabled: continue
                obs = self.get_agent_observation(agent)
                obs_dict[agent.id] = obs
                terminated_success, terminated_stuck = self._check_termination(agent, obs)
                # 计算奖励时，传入的是执行的那一步物理动作
                reward = self._compute_reward(agent, obs, controllers[agent.id], terminated_success, terminated_stuck)
                done_dict[agent.id] = terminated_success or terminated_stuck
                reward_dict[agent.id] = reward
            return obs_dict, reward_dict, done_dict, truncated, info_dict

        else:
            # ====== 模式 B: 单体训练模式 ======
            raw_chunk = action_input
            if raw_chunk is None:
                controllers = None 
                self.prev_action = np.zeros(self.single_action_dim * self.horizon)
                agent = self.engine.agents[0]
            else:
                # 重塑并提取第一步
                chunk = raw_chunk[:5]
                action_step_0 = chunk
                phys_action = self._action_post_process(action_step_0)
                agent = self.engine.agents[0]
                self.prev_action = np.zeros(self.single_action_dim * self.horizon) 
                controllers = {agent.id: phys_action}
                # ====== 【核心修改 5b：更新动作缓存队列 (单体模式)】 ======
                if hasattr(agent, 'history_action_buffer'):
                    agent.history_action_buffer.append(chunk.copy()) # 保存未经缩放的 raw 动作
                    agent.history_action_buffer.pop(0)

            self.engine.step_physics(controllers)
            
            obs = self.get_agent_observation(agent)
            terminated_success, terminated_stuck = self._check_termination(agent, obs)
            
            # 奖励计算使用实际执行的 phys_action
            exec_action = controllers[agent.id] if controllers else np.zeros(5)
            reward = self._compute_reward(agent, obs, exec_action, terminated_success, terminated_stuck)
            
            info = {"reward_terms": getattr(self, 'last_reward_terms', {})}
            truncated = bool(self.engine.steps >= self.max_steps)
            terminated = terminated_success or terminated_stuck
            self.prev_obs = obs
            return obs, reward, terminated, truncated, info