# agent/planning/latent_mpc_search.py
import torch
import numpy as np

class LatentMPCPlanner:
    def __init__(self, forward_model, skill_vecs, device, num_skills=16, pos_limit=500.0, ang_limit=np.pi):
        """
        基于潜空间的模型预测控制 (Latent MPC) 寻优器
        """
        self.forward_model = forward_model
        self.skill_vecs = skill_vecs
        self.device = device
        self.num_skills = num_skills
        self.POS_LIMIT = pos_limit
        self.ANG_LIMIT = ang_limit

    def _align_observation(self, obs_d):
        """
        【模块 1】：状态对齐与归一化
        将原始观测数据归一化并拼接为 68 维张量
        """
        norm_lidar = np.array(obs_d['lidar'], dtype=np.float32)
        norm_semantic = np.array(obs_d['semantic'], dtype=np.float32)
        norm_prev_act = np.array(obs_d['prev_actions'], dtype=np.float32)
        
        norm_phy = np.zeros(3, dtype=np.float32)
        norm_phy[0] = obs_d['rel_goal'][0] / self.POS_LIMIT
        norm_phy[1] = obs_d['rel_goal'][1] / self.POS_LIMIT
        norm_phy[2] = obs_d['rel_goal'][2] / self.ANG_LIMIT
        
        norm_hg = np.zeros(9, dtype=np.float32)
        for i in range(3):
            norm_hg[i*3] = obs_d['history_goal'][i*3] / self.POS_LIMIT
            norm_hg[i*3+1] = obs_d['history_goal'][i*3+1] / self.POS_LIMIT
            norm_hg[i*3+2] = obs_d['history_goal'][i*3+2] / self.ANG_LIMIT
            
        norm_obs = np.concatenate([norm_lidar, norm_phy, norm_semantic, norm_prev_act, norm_hg])
        return torch.FloatTensor(norm_obs).to(self.device).unsqueeze(0) # [1, 68]

    def _parallel_rollout(self, norm_obs_t):
        """
        【模块 2】：并行推演
        利用动力学前向模型，并行评估所有可用技能在未来 horizon 步的影响
        """
        lidar_input = norm_obs_t[:, :36].repeat(self.num_skills, 1)
        aux_input = norm_obs_t[:, 36:].repeat(self.num_skills, 1)
        
        with torch.no_grad():
            deltas = self.forward_model(lidar_input, aux_input, self.skill_vecs)
        
        current_state_t = norm_obs_t.repeat(self.num_skills, 1).unsqueeze(1) 
        future_states = current_state_t + deltas
        return future_states

    def _compute_cost_and_constraints(self, future_states, current_skill_vec):
        """
        【重构版】：严格分离 MPC 的目标函数 (Costs) 与 状态约束 (Constraints)
        """
        horizon = future_states.size(1)
        ex = future_states[:, :, 36]
        ey = future_states[:, :, 37]
        etheta = future_states[:, :, 38]

        # ==========================================
        # 模块 A：计算软代价 (Soft Costs / Objective Function)
        # ==========================================
        # 1. 积分进度代价 (越快靠近目标越好)
        dist_to_goal = torch.sqrt(ex**2 + ey**2)
        time_weights = torch.linspace(0.5, 1.5, steps=horizon).to(self.device) 
        cost_progress = (dist_to_goal * time_weights).mean(dim=1) * 5.0
        # 2. 航向对齐代价 (鼓励车头对准目标)
        cost_heading = (torch.abs(etheta) * time_weights).mean(dim=1) * 1.5
        # 3. 技能平滑代价 (惩罚高频切换，保持宏观动作一致性)
        curr_s_expanded = current_skill_vec.unsqueeze(0).repeat(self.num_skills, 1)
        cost_switch = torch.norm(self.skill_vecs - curr_s_expanded, dim=1) * 2.0
        # 组合纯代价（此时不包含碰撞惩罚）
        base_cost = cost_progress + cost_heading + cost_switch

        # ==========================================
        # 模块 B：计算硬约束 (Hard Constraints)
        # ==========================================
        lidar_preds = torch.clamp(future_states[:, :, :36], 0.0, 1.0)
        safe_margin = 0.3  # 物理边界约束：雷达测距不得小于 0.15
        # 计算每条轨迹违背约束的程度 (侵入安全距离的累计量)
        violation_mask = safe_margin - lidar_preds
        violation_amount = torch.clamp(violation_mask, min=0.0).sum(dim=(1, 2))
        # 布尔型可行集标志：违测量为 0 的才是合法的可行解 (Feasible)
        is_feasible = violation_amount == 0
        return base_cost, is_feasible, violation_amount

    def search_best_skill(self, obs_d):
        """
        【主入口】：带可行集检查与退化保护的寻优
        """
        norm_obs_t = self._align_observation(obs_d)
        current_skill_vec = torch.FloatTensor(obs_d['semantic']).to(self.device)
        
        future_states = self._parallel_rollout(norm_obs_t)
        
        # 拿到代价和约束判定
        base_cost, is_feasible, violation_amount = self._compute_cost_and_constraints(future_states, current_skill_vec)
        
        # ==========================================
        # 模块 C：带约束的过滤与决策逻辑
        # ==========================================
        if is_feasible.any():
            # 正常情况：可行集非空 (至少存在一条不撞车的轨迹)
            # 将违背约束的轨迹代价设为无穷大，直接从候选池剔除
            valid_costs = torch.where(is_feasible, base_cost, torch.tensor(float('inf')).to(self.device))
            best_skill_idx = torch.argmin(valid_costs).item()
        else:
            # 极端情况：可行集为空 (所有技能在未来 Horizon 内都会触发碰撞约束)
            # 退化策略 (Constraint Relaxation)：放弃进度寻优，全力最小化约束违背量 (找一条撞得最轻、或者最晚撞的轨迹)
            best_skill_idx = torch.argmin(violation_amount).item()
            
        best_skill_vec = self.skill_vecs[best_skill_idx].cpu().numpy()
        
        return best_skill_vec