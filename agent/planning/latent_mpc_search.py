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

    def _compute_cost(self, future_states):
        """
        【模块 3】：MPC 代价函数设计 (Cost Function)
        基于推演的未来状态，计算每个技能的代价得分
        """
        ex = future_states[:, :, 36]
        ey = future_states[:, :, 37]
        
        # 1. 进度代价：终端距离目标越近越好
        terminal_dist = torch.sqrt(ex[:, -1]**2 + ey[:, -1]**2) 
        cost_progress = terminal_dist * 1.0

        # 2. 安全代价：雷达视野低于 0.15 施加硬惩罚 (碰撞风险)
        lidar_preds = torch.clamp(future_states[:, :, :36], 0.0, 1.0)
        danger_mask = (lidar_preds < 0.15).float()
        cost_collision = danger_mask.sum(dim=(1, 2)) * 50.0 

        total_cost = cost_progress + cost_collision
        return total_cost

    def search_best_skill(self, obs_d):
        """
        【主入口】：技能寻优
        串联观测对齐、并行推演与代价评估，返回最佳技能向量
        """
        # 1. 对齐观测空间
        norm_obs_t = self._align_observation(obs_d)
        
        # 2. 获取未来状态预测
        future_states = self._parallel_rollout(norm_obs_t)
        
        # 3. 计算轨迹代价
        total_cost = self._compute_cost(future_states)
        
        # 4. 选出代价最小的最优技能
        best_skill_idx = torch.argmin(total_cost).item()
        best_skill_vec = self.skill_vecs[best_skill_idx].cpu().numpy()
        
        return best_skill_vec