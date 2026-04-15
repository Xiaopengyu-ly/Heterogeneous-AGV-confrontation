import torch
import numpy as np

def latent_mpc_search(self, obs_d):
    """
    Latent MPC 寻优算法：在潜空间中推演未来，挑选最优技能
    """
    # 1. 对齐观测空间至 68 维
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
    norm_obs_t = torch.FloatTensor(norm_obs).to(self.device).unsqueeze(0) # [1, 68]

    # 2. 复制 16 份用于并行前向推演
    lidar_input = norm_obs_t[:, :36].repeat(self.num_skills, 1)
    aux_input = norm_obs_t[:, 36:].repeat(self.num_skills, 1)

    with torch.no_grad():
        deltas = self.forward_model(lidar_input, aux_input, self.skill_vecs)
    
    current_state_t = norm_obs_t.repeat(self.num_skills, 1).unsqueeze(1) 
    future_states = current_state_t + deltas

    # 3. 核心代价函数设计 (Cost Function)
    ex = future_states[:, :, 36]
    ey = future_states[:, :, 37]
    # 进度代价：终端距离越小越好
    terminal_dist = torch.sqrt(ex[:, -1]**2 + ey[:, -1]**2) 
    cost_progress = terminal_dist * 1.0

    # 安全代价：雷达视野低于 0.15 施加硬惩罚
    lidar_preds = torch.clamp(future_states[:, :, :36], 0.0, 1.0)
    danger_mask = (lidar_preds < 0.15).float()
    cost_collision = danger_mask.sum(dim=(1, 2)) * 50.0 

    total_cost = cost_progress + cost_collision

    # 4. 选出代价最小的最优技能
    best_skill_idx = torch.argmin(total_cost).item()
    best_skill_vec = self.skill_vecs[best_skill_idx].cpu().numpy()
    
    return best_skill_vec