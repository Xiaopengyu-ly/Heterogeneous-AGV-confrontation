import torch
import numpy as np

from sim.replay_buffer import lidar2d_to_distances


class LatentMPCPlanner:
    _global_step = 0
    _print_interval = 1
    _stats_window = 100
    _stats = {
        'pred_lidar_mse': [], 'pred_phy_mse': [], 'pred_diverged': 0,
        'nav_cost': [], 'feasible_count': [], 'skill_switches': 0,
        'skill_usage': [0] * 16, 'col_cost': [], 'boundary_cost': [], 'cur_dist': [],
    }

    def __init__(self, forward_model, skill_vecs, device, num_skills=16,
                 pos_limit=500.0, ang_limit=np.pi, max_lidar_range=100.0):
        self.forward_model = forward_model
        self.skill_vecs = skill_vecs           # (16, 5)
        self.device = device
        self.num_skills = num_skills
        self.POS_LIMIT = pos_limit
        self.ANG_LIMIT = ang_limit
        self.MAX_LIDAR_RANGE = max_lidar_range

        # 全局地图
        self.nav_field_tensor = None
        self.world_offset = None
        self.nav_grid_size = 1.0
        self.map_height = 0
        self.map_width = 0

        # 遥测缓存
        self.last_pred_state = None
        self.last_skill_idx = None
        self._cached_skill = None

    def update_global_map(self, nav_field_array, world_offset, nav_grid_size):
        self.nav_field_tensor = torch.tensor(nav_field_array, dtype=torch.float32, device=self.device)
        self.world_offset = torch.tensor(world_offset, dtype=torch.float32, device=self.device)
        self.nav_grid_size = nav_grid_size
        self.map_height, self.map_width = self.nav_field_tensor.shape

    def _align_observation(self, obs_d):
        """
        从底层 SAC 4-key Dict 提取 Forward Model 输入。
        返回:
          obs_flat:       (1, 256) 展平的观测向量
          norm_lidar:     (1, 36)  当前 lidar 距离 (归一化)，CBF 基值
          cur_goal_dir:   (2,)     当前 goal_dir (未归一化)
        """
        lidar_2d = np.array(obs_d['lidar_2d'], dtype=np.float32)
        goal_dir = np.array(obs_d['goal_dir'], dtype=np.float32)
        history_goal = np.array(obs_d['history_goal'], dtype=np.float32)
        dynamics = np.array(obs_d['dynamics'], dtype=np.float32)

        obs_flat = np.concatenate([
            lidar_2d.flatten(),    # 180
            goal_dir,              # 2
            history_goal,          # 72
            dynamics               # 2
        ])  # 256

        raw_dist = lidar2d_to_distances(lidar_2d, max_range=self.MAX_LIDAR_RANGE)
        norm_lidar_dist = raw_dist / self.MAX_LIDAR_RANGE  # (36,)

        obs_flat_t = torch.FloatTensor(obs_flat).to(self.device).unsqueeze(0)       # (1, 256)
        norm_lidar_t = torch.FloatTensor(norm_lidar_dist).to(self.device).unsqueeze(0)  # (1, 36)

        return obs_flat_t, norm_lidar_t, goal_dir

    def _parallel_rollout(self, obs_flat_t, skill_vecs):
        """
        并行推演 16 skills 的未来轨迹。
        Forward Model: obs_flat(B,256) + skill(B,5) → autoregressive → deltas(B,T,38)
        Cumsum 得到绝对状态的 lidar 距离和 goal_dir。
        """
        B = skill_vecs.shape[0]
        obs_batch = obs_flat_t.repeat(B, 1)  # (B, 256)

        with torch.no_grad():
            deltas = self.forward_model(obs_batch, skill_vecs)  # (B, T, 38)

        lidar_deltas = deltas[:, :, :36]   # (B, T, 36)
        goal_deltas = deltas[:, :, 36:38]  # (B, T, 2)

        # 基值: obs_flat 中 goal_dir 在前 2 维 (goal_dir at 180:182)
        cur_goal = obs_flat_t[:, 180:182]  # (1, 2)
        cur_lidar = self._last_norm_lidar  # set in search_best_skill

        lidar_abs = cur_lidar.unsqueeze(1) + torch.cumsum(lidar_deltas, dim=1)  # (B, T, 36)
        goal_abs = cur_goal.unsqueeze(1) + torch.cumsum(goal_deltas, dim=1)     # (B, T, 2)

        return lidar_abs, goal_abs

    def _compute_cost_and_constraints(self, lidar_abs, goal_abs, agent_global_info, cur_goal_dir):
        horizon = lidar_abs.size(1)

        rou = goal_abs[:, :, 0] * self.POS_LIMIT
        etheta = goal_abs[:, :, 1] * self.ANG_LIMIT
        cur_rou = cur_goal_dir[0] * self.POS_LIMIT
        cur_etheta = cur_goal_dir[1] * self.ANG_LIMIT

        # ---- 平滑代价 ----
        time_weights = torch.linspace(0.5, 1.5, steps=horizon, device=self.device)
        cost_heading = (torch.abs(etheta) * time_weights).mean(dim=1) * 0.5
        base_cost = cost_heading.clone()
        boundary_penalty = torch.zeros(self.num_skills, device=self.device)

        # ---- 拓扑代价 ----
        if agent_global_info is not None and self.nav_field_tensor is not None:
            target_x = agent_global_info['target_x']
            target_y = agent_global_info['target_y']
            alpha = agent_global_info['vehicle_heading'] - cur_etheta

            world_dx = rou * np.cos(alpha)
            world_dy = rou * np.sin(alpha)
            future_x = target_x - world_dx
            future_y = target_y - world_dy

            raw_gx = (future_x + self.world_offset[0]) / self.nav_grid_size
            raw_gy = (future_y + self.world_offset[1]) / self.nav_grid_size

            over_right = torch.clamp(raw_gx - (self.map_width - 1), min=0.0)
            over_left = torch.clamp(-raw_gx, min=0.0)
            over_bottom = torch.clamp(raw_gy - (self.map_height - 1), min=0.0)
            over_top = torch.clamp(-raw_gy, min=0.0)
            boundary_penalty = (over_right + over_left + over_bottom + over_top).sum(dim=1) * 500.0

            grid_x = torch.clamp(raw_gx.long(), 0, self.map_width - 1)
            grid_y = torch.clamp(raw_gy.long(), 0, self.map_height - 1)

            map_cost = self.nav_field_tensor[grid_y, grid_x]
            map_cost = torch.where(torch.isinf(map_cost),
                                   torch.tensor(100.0, device=self.device), map_cost)
            cost_progress = (map_cost * time_weights).mean(dim=1) * 5.0
            base_cost += cost_progress + boundary_penalty
        else:
            cost_progress = (rou * time_weights).mean(dim=1) * 5.0
            base_cost += cost_progress

        # ---- CBF 防撞 ----
        soft_margin = 0.15
        hard_margin = 0.05
        clamped_lidar = torch.clamp(lidar_abs, 0.0, 1.0)

        soft_violation = torch.clamp(soft_margin - clamped_lidar, min=0.0).sum(dim=(1, 2))
        base_cost += soft_violation * 20.0

        hard_violation = torch.clamp(hard_margin - clamped_lidar, min=0.0).sum(dim=(1, 2))
        is_feasible = hard_violation == 0

        cost_details = {
            'nav_costs': cost_progress,
            'smooth_costs': cost_heading,
            'col_costs': soft_violation,
            'boundary_penalty': boundary_penalty,
            'cur_rou': cur_rou,
        }

        return base_cost, is_feasible, hard_violation, cost_details

    def search_best_skill(self, obs_d, agent_global_info=None):
        LatentMPCPlanner._global_step += 1

        # 1. 编码观测
        obs_flat_t, norm_lidar_t, goal_dir = self._align_observation(obs_d)
        self._last_norm_lidar = norm_lidar_t

        # 2. 并行推演 16 skills
        lidar_abs, goal_abs = self._parallel_rollout(obs_flat_t, self.skill_vecs)

        # 3. 代价与约束
        base_cost, is_feasible, violation_amount, cost_details = \
            self._compute_cost_and_constraints(lidar_abs, goal_abs, agent_global_info, goal_dir)

        # 4. 寻优
        n_feasible = is_feasible.sum().item()
        if n_feasible > 0:
            valid_costs = torch.where(is_feasible, base_cost,
                                      torch.tensor(float('inf'), device=self.device))
            best_idx = torch.argmin(valid_costs).item()
        else:
            best_idx = torch.argmin(violation_amount).item()

        # 5. 缓存状态
        self._cached_skill = self.skill_vecs[best_idx].cpu().numpy()
        self.last_pred_state = lidar_abs[best_idx, 0, :].detach().cpu().numpy()
        self.last_skill_idx = best_idx

        return self.skill_vecs[best_idx].cpu().numpy()
