import torch
import numpy as np

from sim.replay_buffer import lidar2d_to_distances
from models.predictors.agent_dyn_predictor import frame_to_lidar_dist


class LatentMPPIPlanner:
    """
    MPPI (Model Predictive Path Integral) planner.

    Samples N action trajectories from a Gaussian distribution centered at the
    previous optimal action, rolls them out through the ForwardPredictor, and
    computes a cost-weighted average over trajectories.
    """

    _global_step = 0
    _print_interval = 1
    _stats = {
        'pred_lidar_mse': [], 'nav_cost': [], 'num_samples': 64,
        'col_cost': [], 'boundary_cost': [], 'cur_dist': [],
    }

    def __init__(self, forward_model, device, num_samples=64, horizon=10,
                 action_dim=5, temperature=1.0, noise_std=0.3,
                 pos_limit=500.0, ang_limit=np.pi, max_lidar_range=100.0):
        self.forward_model = forward_model
        self.device = device
        self.N = num_samples
        self.T = horizon
        self.action_dim = action_dim
        self.temp = temperature
        self.noise_std = noise_std
        self.POS_LIMIT = pos_limit
        self.ANG_LIMIT = ang_limit
        self.MAX_LIDAR_RANGE = max_lidar_range

        self.prev_action = np.zeros((horizon, action_dim), dtype=np.float32)

        # 全局地图
        self.nav_field_tensor = None
        self.world_offset = None
        self.nav_grid_size = 1.0
        self.map_height = 0
        self.map_width = 0

        self.last_pred_state = None

    def update_global_map(self, nav_field_array, world_offset, nav_grid_size):
        self.nav_field_tensor = torch.tensor(nav_field_array, dtype=torch.float32, device=self.device)
        self.world_offset = torch.tensor(world_offset, dtype=torch.float32, device=self.device)
        self.nav_grid_size = nav_grid_size
        self.map_height, self.map_width = self.nav_field_tensor.shape

    def _sample_actions(self):
        noise = torch.randn(self.N, self.T, self.action_dim, device=self.device) * self.noise_std
        base = torch.from_numpy(self.prev_action).to(self.device).unsqueeze(0)
        return base + noise

    def _frames_to_lidar_goal(self, frame_deltas, goal_deltas, init_frame, cur_goal_dir):
        """
        Convert predicted deltas to absolute lidar & goal states.

        frame_deltas: (N, T, 36, 6) — frame changes per step
        goal_deltas:  (N, T, 2)    — goal_dir changes per step
        init_frame:   (36, 6)       — reference frame (current, from obs)
        cur_goal_dir: (2,)          — initial goal_dir (normalized)

        Returns:
          lidar_abs: (N, T, 36) — absolute lidar distances [0, 1]
          goal_abs:  (N, T, 2)  — absolute goal_dir (normalized)
        """
        # Cumulative deltas + initial frame → absolute frames
        cumsum_deltas = torch.cumsum(frame_deltas, dim=1)  # (N, T, 36, 6)
        init_frame_t = torch.FloatTensor(init_frame).to(self.device).view(1, 1, 36, 6)
        frame_abs = init_frame_t + cumsum_deltas  # (N, T, 36, 6)

        # Lidar distances from lidar bin columns
        lidar_abs = frame_to_lidar_dist(frame_abs)  # (N, T, 36)

        # Goal from delta cumsum
        goal_abs = torch.cumsum(goal_deltas, dim=1)  # (N, T, 2)
        cur_goal_t = torch.FloatTensor(cur_goal_dir).to(self.device).view(1, 1, 2)
        goal_abs = cur_goal_t + goal_abs

        return lidar_abs, goal_abs

    def _compute_cost(self, lidar_abs, goal_abs, agent_global_info, cur_goal_dir):
        N = lidar_abs.size(0)
        horizon = lidar_abs.size(1)

        rou = goal_abs[:, :, 0] * self.POS_LIMIT
        etheta = goal_abs[:, :, 1] * self.ANG_LIMIT

        time_weights = torch.linspace(0.5, 1.5, steps=horizon, device=self.device)
        cost_heading = (torch.abs(etheta) * time_weights).mean(dim=1) * 0.5
        base_cost = cost_heading.clone()

        if agent_global_info is not None and self.nav_field_tensor is not None:
            target_x = agent_global_info['target_x']
            target_y = agent_global_info['target_y']
            cur_etheta = cur_goal_dir[1] * self.ANG_LIMIT
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
            base_cost = base_cost + cost_progress + boundary_penalty
        else:
            cost_progress = (rou * time_weights).mean(dim=1) * 5.0
            base_cost = base_cost + cost_progress

        # CBF hard constraint penalty
        safe_margin = 0.15
        clamped_lidar = torch.clamp(lidar_abs, 0.0, 1.0)
        hard_violation = torch.clamp(safe_margin - clamped_lidar, min=0.0).sum(dim=(1, 2))

        total_cost = base_cost + hard_violation * 100.0
        return total_cost

    def search_best_action(self, obs_frames, dynamics, cur_goal_dir,
                           agent_global_info=None):
        """
        MPPI: 采样 → 批量 rollout → cost → 加权融合。

        obs_frames: (3, 36, 6) numpy array (last frame = current reference)
        dynamics:   (2,) numpy array
        cur_goal_dir: (2,)
        """
        LatentMPPIPlanner._global_step += 1

        # Current frame (last in the multi-frame stack) is the reference
        init_frame = obs_frames[-1, :, :]  # (36, 6)

        # 1. 采样
        action_samples = self._sample_actions()  # (N, T, 5)

        # 2. 批量 rollout
        obs_t = torch.FloatTensor(obs_frames).to(self.device).unsqueeze(0)
        dyn_t = torch.FloatTensor(dynamics).to(self.device).unsqueeze(0)
        obs_batch = obs_t.repeat(self.N, 1, 1, 1)
        dyn_batch = dyn_t.repeat(self.N, 1)

        with torch.no_grad():
            frame_deltas, goal_deltas = self.forward_model(
                obs_batch, dyn_batch, action_samples)

        # 3. 提取 lidar + goal 绝对状态
        lidar_abs, goal_abs = self._frames_to_lidar_goal(
            frame_deltas, goal_deltas, init_frame, cur_goal_dir)

        # 4. Cost
        costs = self._compute_cost(lidar_abs, goal_abs, agent_global_info, cur_goal_dir)

        # 5. MPPI 加权
        costs_clamped = torch.clamp(costs, max=100.0)
        weights = torch.softmax(-costs_clamped / self.temp, dim=0)

        # 6. 加权融合首帧 action
        first_actions = action_samples[:, 0, :]
        best_action = (weights.unsqueeze(1) * first_actions).sum(dim=0)

        # 7. 更新 prev_action
        self._update_prev_action(action_samples, weights)

        # 8. 缓存
        best_idx = torch.argmax(weights).item()
        self.last_pred_state = lidar_abs[best_idx, 0, :].detach().cpu().numpy()

        if LatentMPPIPlanner._global_step % 100 == 0:
            best_cost = costs[best_idx].item()
            avg_cost = costs.mean().item()
            ba = best_action.cpu().numpy()
            print(f"[MPPI] Step {LatentMPPIPlanner._global_step} | "
                  f"best_cost={best_cost:.3f} avg_cost={avg_cost:.3f} | "
                  f"action=[{ba[0]:+.2f} {ba[1]:+.2f} {ba[2]:+.2f} "
                  f"{ba[3]:+.2f} {ba[4]:+.2f}]")

        return best_action.cpu().numpy()

    def _update_prev_action(self, action_samples, weights):
        weighted_all = (weights.view(-1, 1, 1) * action_samples).sum(dim=0)
        self.prev_action = weighted_all.cpu().numpy()
        self.prev_action[:-1] = self.prev_action[1:]
        self.prev_action[-1] = 0.0
