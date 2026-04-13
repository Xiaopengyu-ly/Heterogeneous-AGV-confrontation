This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where comments have been removed, empty lines have been removed, line numbers have been added, content has been compressed (code blocks are separated by ⋮---- delimiter).

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Code comments have been removed from supported file types
- Empty lines have been removed from all files
- Line numbers have been added to the beginning of each line
- Content has been compressed - code blocks are separated by ⋮---- delimiter
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
agent/agent_check.py
agent/agent_comm.py
agent/agent_config.yaml
agent/agent_core.py
agent/agent_get.py
agent/agent_loader.py
agent/agent_models.py
agent/bot_controller.py
agent/MapProcess.py
agent/RoutePlanning.py
attack_resolution_test.png
backup/sac_policy_stable.zip
backup/sac_policy.zip
backup/sac_policy0227.zip
backup/sac_policy0326.zip
backup/sac_policy0327.zip
CLAUDE.md
comm/msg_pool.py
dataset/dual_mapping_E.npy
dataset/dual_mapping_P.npy
dataset/dual_mapping_T.npy
firepower_dynamics_presentation.png
generate_config.py
kinematic_model_test.png
model_A_left_tower.pth
model/costom_model.py
monte_karol_sampling.py
requirements.txt
RL_train/train_initialize.py
RL_train/train_sim_core_lower.py
sac_policy.zip
sampler_and_process.py
sim_env/env_model.py
sim_env/map_data/d_spl_map0 copy 2.npy
sim_env/map_data/d_spl_map0 copy.npy
sim_env/map_data/d_spl_map0.npy
sim_env/map_data/d_spl_map1.npy
sim_env/map_data/grid_map0 copy 2.npy
sim_env/map_data/grid_map0 copy.npy
sim_env/map_data/grid_map0.npy
sim_env/map_data/grid_map1.npy
sim_env/map_data/start.py
sim_env/map_generator.py
sim_replay/0.pkl
sim_replay/1.pkl
sim_replay/10.pkl
sim_replay/100.pkl
sim_replay/101.pkl
sim_replay/102.pkl
sim_replay/103.pkl
sim_replay/104.pkl
sim_replay/105.pkl
sim_replay/106.pkl
sim_replay/107.pkl
sim_replay/108.pkl
sim_replay/109.pkl
sim_replay/11.pkl
sim_replay/110.pkl
sim_replay/111.pkl
sim_replay/112.pkl
sim_replay/113.pkl
sim_replay/114.pkl
sim_replay/115.pkl
sim_replay/116.pkl
sim_replay/117.pkl
sim_replay/118.pkl
sim_replay/119.pkl
sim_replay/12.pkl
sim_replay/120.pkl
sim_replay/121.pkl
sim_replay/122.pkl
sim_replay/123.pkl
sim_replay/124.pkl
sim_replay/125.pkl
sim_replay/126.pkl
sim_replay/127.pkl
sim_replay/128.pkl
sim_replay/129.pkl
sim_replay/13.pkl
sim_replay/130.pkl
sim_replay/131.pkl
sim_replay/132.pkl
sim_replay/133.pkl
sim_replay/134.pkl
sim_replay/135.pkl
sim_replay/136.pkl
sim_replay/137.pkl
sim_replay/138.pkl
sim_replay/139.pkl
sim_replay/14.pkl
sim_replay/140.pkl
sim_replay/141.pkl
sim_replay/142.pkl
sim_replay/143.pkl
sim_replay/144.pkl
sim_replay/145.pkl
sim_replay/146.pkl
sim_replay/147.pkl
sim_replay/148.pkl
sim_replay/149.pkl
sim_replay/15.pkl
sim_replay/150.pkl
sim_replay/151.pkl
sim_replay/152.pkl
sim_replay/153.pkl
sim_replay/154.pkl
sim_replay/155.pkl
sim_replay/156.pkl
sim_replay/157.pkl
sim_replay/158.pkl
sim_replay/159.pkl
sim_replay/16.pkl
sim_replay/160.pkl
sim_replay/161.pkl
sim_replay/162.pkl
sim_replay/163.pkl
sim_replay/164.pkl
sim_replay/165.pkl
sim_replay/166.pkl
sim_replay/167.pkl
sim_replay/168.pkl
sim_replay/169.pkl
sim_replay/17.pkl
sim_replay/170.pkl
sim_replay/171.pkl
sim_replay/172.pkl
sim_replay/173.pkl
sim_replay/174.pkl
sim_replay/175.pkl
sim_replay/176.pkl
sim_replay/177.pkl
sim_replay/178.pkl
sim_replay/179.pkl
sim_replay/18.pkl
sim_replay/180.pkl
sim_replay/181.pkl
sim_replay/182.pkl
sim_replay/183.pkl
sim_replay/184.pkl
sim_replay/185.pkl
sim_replay/186.pkl
sim_replay/187.pkl
sim_replay/188.pkl
sim_replay/189.pkl
sim_replay/19.pkl
sim_replay/190.pkl
sim_replay/191.pkl
sim_replay/192.pkl
sim_replay/193.pkl
sim_replay/194.pkl
sim_replay/195.pkl
sim_replay/196.pkl
sim_replay/197.pkl
sim_replay/198.pkl
sim_replay/199.pkl
sim_replay/2.pkl
sim_replay/20.pkl
sim_replay/21.pkl
sim_replay/22.pkl
sim_replay/23.pkl
sim_replay/24.pkl
sim_replay/25.pkl
sim_replay/26.pkl
sim_replay/27.pkl
sim_replay/28.pkl
sim_replay/29.pkl
sim_replay/3.pkl
sim_replay/30.pkl
sim_replay/31.pkl
sim_replay/32.pkl
sim_replay/33.pkl
sim_replay/34.pkl
sim_replay/35.pkl
sim_replay/36.pkl
sim_replay/37.pkl
sim_replay/38.pkl
sim_replay/39.pkl
sim_replay/4.pkl
sim_replay/40.pkl
sim_replay/41.pkl
sim_replay/42.pkl
sim_replay/43.pkl
sim_replay/44.pkl
sim_replay/45.pkl
sim_replay/46.pkl
sim_replay/47.pkl
sim_replay/48.pkl
sim_replay/49.pkl
sim_replay/5.pkl
sim_replay/50.pkl
sim_replay/51.pkl
sim_replay/52.pkl
sim_replay/53.pkl
sim_replay/54.pkl
sim_replay/55.pkl
sim_replay/56.pkl
sim_replay/57.pkl
sim_replay/58.pkl
sim_replay/59.pkl
sim_replay/6.pkl
sim_replay/60.pkl
sim_replay/61.pkl
sim_replay/62.pkl
sim_replay/63.pkl
sim_replay/64.pkl
sim_replay/65.pkl
sim_replay/66.pkl
sim_replay/67.pkl
sim_replay/68.pkl
sim_replay/69.pkl
sim_replay/7.pkl
sim_replay/70.pkl
sim_replay/71.pkl
sim_replay/72.pkl
sim_replay/73.pkl
sim_replay/74.pkl
sim_replay/75.pkl
sim_replay/76.pkl
sim_replay/77.pkl
sim_replay/78.pkl
sim_replay/79.pkl
sim_replay/8.pkl
sim_replay/80.pkl
sim_replay/81.pkl
sim_replay/82.pkl
sim_replay/83.pkl
sim_replay/84.pkl
sim_replay/85.pkl
sim_replay/86.pkl
sim_replay/87.pkl
sim_replay/88.pkl
sim_replay/89.pkl
sim_replay/9.pkl
sim_replay/90.pkl
sim_replay/91.pkl
sim_replay/92.pkl
sim_replay/93.pkl
sim_replay/94.pkl
sim_replay/95.pkl
sim_replay/96.pkl
sim_replay/97.pkl
sim_replay/98.pkl
sim_replay/99.pkl
sim/config_data/config0 copy 2.yaml
sim/config_data/config0.yaml
sim/config_data/config1.yaml
sim/physics_engine.py
sim/sim_initialize.py
stage_presentation_mapping.png
start.py
task_matching_radar_test.png
train_SAC.py
Two_Tower_Evaluation.py
version.yaml
vis/agentvis.py
vis/base_vis.py
vis/controlled_window.py
vis/info_panel.py
vis/replay_buffer.py
vis/sim_controller.py
vis/smokevis.py
```

# Files

## File: agent/agent_check.py
````python
class CheckSystem
⋮----
def __init__(self, agent)
def check_hit(self)
⋮----
distance_to_r_point = np.sqrt((self.agent.position[0] - self.agent.r_point[0])**2 + (self.agent.position[1] - self.agent.r_point[1])**2)
⋮----
def check_env(self, env_feedback : dict )
⋮----
live_id = env_feedback['live_ids']
⋮----
channel_dict = env_feedback['channel_dict']
⋮----
# 【注意】这里调用 agent 的接口，agent 会自动转发给 comm_system
⋮----
# 烟雾区域反馈
⋮----
# 障碍物扇区信息反馈
⋮----
def check_rtPlan(self)
⋮----
# 判断路径规划必要性
'''
            确保路径规划只在一种情况下被调用：
            车辆视线/未来行驶路径上被静态障碍物遮挡，对动态障碍物（一般为其他车辆）不使用A*规划路径
        '''
r_target = self.position - self.t_pos
dist_target = np.linalg.norm(r_target)
⋮----
obs_array = np.array(self.local_obstacles)
r_obs = self.position - obs_array
⋮----
dist_obs = np.linalg.norm(r_obs[index, :])
shade_angle = np.dot(r_obs[index, :], r_target) / (dist_target * dist_obs)
⋮----
# print("agent ", self.id, "pos at", self.position, "need rtplan \n")
````

## File: agent/agent_comm.py
````python
class CommSystem
⋮----
def __init__(self, agent)
def broadcast_msg(self, pool: MsgPool)
⋮----
msg = {
⋮----
def recieve_msg(self, pool: MsgPool)
⋮----
channel_id = self.agent.neighbors_info["channelid"][f"{nid}"]
msg = pool.download(channel_id)
⋮----
target_id = self.agent.targets_id
channel_id = self.agent.targets_info["channelid"][f"{target_id}"] if not target_id == 0 else "Empty"
⋮----
cannon_target_id = self.agent.cannon_targets_id
channel_id = self.agent.cannon_targets_info["channelid"][f"{cannon_target_id}"] if not cannon_target_id == 0 else "Empty"
⋮----
def upload_toPanel(self, pool: MsgPool)
````

## File: agent/agent_config.yaml
````yaml
default:
  v_max: 133.41
  r_turn_min: 24.76
  s_max: 10000
  sense_field: 278.48
  sense_angle_deg: 39.11
  sense_variance: 0.13
  attack_range: 194
  cannon_w_max_deg: 1138.22
  launch_delay: 12
  num_per_launch: 1
  attk_radius: 15.28
  attk_variance: 0.1
  cannon_capacity: 10
  smoke_capacity: 5
  reflective_surface: 0.89
  exposed_area: 7.07
  decision_delay: 4
  task_preference: 1
  task_assignment: 1
  weapon_assignment: 1
  connect_dist: 104
water:
  v_max: 109.04
  r_turn_min: 23.65
  s_max: 10000
  sense_field: 341.66
  sense_angle_deg: 40.06
  sense_variance: 0.29
  attack_range: 184
  cannon_w_max_deg: 832.62
  launch_delay: 8
  num_per_launch: 1
  attk_radius: 19.23
  attk_variance: 0.23
  cannon_capacity: 9
  smoke_capacity: 5
  reflective_surface: 1.1
  exposed_area: 7.07
  decision_delay: 3
  task_preference: 1
  task_assignment: 1
  weapon_assignment: 0
  connect_dist: 98
land:
  v_max: 132.75
  r_turn_min: 23.92
  s_max: 10000
  sense_field: 305.5
  sense_angle_deg: 38.33
  sense_variance: 0.12
  attack_range: 185
  cannon_w_max_deg: 954.1
  launch_delay: 13
  num_per_launch: 1
  attk_radius: 15.88
  attk_variance: 0.25
  cannon_capacity: 10
  smoke_capacity: 5
  reflective_surface: 0.91
  exposed_area: 7.07
  decision_delay: 2
  task_preference: 4
  task_assignment: 0
  weapon_assignment: 0
  connect_dist: 109
````

## File: agent/agent_core.py
````python
def normalize_angle(angle)
class Agent
⋮----
def __init__(self, id: int, position: np.ndarray, velocity: np.ndarray, dT: float = 0.02, side=0,)
⋮----
config = load_agent_config(config_name = "default")
⋮----
dx = self.TARGET_POS[0] - self.position[0]
dy = self.TARGET_POS[1] - self.position[1]
⋮----
def check_hit(self)
def check_env(self, env_feedback)
def get_connect(self, pool)
def get_init_parameters(self, channel_id, target_distance)
def get_net_neighbors(self, neighbors_dict, channel_dict, formation_structure, targets_dict, cannon_targets_dict)
def get_trajectory(self)
def get_grid_map(self, map_layers=None, grid_size=1)
def get_route_point(self, case="A_star", action=np.array([0,0,0,0,0]))
def broadcast_msg(self, pool)
def recieve_msg(self, pool)
def upload_toPanel(self, pool)
def update_movement_model(self)
def update_sense_model(self)
def update_smoke_model(self)
def update_task_allocate_model(self)
def update_attack_model(self)
def update_msg(self)
def update_model(self)
def update(self, global_time: float = None)
````

## File: agent/agent_get.py
````python
class DataSystem
⋮----
def __init__(self, agent)
def get_connect(self, pool: MsgPool)
def get_init_parameters(self, channel_id: int, target_distance: np.ndarray)
def get_net_neighbors(self, neighbors_dict: dict, channel_dict: dict, formation_structure: dict, targets_dict: dict, cannon_targets_dict: dict)
⋮----
nodes = neighbors_dict[f"{self.agent.id}"]
⋮----
nid = nodes[iter][0]
weight = nodes[iter][1]
⋮----
# Targets Logic
nodes = targets_dict[f"{self.agent.id}"]
⋮----
# Cannon Targets Logic
nodes = cannon_targets_dict[f"{self.agent.id}"]
⋮----
def get_trajectory(self)
def get_grid_map(self, map_layers: MapGenerator = None, grid_size: int = 1)
def _angle_diff(self, a, b)
⋮----
diff = a - b
diff = (diff + np.pi) % (2 * np.pi) - np.pi
⋮----
def get_route_point(self, case: str = "A_star", action : np.ndarray = np.array([0,0,0,0,0]) )
⋮----
block_h = height // d_height
block_w = width // d_width
grid_size = self.agent.grid_size
panel_center = np.array([width * grid_size / 2, height * grid_size / 2])
start_pos = np.array([
goal_pos = np.array([
path = self.agent.planner.search(start_pos, goal_pos)
⋮----
raw_state = self.agent.planner.extract_waypoints(path)
⋮----
raw_state = action
````

## File: agent/agent_loader.py
````python
def load_agent_config(config_name="default", config_path = None)
⋮----
config_path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
⋮----
config_all = yaml.safe_load(f)
⋮----
config = config_all[config_name]
# 自动转换角度单位
````

## File: agent/agent_models.py
````python
def normalize_angle(angle)
class BehaviorSystem(MapProcesser)
⋮----
def __init__(self, agent)
⋮----
@property
    def position(self): return self.agent.position
⋮----
@property
    def grid_map(self): return self.agent.grid_map
⋮----
@property
    def grid_size(self): return self.agent.grid_size
⋮----
@property
    def smoke_zones(self): return self.agent.smoke_zones
⋮----
@property
    def smoke_attenuation(self): return self.agent.smoke_attenuation
⋮----
@property
    def sense_field(self): return self.agent.sense_field
⋮----
@property
    def local_obstacles(self): return self.agent.local_obstacles
⋮----
@local_obstacles.setter
    def local_obstacles(self, value): self.agent.local_obstacles = value
⋮----
@property
    def obs_sector(self): return self.agent.obs_sector
def move_logic_model(self)
def guidance_control(self)
⋮----
guide_state = self.agent.r_point if (self.agent.r_point is not None and np.all(np.isfinite(self.agent.r_point))) \
⋮----
def Kinematic_model(self, v_left: float, v_right: float)
def task_allocate_model(self)
def smoke_model(self)
⋮----
can_smoke = (
⋮----
def _angle_diff(self, a, b)
⋮----
diff = a - b
diff = (diff + np.pi) % (2 * np.pi) - np.pi
⋮----
def cannon_turning_control(self, error)
⋮----
us1 = np.abs(error) ** (0.2)
us2 = np.abs(error) ** (0.8)
u = us1 + us2
⋮----
max_w = self.agent.cannon_w_max
⋮----
def attack_model(self)
⋮----
target_dir = -(self.agent.position - self.agent.attk_pos)
target_angle = np.arctan2(target_dir[1], target_dir[0])
error_eta = self._angle_diff(target_angle, self.agent.cannon_theta)
target_dist = np.linalg.norm(target_dir)
can_fire = (
target_outof_sight = (
⋮----
sigma = self.agent.attk_variance
noise = np.random.normal(loc=0.0, scale=sigma, size=2)
attk_pos = self.agent.attk_pos + noise
⋮----
error_eta_normal =  (self.agent.theta - self.agent.cannon_theta)
⋮----
def sense_model(self)
def update_target(self)
⋮----
p_pos = self.agent.targets_info["position"][f"{self.agent.targets_id}"]
max_sense_range = self.agent.sense_field
max_sense_angle = self.agent.sense_angle
# 新增判断：如果坐标还是初始的 [0, 0]，说明还没收到通信更新，跳过探测
⋮----
Tpos = self.target_detect_model(p_pos, max_sense_range, max_sense_angle)
⋮----
attk_pos = self.agent.cannon_targets_info["position"][f"{self.agent.cannon_targets_id}"]
⋮----
Tpos = self.target_detect_model(attk_pos, max_sense_range, max_sense_angle)
⋮----
def target_detect_model(self, t_pos, max_sense_range, max_sense_angle)
⋮----
target_dir = -(self.agent.position - t_pos)
t_angle = np.arctan2(target_dir[1], target_dir[0]) - self.agent.theta
t_dist = np.linalg.norm(t_pos - self.agent.position)
⋮----
delta = self.block_and_smoke_check(t_pos)
⋮----
sense_P = delta * self.agent.sense_basic_P0
⋮----
sense_P = delta * self.agent.sense_basic_P0 * np.exp(-self.agent.sense_P_attenuation * (t_dist - max_sense_range)**2)
⋮----
sigma = self.agent.sense_variance
⋮----
observed_pos = t_pos + noise
````

## File: agent/bot_controller.py
````python
def normalize_angle(angle)
def guidance_with_obstacle_avoidance(agent, ref_point)
def feedback_control(agent, ref_point)
⋮----
e_x = ref_point[0]
e_y = ref_point[1]
e_theta = ref_point[2]
v_r = ref_point[3]
omega_r = ref_point[4]
k1 = 4.0
k2 = 10.0
k3 = 15.0
v_nom = v_r * np.cos(e_theta) + k1 * e_x
omega_nom = omega_r + k2 * v_r * e_y + k3 * v_r * np.sin(e_theta)
⋮----
def safety_filter(agent, v_ref, w_ref)
⋮----
v_constraints = []
⋮----
v_safe = min(v_constraints)
v_safe = max(0, v_safe)
dv = v_safe - v_ref
⋮----
eps = agent.v_min
⋮----
w_safe = min(w_ref, v_safe / agent.r_turn_min)
⋮----
w_safe = min(w_ref, eps / agent.r_turn_min)
dw = w_safe - w_ref
⋮----
center_angle = agent.sector_center[i]
gamma = 30
d_min = 5
tau = 0.001
⋮----
v_sector = gamma * (agent.obs_sector[i] - d_min) / np.cos(center_angle)
⋮----
v_sector = (1 - gamma * tau) * (tau * (agent.r_point[3] - agent.prev_r_point[3]) / agent.dT + gamma * (agent.obs_sector[i] - d_min) / np.cos(center_angle) )
````

## File: agent/MapProcess.py
````python
class MapProcesser
⋮----
def chord_length_opt(self, p1, p2, cx, cy, r)
⋮----
r2 = r * r
dx = x2 - x1
dy = y2 - y1
A = dx * dx + dy * dy
⋮----
invA = 1.0 / A
fx = x1 - cx
fy = y1 - cy
d1_sq = fx * fx + fy * fy
gx = x2 - cx
gy = y2 - cy
d2_sq = gx * gx + gy * gy
⋮----
dot = fx * dx + fy * dy
t_proj = -dot * invA
t_clamped = t_proj
⋮----
t_clamped = 0.0
⋮----
t_clamped = 1.0
cx_closest = x1 + t_clamped * dx
cy_closest = y1 + t_clamped * dy
dist2 = (cx_closest - cx) * (cx_closest - cx) + (cy_closest - cy) * (cy_closest - cy)
⋮----
B = 2.0 * dot
C = d1_sq - r2
disc = B * B - 4.0 * A * C
⋮----
sqrt_disc = np.sqrt(disc)
inv_2A = 0.5 * invA
t1 = (-B - sqrt_disc) * inv_2A
t2 = (-B + sqrt_disc) * inv_2A
t_low = t1 if t1 < t2 else t2
t_high = t2 if t1 < t2 else t1
⋮----
t_low = 0.0
⋮----
t_high = 1.0
⋮----
def block_and_smoke_check(self, p_pos)
⋮----
smoke_zones = self.smoke_zones
attenuation_coeff = self.smoke_attenuation
x0 = int(self.position[0] / self.grid_size)
y0 = int(self.position[1] / self.grid_size)
x1 = int(p_pos[0] / self.grid_size)
y1 = int(p_pos[1] / self.grid_size)
dx = abs(x1 - x0)
dy = abs(y1 - y0)
sx = -1 if x0 > x1 else 1
sy = -1 if y0 > y1 else 1
err = dx - dy
⋮----
e2 = 2 * err
⋮----
attenuation = 1
p1 = self.position
p2 = p_pos
⋮----
length_in_smoke = self.chord_length_opt(p1, p2, zones[0][0], zones[0][1], zones[2])
⋮----
def update_obstacles(self, n_closest=4, n_comp=2, connectivity=8)
⋮----
mode = "sector"
⋮----
grid_size = self.grid_size
panel_center = np.array([width * grid_size * 0.5, height * grid_size * 0.5])
grid_x = int((self.position[0] + panel_center[0]) / grid_size)
grid_y = int((self.position[1] + panel_center[1]) / grid_size)
r = int(self.sense_field / grid_size)
scanned_obstacles = []
⋮----
dx = i - grid_x
dx2 = dx * dx
⋮----
dy = j - grid_y
⋮----
def get_connected_components(points)
⋮----
visited = set()
components = []
point_set = set(points)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else \
⋮----
q = deque([p])
comp = []
⋮----
cur = q.popleft()
⋮----
nb = (cur[0] + dx, cur[1] + dy)
⋮----
components = get_connected_components(scanned_obstacles)
representative_points = []
⋮----
closest_pts = heapq.nsmallest(
⋮----
dist2 = (pt[0] - grid_x) ** 2 + (pt[1] - grid_y) ** 2
⋮----
closest = heapq.nsmallest(n_closest, representative_points)
````

## File: agent/RoutePlanning.py
````python
class AStarAPF
⋮----
def __init__(self, grid_map, lam=5.0, gamma=2.0)
def heuristic(self, a, b)
def potential(self, x, y)
⋮----
d = self.dist[y, x]
dist_sq = d ** 2
mu = 0
sigma = 7
repulsion_cost = 3000 * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((dist_sq - mu) / sigma) ** 2)
⋮----
def is_valid(self, x, y)
def search(self, start_npy : np.array, goal_npy : np.array)
⋮----
start = (start_npy[0],start_npy[1])
⋮----
goal =  (goal_npy[0], goal_npy[1])
open_set = []
⋮----
came_from = {}
cost_so_far = {start: 0}
index = 1
⋮----
step_cost = np.hypot(dx, dy)
turn_penalty = 0
⋮----
turn_penalty = self.gamma
new_cost = cost + step_cost + turn_penalty
neighbor = (nx, ny)
⋮----
priority = (
⋮----
path = []
cur = goal
⋮----
cur = came_from[cur]
⋮----
def extract_waypoints(self, path)
⋮----
DEFAULT_V = 1
DEFAULT_W = 0.0
⋮----
dx = path[1][0] - path[0][0]
dy = path[1][1] - path[0][1]
theta = np.arctan2(dy, dx)
⋮----
prev_dx = path[1][0] - path[0][0]
prev_dy = path[1][1] - path[0][1]
⋮----
dx = path[i][0] - path[i-1][0]
dy = path[i][1] - path[i-1][1]
⋮----
theta = np.arctan2(prev_dy, prev_dx)
⋮----
def main()
⋮----
env = train_initialize(0)
grid_map = env.engine.grid_map
⋮----
grid_size = env.engine.grid_size
panel_center = np.array([width * grid_size / 2, height * grid_size / 2])
start = np.array([
goal = np.array([
planner = AStarAPF(grid_map, lam=5.0, gamma=2.0)
path = planner.search(start, goal)
waypoints = planner.extract_waypoints(path)
````

## File: CLAUDE.md
````markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent swarm simulation and reinforcement learning training framework for tactical UAV scenarios with:
- Physics-based agent dynamics with smoke screens, sensing, and weapon systems
- Hierarchical RL architecture (lower-level navigation + higher-level task allocation)
- PyQt5 visualization for real-time simulation and replay

## Commands

### Environment Setup
```bash
conda activate <your-env>  # Requires conda, see .vscode/settings.json
pip install -r requirements.txt
```

### Training
```bash
python train_SAC.py           # SAC training with stable-baselines3 (12 parallel envs)
python backup/train_ray_sac.py # Ray RLlib training with curriculum learning
```

### Inference & Visualization
```bash
python start.py               # Run simulation with pre-trained SAC policy
```

### Testing
```bash
pytest test/test_env.py       # Environment reset/step smoke test
```

## Architecture

### Core Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Visualization Layer (vis/)                                  │
│  - ControlledVisWindow: Main GUI window                      │
│  - SimulationController: Real-time vs replay mode            │
│  - ReplayBuffer: Transition storage                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RL Environment Adapter (RL_train/train_sim_core_lower.py)   │
│  - RLEnvAdapter: Gymnasium wrapper around PhysicsEngine      │
│  - Action space: 15-dim (5-dim × 3-step chunking)            │
│  - Reward: Potential-based + CBF + smoothness                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Physics Engine (sim/physics_engine.py)                      │
│  - step_physics(): Controller execution                      │
│  - env_update(): Smoke, attack resolution, obstacle sensing  │
│  - reset_engine(): State restoration from snapshot           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent System (agent/)                                       │
│  - Agent: Main facade with 4 components:                     │
│    • BehaviorSystem: Move/Sense/Attack models                │
│    • DataSystem: Grid map, route planning (A*/RL)            │
│    • CheckSystem: Hit detection, env feedback                │
│    • CommSystem: Message pool broadcasting                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Environment Model (sim_env/env_model.py)                    │
│  - env_model: Base class for smoke, attack, sensing logic    │
│  - MapGenerator: Procedural obstacle generation              │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns

**Configuration Management**: `generate_config.py` generates YAML configs with:
- Map layout (256×256 grid, downsampled to 32×32)
- Agent profiles, positions, communication tensors
- Target directions and formation structures

**Agent Profiles**: Loaded via `agent_loader.py` from `agent_config/` directory with parameters for velocity, sensing, weapons, and communication.

**Action Chunking**: RL policies output 15-dim vectors (5 dims × 3 steps), but only the first 5-dim step is executed per environment step for smoother control.

**Multi-Agent Support**: `SubprocVecEnv` with 12 parallel environments for training; per-agent dict observations/actions in petting-zoo style.

### File Structure
```
├── agent/          # Agent logic (core, comm, check, get, models, loader)
├── comm/           # Message pool for inter-agent communication
├── model/          # Neural network models (VQVAE, custom)
├── RL_train/       # RL environment adapters and training init
├── sim/            # Physics engine and simulation init
├── sim_env/        # Map generator and base env model
├── test/           # Unit tests
├── vis/            # Visualization (PyQt5 windows, controllers)
├── backup/         # Archived training scripts (PPO, SAC, Ray)
├── generate_config.py  # Config/map generation
├── train_SAC.py    # Main SAC training entry point
└── start.py        # Inference/visualization entry point
```
````

## File: comm/msg_pool.py
````python
class MsgPool()
⋮----
def __init__ (self, channel_num : int = 500)
def check(self)
def upload(self, channelid : int, msg : dict)
⋮----
elif channelid == 0: # 解析全局信息
id = msg["id"]
⋮----
def download(self , channelid : int)
def main()
⋮----
# 测试信息收发
pool1 = MsgPool(5)
a1 = {"id" : 1 ,"b" : [1,2]}
a2 = {"id" : 2 ,"b" : [1,3]}
a3 = {"id" : 3 ,"b" : [1,4]}
a4 = {"id" : 4 ,"b" : [1,5]}
⋮----
b = pool1.download(0)
c = []
position = np.array([1,1])
m_obs = np.array([d["b"] for d in b.values()])
⋮----
xy_tuple = (obs[0],obs[1])
````

## File: generate_config.py
````python
def generate_or_load_map(width: int, height: int, full_size_map_path: str, d_sample_map_path : str, map_Fixed : bool, isBlank : bool, d_sample_hw : np.ndarray) -> np.ndarray
⋮----
obs_map = np.load(full_size_map_path)
d_spl_map = np.load(d_sample_map_path)
⋮----
map_gen = MapGenerator(width, height, isBlank)
⋮----
obs_map = map_gen.obs_map
d_spl_map = map_gen.down_sampled_map
⋮----
def get_random_positions(grid_map: np.ndarray, agent_num: int, red_num: int, grid_size: float, min_clearance: int = 5) -> np.ndarray
⋮----
"""在无障碍区域生成初始物理坐标，前 red_num 个在上半场，其余在下半场"""
⋮----
blue_num = agent_num - red_num
⋮----
# 建立两个独立的备选库
valid_cells_top = []
valid_cells_bottom = []
⋮----
# 1. 检查中心安全禁区 (曼哈顿距离)
⋮----
# 2. 过滤障碍点
⋮----
# 3. 过滤 Clearance 不足的区域
window = grid_map[
⋮----
# 4. 核心改动：按行号归类到上下半场
⋮----
# 分别进行容量安全校验
⋮----
# 分别独立无放回抽样
selected_top = random.sample(valid_cells_top, red_num)
selected_bottom = random.sample(valid_cells_bottom, blue_num)
# 合并结果，保证前 red_num 个必定来自上半场
selected = selected_top + selected_bottom
# 转换为物理坐标
positions = [
⋮----
# ======================
# 2. 通信与关系张量
⋮----
REL_COOP = np.array([1, 0, 0], dtype=int)
REL_ATTK = np.array([0, 0, 1], dtype=int)
REL_CHASE = np.array([0, 1, 0], dtype=int)
def build_com_tensor(agent_num: int, agent_side: np.ndarray) -> np.ndarray
⋮----
"""构建通信/关系张量 (N, N, 3)"""
com_tensor = np.zeros((agent_num, agent_num, 3), dtype=int)
⋮----
# 3. 目标方向角 & target_distance
⋮----
def compute_target_angles(agent_side: np.ndarray, agent_id: List[int]) -> Dict[str, List[float]]
⋮----
"""为每个 agent 分配目标方向（同阵营均匀分布，敌对阵营对面）"""
sides = np.unique(agent_side)
side_agents = {side: [] for side in sides}
⋮----
target_distance = {}
⋮----
n = len(ids)
⋮----
angle = 2 * np.pi * idx / n
if side == 0:  # 假设 0 是蓝方，从对面出发
⋮----
angle = angle % (2 * np.pi)
dx = float(np.cos(angle))
dy = float(np.sin(angle))
⋮----
# 4. 编队结构
⋮----
"""仅对合作邻居构建编队偏移"""
formation = {}
agent_num = len(agent_id)
id_to_idx = {aid: i for i, aid in enumerate(agent_id)}
⋮----
if agent_side[i] == 1:  # 仅蓝方有编队
dx = x_sq * np.sign(pos[j, 0] - pos[i, 0])
dy = y_sq * np.sign(pos[j, 1] - pos[i, 1])
⋮----
key = f"{agent_id[i]}-{agent_id[j]}"
⋮----
# 5. Agent 配置分配（核心新增）
⋮----
def assign_agent_profiles(agent_num: int, agent_side: np.ndarray) -> List[str]
⋮----
"""
    为每个 agent 分配配置 profile 名称。
    可根据 side 或随机选择预定义类型。
    """
profiles = []
⋮----
# 红方
# profile = random.choice(["blue_recon", "blue_comm"])
profile = "water"
⋮----
profile = "land"
⋮----
def generate_config(i)
⋮----
grid_size = int(3)
d_sample_hw = np.array([32,32])
side_num = {"red" : 3, "blue" : 3}
agent_num = sum(side_num.values())
⋮----
version_config = yaml.safe_load(f)
version = version_config["version"]["id"]
config_save_path = f"E:/code/v3/version{version}/sim/config_data/config{i}.yaml"
full_size_map_path= f"E:/code/v3/version{version}/sim_env/map_data/grid_map{i}.npy"
d_sample_map_path = f"E:/code/v3/version{version}/sim_env/map_data/d_spl_map{i}.npy"
map_Fixed = True
isBlank = True if random.random() < 0.2 else False
⋮----
agent_dT = [0.02] * agent_num
agent_id = random.sample(range(1, agent_num * 5), agent_num)
pos = get_random_positions(d_spl_map, agent_num, side_num['red'], grid_size * width // d_sample_hw[0], min_clearance=2)
theta = [[2*random.random()-1 for _ in range(2)] for _ in range(agent_num)]
agent_side = np.array([0] * side_num["red"] + [1] * side_num["blue"])
com_tensor = build_com_tensor(agent_num, agent_side)
target_distance = compute_target_angles(agent_side, agent_id)
formation_structure = build_formation_structure(agent_id, agent_side, pos, com_tensor)
agent_profiles = assign_agent_profiles(agent_num, agent_side)
config = {
⋮----
num = generate_config(0)
````

## File: model/costom_model.py
````python
class MARL_CNN_Model(TorchModelV2, nn.Module)
⋮----
def __init__(self, obs_space, action_space, num_outputs, model_config, name)
⋮----
orig_space = getattr(obs_space, "original_space", obs_space)
⋮----
dummy_map = torch.zeros(1, 1, self.map_shape[0], self.map_shape[1])
⋮----
combined_dim = self.cnn_out_dim + 256
⋮----
def forward(self, input_dict, state, seq_lens)
⋮----
obs = input_dict["obs"]
map_input = obs["global_state"]["map"].unsqueeze(1)
cnn_feat = self.cnn_branch(map_input)
lidar = obs["lidar"]
pos = obs["agentpos"]
att = obs["agentatt"]
goal = obs["global_state"]["goal"]
others = obs["global_state"]["neighbors"].flatten(start_dim=1)
vec_input = torch.cat([lidar, pos, att, goal, others], dim=1)
vec_feat = self.vector_encoder(vec_input)
combined = torch.cat([cnn_feat, vec_feat], dim=1)
logits = self.action_head(combined)
⋮----
def value_function(self)
````

## File: monte_karol_sampling.py
````python
def sample_agent_profile(name: str)
⋮----
config = {}
def rnd(x)
⋮----
def generate_agent_config(output_path="agent/agent_config.yaml", profiles=None)
⋮----
profiles = ["default"]
full_config = {}
⋮----
# 确保输出目录存在
⋮----
# 写入 YAML
````

## File: requirements.txt
````
cffi==2.0.0
clarabel==0.11.1
cvxpy==1.7.3
Jinja2==3.1.6
joblib==1.5.2
MarkupSafe==3.0.3
numpy==2.3.4
osqp==1.0.5
pycparser==2.23
PyQt5==5.15.11
PyQt5-Qt5==5.15.2
PyQt5_sip==12.17.1
PyYAML==6.0.3
scipy==1.16.3
scs==3.2.9
````

## File: RL_train/train_initialize.py
````python
def train_initialize(i)
⋮----
version_config = yaml.safe_load(f)
version = version_config["version"]["id"]
config_path = f"E:/code/v3/version{version}/sim/config_data/config{i}.yaml" if isinstance(i, int) else i
⋮----
config = yaml.safe_load(f)
width = config["map"]["width"]
height = config["map"]["height"]
grid_size = config["map"]["grid_size"]
save_path = config["map"]["save_path"]
map_layers = MapGenerator(width, height)
⋮----
engine = PhysicsEngine(map_layers=map_layers, grid_size=grid_size, dT=0.01)
msg_pool = MsgPool()
⋮----
agent_num = config["agents"]["num"]
agent_id = config["agents"]["id"]
pos = np.array(config["agents"]["pos"])
agent_dT = config["agents"]["dT"]
agent_side = np.array(config["agents"]["side"])
theta = np.array(config["agents"]["theta"])
agents = []
⋮----
com_tensor = np.array(config["agents"]["com_tensor"])
init_channel = msg_pool.channel_id[0:agent_num]
target_distance = {k: np.array(v) for k, v in config["target_distance"].items()}
formation_structure = {
⋮----
env = RLEnvAdapter(engine, agent_id)
⋮----
def main()
⋮----
sim = train_initialize(0)
````

## File: RL_train/train_sim_core_lower.py
````python
class RLEnvAdapter(gym.Env)
⋮----
def __init__(self, engine: PhysicsEngine, agent_id : np.ndarray)
⋮----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
⋮----
def __getattr__(self, name)
def _angle_diff(self, a, b)
⋮----
diff = a - b
diff = (diff + np.pi) % (2 * np.pi) - np.pi
⋮----
def _check_termination(self, agent, obs)
⋮----
ex = agent.t_pos[0] - agent.position[0]
ey = agent.t_pos[1] - agent.position[1]
dist = np.hypot(ex, ey)
terminated_success = bool(dist < 10)
stuck = self._is_stuck(agent, window=30, pos_threshold=0.2)
terminated_stuck = stuck and not terminated_success
⋮----
def _is_stuck(self, agent, window=5, pos_threshold=2)
⋮----
traj = np.array(agent.position_history)
max_disp = np.max(np.linalg.norm(traj - traj[0], axis=1))
⋮----
def _action_post_process(self, raw_actions)
⋮----
"""
        注意：此处接收的 raw_actions 必须是单步的 (5,) 向量
        """
rou = 15.0 * (raw_actions[0] + 1) + 1.0
# 【修改点 1】将 0.25 * np.pi 扩大为 0.6 * np.pi，提供更大的横向机动自由度
phi = 0.25 * np.pi * raw_actions[1]
e_theta = 0.25 * np.pi * raw_actions[2]
v_r = (raw_actions[3] + 1)
w_r = raw_actions[4]
e_x = rou * np.cos(phi)
e_y = rou * np.sin(phi)
actions = np.array([e_x, e_y, e_theta, v_r, w_r], dtype=np.float32)
⋮----
def get_agent_observation(self, agent)
⋮----
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
semantic_obs = np.zeros(5, dtype=np.float32)
⋮----
def _compute_reward(self, agent, obs, action, terminated_success, terminated_stuck)
⋮----
goal_potential = dist
⋮----
delta_goal = 0.0
⋮----
delta_goal = self.prev_potential - goal_potential
⋮----
r_goal = - 0.2 * goal_potential + 10.0 * delta_goal
r_heading = 5.0 * np.cos(etheta) if abs(etheta) < np.pi/2 else -1.0
⋮----
dv = getattr(agent, 'dv', 0.0)
dw = getattr(agent, 'dw', 0.0)
r_cbf = - 5.0 * abs(dv) - 2.0 * abs(dw)
r_smooth_w = - abs(agent.w) if abs(agent.v) > 50 else - 0.1 * abs(agent.w)
⋮----
delta_v = 0.0
⋮----
delta_v = agent.v - self.prev_v
⋮----
r_smooth_v = delta_v + agent.v if abs(agent.v) < 50 else delta_v
r_consistency = 0.0
⋮----
delta_e = np.array([cur_ex - prev_ex, cur_ey - prev_ey, self._angle_diff(cur_theta, prev_theta)])
delta_e_pred = np.array([prev_vr * np.cos(prev_theta) * self.dT, prev_vr * np.sin(prev_theta) * self.dT, prev_wr * self.dT])
eps = 1e-6
⋮----
cos_sim = np.dot(delta_e, delta_e_pred) / (np.linalg.norm(delta_e) * np.linalg.norm(delta_e_pred))
⋮----
cos_sim = 0.0
mag_ratio = np.linalg.norm(delta_e) / (np.linalg.norm(delta_e_pred) + eps)
mag_penalty = abs(mag_ratio - 1.0)
r_consistency = (2.0 * cos_sim - 0.2 * mag_penalty)
⋮----
r_time = - 0.1
reward = r_goal + r_heading + r_cbf + r_smooth_w + r_smooth_v + r_consistency + r_time
⋮----
def reset(self, *, seed=None, options=None)
⋮----
obs = self.get_agent_observation(agent)
⋮----
def step(self, action_input)
⋮----
controllers = {}
⋮----
chunk = raw_chunk[:5]
action_step_0 = chunk
⋮----
obs_dict = {}
reward_dict = {}
done_dict = {}
info_dict = {}
truncated = bool(self.engine.steps >= self.max_steps)
⋮----
reward = self._compute_reward(agent, obs, controllers[agent.id], terminated_success, terminated_stuck)
⋮----
raw_chunk = action_input
⋮----
controllers = None
⋮----
agent = self.engine.agents[0]
⋮----
phys_action = self._action_post_process(action_step_0)
⋮----
controllers = {agent.id: phys_action}
⋮----
exec_action = controllers[agent.id] if controllers else np.zeros(5)
reward = self._compute_reward(agent, obs, exec_action, terminated_success, terminated_stuck)
info = {"reward_terms": getattr(self, 'last_reward_terms', {})}
⋮----
terminated = terminated_success or terminated_stuck
````

## File: sampler_and_process.py
````python
def sampler(num_samples=100)
⋮----
model = SAC.load("sac_policy")
⋮----
model = None
config_id = 0
count = 0
⋮----
# 1. 本阶段效能评估，固定初始config配置，单纯评估内部各类概率性模型
# 2. 初始化环境
sim = sim_initialize(config_id)
⋮----
config_id = 1
⋮----
# 3. 配置控制器 (无 GUI 模式)
config = {
controller = SimulationController(sim, config)
⋮----
save_path = f"sim_replay/{i}.pkl"
⋮----
def data_processer_for_TwoTower_Mapping()
⋮----
"""
    读取 sim_replay 目录下的 .pkl 仿真轨迹，提取双重映射模型所需的三元组数据集
    - P (装备性能): 提取 agent_config.yaml 中的 water/land 各 20 个参数
    - T (体系能力): 按红蓝阵营分开计算，仅计算到全灭或 MAX_SEQ_LEN
    - E (任务效能): 每局包含 2 个指标 (红方存活率, 蓝方存活率)
    """
⋮----
files_pattern = "*.pkl"
source_dir = "sim_replay"
dataset_path = "dataset/dual_mapping"
⋮----
P_dataset = []
T_dataset = []
E_dataset = []
MAX_SEQ_LEN = 300
files = glob.glob(os.path.join(source_dir, files_pattern))
⋮----
# ==========================================
# 预先读取 agent_config.yaml 获取红蓝双方装备参数 (20维)
⋮----
config_path = "agent/agent_config.yaml"
red_params = np.zeros(20, dtype=np.float32)
blue_params = np.zeros(20, dtype=np.float32)
param_keys = [
⋮----
cfg = yaml.safe_load(cfg_f)
water_cfg = cfg.get('water', {})
land_cfg = cfg.get('land', {})
red_params = np.array([water_cfg.get(k, 0.0) for k in param_keys], dtype=np.float32)
blue_params = np.array([land_cfg.get(k, 0.0) for k in param_keys], dtype=np.float32)
⋮----
buffer = pickle.load(f)
⋮----
initial_state = buffer[0][0]
# side=0 认为是红方，side=1 认为是蓝方
red_initial_count = sum(1 for a in initial_state if a.get('side', 0) == 0)
blue_initial_count = sum(1 for a in initial_state if a.get('side', 0) == 1)
⋮----
# 1. 寻找真正的任务结束点 (task_steps)
⋮----
actual_len = len(buffer)
task_steps = MAX_SEQ_LEN
⋮----
state = buffer[step_idx][0]
red_alive = sum(1 for a in state if a.get('side', 0) == 0 and not a.get('disabled', False))
blue_alive = sum(1 for a in state if a.get('side', 0) == 1 and not a.get('disabled', False))
# 如果某一方全军覆没，录入当前步数为结束点
⋮----
task_steps = step_idx + 1
⋮----
task_steps = min(task_steps, actual_len, MAX_SEQ_LEN)
⋮----
# 2. 计算【任务效能 E】
⋮----
# 截取任务结束帧，计算最终存活率
final_state = buffer[task_steps - 1][0]
red_alive_final = sum(1 for a in final_state if a.get('side', 0) == 0 and not a.get('disabled', False))
blue_alive_final = sum(1 for a in final_state if a.get('side', 0) == 1 and not a.get('disabled', False))
red_survival_rate = red_alive_final / red_initial_count if red_initial_count > 0 else 0.0
blue_survival_rate = blue_alive_final / blue_initial_count if blue_initial_count > 0 else 0.0
E_vec = np.array([red_survival_rate, blue_survival_rate, task_steps], dtype=np.float32)
⋮----
# 3. 按 task_steps 遍历计算【体系能力 T】
⋮----
T_red_seq = []
T_blue_seq = []
⋮----
transition = buffer[step_idx]
state = transition[0]
actions = transition[2] # dict: {agent_id: action_array}
red_active = [a for a in state if a.get('side', 0) == 0 and not a.get('disabled', False)]
blue_active = [a for a in state if a.get('side', 0) == 1 and not a.get('disabled', False)]
# 计算红方指标 (距离 & 机动平滑度)
red_dist = 0.0
⋮----
dists = [np.linalg.norm(np.array(red_active[i]['position']) - np.array(red_active[j]['position']))
red_dist = np.mean(dists)
red_mag = 0.0
red_ids = [a['id'] for a in red_active]
red_actions = [actions[aid][:2] for aid in red_ids if aid in actions and actions[aid] is not None]
⋮----
red_mag = np.mean([np.linalg.norm(act) for act in red_actions])
⋮----
# 计算蓝方指标 (距离 & 机动平滑度)
blue_dist = 0.0
⋮----
dists = [np.linalg.norm(np.array(blue_active[i]['position']) - np.array(blue_active[j]['position']))
blue_dist = np.mean(dists)
blue_mag = 0.0
blue_ids = [a['id'] for a in blue_active]
blue_actions = [actions[aid][:2] for aid in blue_ids if aid in actions and actions[aid] is not None]
⋮----
blue_mag = np.mean([np.linalg.norm(act) for act in blue_actions])
⋮----
# 转换为 Numpy 数组并 Padding 对齐到 MAX_SEQ_LEN
T_red_seq = np.array(T_red_seq, dtype=np.float32)
T_blue_seq = np.array(T_blue_seq, dtype=np.float32)
⋮----
pad_len = MAX_SEQ_LEN - len(T_red_seq)
T_red_seq = np.pad(T_red_seq, ((0, pad_len), (0, 0)), 'constant')
T_blue_seq = np.pad(T_blue_seq, ((0, pad_len), (0, 0)), 'constant')
⋮----
# 4. 组装数据 (分别将红蓝方压入)
⋮----
# 最终保存
⋮----
P_dataset = np.stack(P_dataset)
T_dataset = np.stack(T_dataset)
E_dataset = np.stack(E_dataset)
````

## File: sim_env/env_model.py
````python
class TargetItem
⋮----
def __init__(self, init_pos, vmax=60.0, accel=100.0, dT=0.2)
def update(self)
REL_COOP   = np.array([1,0,0], dtype=int)
REL_CHASE  = np.array([0,1,0], dtype=int)
REL_ATTK   = np.array([0,0,1], dtype=int)
REL_NONE   = np.array([0,0,0], dtype=int)
class env_model
⋮----
agent_ids = [agent.id for agent in agents]
neighbors_dict = {}
targets_dict = {}
cannon_targets_dict = {}
⋮----
neighbors = []
targets = []
cannon_targets = []
⋮----
def init_msgpool(self, msg_pool: MsgPool)
def smoke_area(self,agents_data)
⋮----
def smoke_update(smoke)
⋮----
alpha = 4 * self.smoke_radius / self.smoke_last_time ** 2
radius = self.smoke_radius - alpha * ((smoke[1]-1) - self.smoke_last_time / 2) ** 2
⋮----
def attack_results(self,agents_data)
⋮----
live_ids = []
⋮----
attk_error = agent.get('position') - items[0]
⋮----
def obs_sector_sampling(self,agents_data)
⋮----
obs_sector_dict = {}
grid_size = self.grid_size
⋮----
panel_center = np.array([W * grid_size * 0.5, H * grid_size * 0.5])
⋮----
xi = 0.5 * np.pi / self.sector_num
sector_edges = np.linspace(0, 2 * np.pi, self.sector_num + 1)
⋮----
R = 100
pos = agent.get('position')
yaw = agent.get('angle')
agent_id = agent.get('id')
x_idx_global = int((pos[0] - origin_x) / grid_size)
y_idx_global = int((pos[1] - origin_y) / grid_size)
local_half = int(np.ceil(R / grid_size))
x_start = max(0, x_idx_global - local_half)
x_end   = min(W, x_idx_global + local_half + 1)
y_start = max(0, y_idx_global - local_half)
y_end   = min(H, y_idx_global + local_half + 1)
local_grid = self.grid_map[y_start:y_end, x_start:x_end]
⋮----
obs_sector = np.full(self.sector_num, R)
⋮----
obs_world_x = (x_start + obs_xs) * grid_size + origin_x
obs_world_y = (y_start + obs_ys) * grid_size + origin_y
dx = obs_world_x - pos[0]
dy = obs_world_y - pos[1]
distances = np.hypot(dx, dy)
mask = distances <= R
⋮----
distances = distances[mask]
abs_angles = np.arctan2(dy, dx)
rel_angles = (abs_angles - yaw) % (2 * np.pi)
⋮----
left_bound = sector_edges[i] - xi
right_bound = sector_edges[i + 1] + xi
in_sec = (
⋮----
def env_update(self,agents_data)
````

## File: sim_env/map_data/start.py
````python
def main()
⋮----
data_id = 0
⋮----
sim = sim_initialize(data_id)
model = SAC.load("sac_policy")
app = QApplication(sys.argv)
max_steps = 2000
config = {
window = ControlledVisWindow(sim, config)
````

## File: sim_env/map_generator.py
````python
class MapGenerator
⋮----
def __init__(self, width, height, isBlank = False, scale=30, threshold=0.5, seed=None)
def generate_map(self, d_sample_hw)
⋮----
def fade(t)
def lerp(t, a, b)
def grad(hash, x, y)
⋮----
h = hash & 15
u = np.where(h < 8, x, y)
v = np.where(h < 4, y, x)
⋮----
def perlin_2d(x, y, p)
⋮----
xi = np.floor(x).astype(int) & 255
yi = np.floor(y).astype(int) & 255
xf = x - np.floor(x)
yf = y - np.floor(y)
u = fade(xf)
v = fade(yf)
aa = p[p[xi] + yi]
ab = p[p[xi] + yi + 1]
ba = p[p[xi + 1] + yi]
bb = p[p[xi + 1] + yi + 1]
x1 = lerp(u, grad(aa, xf, yf), grad(ba, xf - 1, yf))
x2 = lerp(u, grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1))
⋮----
def down_sampling(grid_map : np.ndarray, target_hw : np.ndarray)
⋮----
original_hw = grid_map.shape
⋮----
block_h = H // H_target
block_w = W // W_target
crop_H = block_h * H_target
crop_W = block_w * W_target
cropped = grid_map[:crop_H, :crop_W]
downsampled = np.zeros((H_target, W_target), dtype=grid_map.dtype)
⋮----
block = cropped[h_start:h_end, w_start:w_end]
⋮----
grid_map = np.zeros((self.width, self.height), dtype=np.uint8)
down_sampled_map = down_sampling(grid_map, self.d_sample_hw)
⋮----
p = np.arange(256, dtype=int)
⋮----
p = np.tile(p, 2)
⋮----
x = x / self.scale
y = y / self.scale
noise = perlin_2d(x, y, p)
grid_map = (noise > self.threshold).astype(np.uint8)
⋮----
def load_map(self, obs_map, down_sampled_map)
def gridmap2axis(self, )
def axis2gridmap(self, )
````

## File: sim/config_data/config0 copy 2.yaml
````yaml
map:
  width: 256
  height: 256
  grid_size: 3
  d_sample_hw:
  - 32
  - 32
  save_path:
  - E:/code/v3/version3.4/sim_env/map_data/grid_map0.npy
  - E:/code/v3/version3.4/sim_env/map_data/d_spl_map0.npy
agents:
  num: 6
  dT:
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  id:
  - 22
  - 15
  - 8
  - 16
  - 4
  - 3
  theta:
  - - 0.10024331752073401
    - 0.7762626022640395
  - - 0.22678783582885575
    - -0.6593166864653162
  - - 0.009477942707859288
    - -0.04500190052991648
  - - -0.5656857859430184
    - 0.9830349728322796
  - - 0.5775985420871252
    - 0.5441784858096328
  - - 0.3216420546826988
    - -0.6641975472244921
  pos:
  - - -48.0
    - 216.0
  - - -24.0
    - 240.0
  - - -48.0
    - 192.0
  - - 72.0
    - -24.0
  - - -24.0
    - -24.0
  - - 264.0
    - -144.0
  side:
  - 0
  - 0
  - 0
  - 1
  - 1
  - 1
  com_tensor:
  - - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
  profiles:
  - water
  - water
  - water
  - land
  - land
  - land
target_distance:
  '22':
  - -1.0
  - 1.2246467991473532e-16
  '15':
  - 0.49999999999999933
  - -0.866025403784439
  '8':
  - 0.5000000000000003
  - 0.8660254037844385
  '16':
  - 1.0
  - 0.0
  '4':
  - -0.4999999999999998
  - 0.8660254037844387
  '3':
  - -0.5000000000000004
  - -0.8660254037844385
formation_structure:
  22-15:
  - 10.0
  - 10.0
  22-8:
  - 0.0
  - -10.0
  15-22:
  - -10.0
  - -10.0
  15-8:
  - -10.0
  - -10.0
  8-22:
  - 0.0
  - 10.0
  8-15:
  - 10.0
  - 10.0
  16-4:
  - -10.0
  - 0.0
  16-3:
  - 10.0
  - -10.0
  4-16:
  - 10.0
  - 0.0
  4-3:
  - 10.0
  - -10.0
  3-16:
  - -10.0
  - 10.0
  3-4:
  - -10.0
  - 10.0
````

## File: sim/config_data/config0.yaml
````yaml
map:
  width: 256
  height: 256
  grid_size: 3
  d_sample_hw:
  - 32
  - 32
  save_path:
  - E:/code/v3/version3.4/sim_env/map_data/grid_map0.npy
  - E:/code/v3/version3.4/sim_env/map_data/d_spl_map0.npy
agents:
  num: 6
  dT:
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  id:
  - 19
  - 20
  - 10
  - 3
  - 9
  - 22
  theta:
  - - 0.10038844340629138
    - 0.23738310503742888
  - - 0.7030298270414115
    - 0.1404624240302319
  - - 0.7108380523811031
    - 0.5670764571368414
  - - 0.1534248669401399
    - 0.002183025457272336
  - - 0.16815960240105543
    - 0.8362777898513254
  - - -0.6037114300891873
    - 0.000813095408589648
  pos:
  - - 24.0
    - 192.0
  - - 96.0
    - 288.0
  - - -72.0
    - 0.0
  - - 120.0
    - -264.0
  - - 144.0
    - -264.0
  - - -144.0
    - -288.0
  side:
  - 0
  - 0
  - 0
  - 1
  - 1
  - 1
  com_tensor:
  - - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
  profiles:
  - water
  - water
  - water
  - land
  - land
  - land
target_distance:
  '19':
  - -1.0
  - 1.2246467991473532e-16
  '20':
  - 0.49999999999999933
  - -0.866025403784439
  '10':
  - 0.5000000000000003
  - 0.8660254037844385
  '3':
  - 1.0
  - 0.0
  '9':
  - -0.4999999999999998
  - 0.8660254037844387
  '22':
  - -0.5000000000000004
  - -0.8660254037844385
formation_structure:
  19-20:
  - 10.0
  - 10.0
  19-10:
  - -10.0
  - -10.0
  20-19:
  - -10.0
  - -10.0
  20-10:
  - -10.0
  - -10.0
  10-19:
  - 10.0
  - 10.0
  10-20:
  - 10.0
  - 10.0
  3-9:
  - 10.0
  - 0.0
  3-22:
  - -10.0
  - -10.0
  9-3:
  - -10.0
  - 0.0
  9-22:
  - -10.0
  - -10.0
  22-3:
  - 10.0
  - 10.0
  22-9:
  - 10.0
  - 10.0
````

## File: sim/config_data/config1.yaml
````yaml
map:
  width: 256
  height: 256
  grid_size: 3
  d_sample_hw:
  - 32
  - 32
  save_path:
  - E:/code/v3/version3.4/sim_env/map_data/grid_map1.npy
  - E:/code/v3/version3.4/sim_env/map_data/d_spl_map1.npy
agents:
  num: 6
  dT:
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  - 0.02
  id:
  - 10
  - 9
  - 16
  - 25
  - 2
  - 7
  theta:
  - - 0.6354691514432829
    - -0.5148956826807443
  - - -0.43971043397781173
    - 0.8714155133296793
  - - -0.3143858123716061
    - 0.7658517366534434
  - - 0.20588625296752383
    - 0.46005568120199847
  - - -0.8032249236836286
    - -0.071366982374087
  - - -0.8535322048835738
    - 0.7507786831087064
  pos:
  - - 48.0
    - 264.0
  - - -144.0
    - 288.0
  - - -48.0
    - 96.0
  - - 24.0
    - -144.0
  - - 192.0
    - -48.0
  - - 168.0
    - -264.0
  side:
  - 0
  - 0
  - 0
  - 1
  - 1
  - 1
  com_tensor:
  - - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
    - - 1
      - 0
      - 0
  - - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 0
      - 0
      - 1
    - - 1
      - 0
      - 0
    - - 1
      - 0
      - 0
    - - 0
      - 0
      - 0
  profiles:
  - water
  - water
  - water
  - land
  - land
  - land
target_distance:
  '10':
  - -1.0
  - 1.2246467991473532e-16
  '9':
  - 0.49999999999999933
  - -0.866025403784439
  '16':
  - 0.5000000000000003
  - 0.8660254037844385
  '25':
  - 1.0
  - 0.0
  '2':
  - -0.4999999999999998
  - 0.8660254037844387
  '7':
  - -0.5000000000000004
  - -0.8660254037844385
formation_structure:
  10-9:
  - -10.0
  - 10.0
  10-16:
  - -10.0
  - -10.0
  9-10:
  - 10.0
  - -10.0
  9-16:
  - 10.0
  - -10.0
  16-10:
  - 10.0
  - 10.0
  16-9:
  - -10.0
  - 10.0
  25-2:
  - 10.0
  - 10.0
  25-7:
  - 10.0
  - -10.0
  2-25:
  - -10.0
  - -10.0
  2-7:
  - -10.0
  - -10.0
  7-25:
  - -10.0
  - 10.0
  7-2:
  - 10.0
  - 10.0
````

## File: sim/physics_engine.py
````python
class PhysicsEngine(env_model)
⋮----
def __init__(self, map_layers=None, grid_size=5, dT=0.1)
def init_msgpool(self, msg_pool: MsgPool)
⋮----
agent_ids = [agent.id for agent in agents]
⋮----
neighbors_dict = {}
targets_dict = {}
cannon_targets_dict = {}
⋮----
REL_COOP = np.array([1, 0, 0])
REL_CHASE = np.array([0, 1, 0])
REL_ATTK = np.array([0, 0, 1])
⋮----
neighbors = []
targets = []
cannon_targets = []
⋮----
sid = str(agent_ids[i])
⋮----
def step_physics(self, controllers=None)
⋮----
agents_data = self._get_agent_data_struct()
⋮----
action = controllers[agent.id]
case = "RL_Actor"
⋮----
def _get_agent_data_struct(self)
⋮----
agent_data = []
⋮----
def get_render_data(self)
⋮----
agent_data = self._get_agent_data_struct()
env_data = {'SmokeArea': self.smoke}
⋮----
def reset_engine(self)
⋮----
attrs = self.initial_state.get('env_model_attrs', {})
````

## File: sim/sim_initialize.py
````python
def sim_initialize(i=None)
⋮----
version_config = yaml.safe_load(f)
version = version_config["version"]["id"]
config_path = f"E:/code/v3/version{version}/sim/config_data/config{i}.yaml" if isinstance(i, int) else i
⋮----
config = yaml.safe_load(f)
width = config["map"]["width"]
height = config["map"]["height"]
grid_size = config["map"]["grid_size"]
save_path = config["map"]["save_path"]
map_layers = MapGenerator(width, height)
⋮----
engine = PhysicsEngine(map_layers=map_layers, grid_size=grid_size, dT=0.03)
msg_pool = MsgPool()
⋮----
agent_num = config["agents"]["num"]
agent_id = config["agents"]["id"]
pos = np.array(config["agents"]["pos"])
agent_dT = config["agents"]["dT"]
agent_side = np.array(config["agents"]["side"])
theta = np.array(config["agents"]["theta"])
agents = []
⋮----
com_tensor = np.array(config["agents"]["com_tensor"])
init_channel = msg_pool.channel_id[0:agent_num]
target_distance = {k: np.array(v) for k, v in config["target_distance"].items()}
formation_structure = {
⋮----
env = RLEnvAdapter(engine, agent_id)
⋮----
def main()
⋮----
env = sim_initialize()
````

## File: start.py
````python
def main()
⋮----
data_id = 0
sim = sim_initialize(data_id)
model = SAC.load("sac_policy")
app = QApplication(sys.argv)
max_steps = 2000
config = {
window = ControlledVisWindow(sim, config)
````

## File: train_SAC.py
````python
def make_env(i)
⋮----
def _init()
⋮----
sim = sim_initialize(i)
⋮----
def train_agent()
⋮----
iteration = 1
⋮----
num_envs = 12
⋮----
env = SubprocVecEnv([make_env(0) for _ in range(num_envs)])#[make_env(i) for i in range(num_envs)])
env = VecMonitor(env, filename=f"./logs/monitor{iter}.csv")
case = 1 if iter == 0 else 1
model_path = "sac_policy"
⋮----
model = SAC(
reset_timesteps = True
⋮----
model = SAC.load(model_path, env=env)
⋮----
reset_timesteps = False
⋮----
total_timesteps = 300000,   # 至少 total_timesteps 步 ，200000一轮课程比较稳定
⋮----
log_interval = 2000,          # 每 log_interval 步打印一次
⋮----
def test_and_vis()
⋮----
# # 清理子文件夹中的匹配文件
# targets = ["config*", "grid_map*", "d_spl_map*"]
# dirs = ["sim/config_data","sim_env/map_data"]
⋮----
i = 1
⋮----
env = sim_initialize(i)
model = SAC.load("sac_policy")
app = QApplication(sys.argv)
max_steps = 2000
data_id = 0
config = {
window = ControlledVisWindow(env, config)
````

## File: Two_Tower_Evaluation.py
````python
CONFIG = {
device = torch.device(CONFIG['device'])
⋮----
# ================= 1. 网络结构定义 =================
# 辅助模块：位置编码 (复用你预测器中的设计)
class PositionalEncoding(nn.Module)
⋮----
def __init__(self, d_model, max_len=200)
⋮----
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
⋮----
def forward(self, x)
# 【左塔】装备性能编码器 (Static Parameter Encoder)
class LeftTower(nn.Module)
⋮----
def __init__(self, input_dim=20, embed_dim=128)
⋮----
# 简单的 MLP 提取静态参数特征
⋮----
def forward(self, p_seq)
⋮----
# p_seq: (Batch, 20)
⋮----
# 【右塔】体系能力时序编码器 (Trajectory/Capability Encoder)
class RightTower(nn.Module)
⋮----
def __init__(self, input_dim=2, hidden_dim=64, embed_dim=128)
⋮----
encoder_layer = nn.TransformerEncoderLayer(
⋮----
def forward(self, t_seq)
⋮----
# t_seq: (Batch, Seq_len=200, 2)
x = self.input_proj(t_seq)          # (B, 200, 64)
x = self.pos_encoder(x)             # 注入时间位置信息
x = self.transformer(x)             # (B, 200, 64)
# 时序池化 (取序列的平均特征)
x_pooled = x.mean(dim=1)            # (B, 64)
# 映射到最终的对齐空间
return self.output_proj(x_pooled)   # (B, 128)
# ================= 2. 训练主流程 =================
def train_model_A()
⋮----
# 1. 加载数据
⋮----
P_data = np.load(CONFIG['dataset_P'])
T_data = np.load(CONFIG['dataset_T'])
P_tensor = torch.FloatTensor(P_data).to(device)
T_tensor = torch.FloatTensor(T_data).to(device)
dataset = TensorDataset(P_tensor, T_tensor)
dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
left_tower = LeftTower(input_dim=20, embed_dim=CONFIG['embed_dim']).to(device)
right_tower = RightTower(input_dim=2, embed_dim=CONFIG['embed_dim']).to(device)
optimizer = optim.Adam(
criterion = nn.MSELoss()
⋮----
epoch_loss = 0.0
⋮----
feat_left = left_tower(batch_P)
feat_right = right_tower(batch_T)
loss = criterion(feat_left, feat_right)
⋮----
# 4. 重点：只保存左塔 (Left Tower)
# 因为在推理阶段，右塔(仿真过程)是不存在的，我们只需要左塔！
````

## File: version.yaml
````yaml
version:
  id: 3.4
````

## File: vis/agentvis.py
````python
class AgentItem
⋮----
def __init__(self, agent_data, panel_center, color=None)
def draw(self, painter: QtGui.QPainter)
⋮----
pos = self.agent_data.get('position')
⋮----
x = int(pos[0] + self.panel_center[0])
y = int(pos[1] + self.panel_center[1])
livestate = self.agent_data.get('disabled')
⋮----
r_point = self.agent_data.get('rpoint')
⋮----
dist = np.linalg.norm(r_point[0:1])
⋮----
rx = int(pos[0] + r_point[0] * np.cos(self.agent_data.get('angle')) - r_point[1] * np.sin(self.agent_data.get('angle')) + self.panel_center[0])
ry = int(pos[1] + r_point[0] * np.sin(self.agent_data.get('angle')) + r_point[1] * np.cos(self.agent_data.get('angle')) + self.panel_center[1])
pen_line = QtGui.QPen(QtGui.QColor(150, 150, 150), 1, Qt.DashLine)
⋮----
pen_marker = QtGui.QPen(QtGui.QColor(self.color), 2)
⋮----
ms = 4
⋮----
p_pos = self.agent_data.get('p_pos')
⋮----
px = int(p_pos[0] + self.panel_center[0])
py = int(p_pos[1] + self.panel_center[1])
pen_line = QtGui.QPen(QtGui.QColor(100, 100, 100), 1, Qt.DashLine)
⋮----
pen_formation = QtGui.QPen(QtGui.QColor(0, 100, 255, 180), 1, Qt.DashLine)
⋮----
ms = 6
⋮----
WP_theta = self.agent_data.get('WPangle')
⋮----
theta = int(self.agent_data.get('angle')* 180/np.pi)
new_radius = 30
theta_span = int(self.agent_data.get('sense_angle') * 180 / np.pi)
start_angle = (- theta - theta_span ) * 16
span_angle  = (theta_span * 2) * 16
rect = QtCore.QRectF(x - new_radius, y - new_radius, 2*new_radius, 2*new_radius)
color_with_alpha = QtGui.QColor(0,255,0, alpha= 64)
⋮----
attk_pos = self.agent_data.get('ATTKpos')
⋮----
attk_x = int(attk_pos[0] + self.panel_center[0])
attk_y = int(attk_pos[1] + self.panel_center[1])
⋮----
def draw_id(self, painter: QtGui.QPainter, x: int, y: int)
⋮----
font = QtGui.QFont("Arial", 12)
⋮----
display_id = self.agent_data['id']
````

## File: vis/base_vis.py
````python
class VisualizationWindow(QtWidgets.QWidget)
⋮----
def __init__(self, simulation)
⋮----
canvas_w = width * self.grid_size
canvas_h = height * self.grid_size
⋮----
def update_background(self)
⋮----
painter = QtGui.QPainter(self.background_pixmap)
⋮----
land_path = QtGui.QPainterPath()
⋮----
base_y = canvas_h * 0.4
⋮----
curve_y_coords = {}
⋮----
y = base_y + 20 * math.sin(x / 40.0) + 15 * math.cos(x / 90.0)
⋮----
foam_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 150), 3)
⋮----
obstacle_coords = np.argwhere(self.grid_map == 1)
⋮----
padding = 2
rect_size = self.grid_size - padding * 2
⋮----
rect_x = x * self.grid_size + padding
rect_y = y * self.grid_size + padding
⋮----
def update_simulation(self)
⋮----
dummy_action = np.zeros(5, dtype=np.float32)
⋮----
def paintEvent(self, event)
⋮----
painter = QtGui.QPainter(self)
⋮----
render_data = self.engine.get_render_data()
⋮----
side = agent_data.get('side', 0)
color = self.group_color[side] if side < len(self.group_color) else "gray"
item = AgentItem(agent_data, self.panel_center, color=color)
⋮----
env_data = render_data['env']
smoke_areas = env_data.get('SmokeArea', 0)
⋮----
color = self.smoke_color
item = SmokeItem(area, self.panel_center, color=color)
````

## File: vis/controlled_window.py
````python
class ControlledVisWindow(QMainWindow)
⋮----
def __init__(self, simulation, config=None)
def on_timer_tick(self)
⋮----
continue_flag = self.controller.step()
⋮----
info = self.controller.get_info()
````

## File: vis/info_panel.py
````python
class InfoPanelWidget(QWidget)
⋮----
def __init__(self)
def _on_target_changed(self, text)
def update_info(self, info_dict)
⋮----
current_step = info_dict['step']
⋮----
max_x = max(20, current_step)
min_x = max_x - 20
⋮----
current_plot_target = self.plot_selector.currentText()
current_ids = set()
⋮----
a_id = agent['id']
⋮----
container = QWidget()
h_layout = QHBoxLayout(container)
⋮----
lbl = QLabel()
⋮----
bar = QProgressBar()
⋮----
ui = self.agent_ui_map[a_id]
⋮----
# --- 处理画图逻辑 ---
⋮----
color = pg.intColor(a_id, hues=9)
curve = self.plot_widget.plot(pen=pg.mkPen(color, width=2))
text_item = pg.TextItem(f"ID:{a_id}", color=color)
⋮----
'w': [],  # 新增 w 的缓存
⋮----
p_data = self.plot_data_map[a_id]
⋮----
# 后台同时记录 v 和 w
⋮----
# 根据下拉框的选择，决定提取哪组数据去画图
y_data = p_data['v'] if current_plot_target == "agent.v" else p_data['w']
⋮----
existing_ids = list(self.agent_ui_map.keys())
⋮----
ui = self.agent_ui_map.pop(old_id)
⋮----
p_data = self.plot_data_map.pop(old_id)
````

## File: vis/replay_buffer.py
````python
class ReplayBuffer
⋮----
def __init__(self, capacity)
def push(self, state, obs, action, reward, next_state, next_obs, done)
def reset(self)
def sample(self, batch_size)
def __len__(self)
def save_buffer(self, filepath: str = "sim_replay/0.pkl")
⋮----
buffer_list = list(self.buffer)
⋮----
def read_buffer(self, filepath: str = "sim_replay/0.pkl")
⋮----
buffer_list = pickle.load(f)
⋮----
def extract_action_dataset(self, filepath: str = "sim_replay/0.pkl")
⋮----
slice_len = 10
all_action_samples = []
agent_sequences = {}
⋮----
ACTION_THRESHOLD = 0.1
STATE_DIFF_MIN = 0.001
⋮----
# 解包 transition
# 注意：根据你的描述，action 是一个字典 {agent_id: array([...])}
⋮----
# 处理 done：如果这一帧 done 了，由于这一帧的 action 导致了终止或重启，
# 为了动作连续性，我们通常直接清空所有 agent 的当前序列缓存
⋮----
# --- 物理一致性过滤 (Consistency Check) ---
# 提取该 agent 的状态变化, 当前版本仅可用于单体采样
s_curr = np.array([
s_next = np.array([
state_diff = np.linalg.norm(s_next - s_curr)
action_mag = np.linalg.norm(action_vec)
# 逻辑：如果 action 很大 (>阈值) 但 state 几乎没动 (<最小值)
# 这通常意味着撞墙、打滑或传感器数据丢失
⋮----
agent_sequences[agent_id] = [] # 丢弃该 agent 之前的累积，因为它遇到了异常
⋮----
# 1. 如果该 agent 还没在临时字典里，初始化它
⋮----
# 2. 将当前动作存入该 agent 的序列
⋮----
# 3. 如果该 agent 的序列达到了指定长度 slice_len
⋮----
# 转换成 numpy 数组并存入最终样本池
sample = np.array(agent_sequences[agent_id], dtype=np.float32)
⋮----
# 4. 清空该 agent 的缓存，开始记录下一个片段
# 滑动窗口
⋮----
# # 采用非重叠切片
# agent_sequences[agent_id] = []
# 转换成 VQ-VAE 需要的最终形状 (N, T, D)
final_data = np.stack(all_action_samples, axis=0)
⋮----
def extract_dynamics_dataset(self, vq_model, Horizen_len: int, filepath: str = "sim_replay/0.pkl")
⋮----
device = next(vq_model.parameters()).device
POS_LIMIT = 500.0
ANG_LIMIT = np.pi
SKILL_NUM = 16
def normalize_state(lidar, rel_goal)
⋮----
norm_lidar = np.array(lidar)
norm_phy = np.zeros(3, dtype=np.float32)
⋮----
dataset_curr_obs = []
dataset_skills = []
dataset_actions = []
dataset_future_obs = []
agent_caches = {}
⋮----
# 记录当前帧原始信息
⋮----
# 积累够 Horizen_len + 1 帧 (需未来帧作为标签)
⋮----
# --- A. 获取 Skill ID ---
# 取前 10 步动作
act_seq = np.array(agent_caches[agent_id]['actions'][:Horizen_len], dtype=np.float32)
act_tensor = torch.from_numpy(act_seq).unsqueeze(0).to(device)
⋮----
z = vq_model.enc(act_tensor.view(1, -1))
⋮----
skill_id = indices.item()
skill_z_np = z.cpu().numpy().flatten().tolist()
⋮----
# --- B. 准备输入 (第 0 帧) ---
⋮----
state_input = np.concatenate([curr_l, curr_p])
# --- C. 准备目标标签 (未来 1 到 Horizen_len 帧) ---
future_sequence = []
⋮----
# 获取未来帧的归一化状态
⋮----
# 组合成 39 维状态
f_state = np.concatenate([f_l, f_p])
# 存储相对偏移 (未来状态 - 当前状态)
# 此时所有数值都在 [-1, 1] 附近的量纲下
⋮----
# --- D. 存入数据集 ---
⋮----
# --- E. 滑动窗口 ---
⋮----
# 转换并返回
final_obs = np.array(dataset_curr_obs, dtype=np.float32)
final_skills = np.array(dataset_skills, dtype=np.float32)
final_actions = np.array(dataset_actions, dtype=np.float32)
final_future_obs = np.array(dataset_future_obs, dtype=np.float32)
⋮----
def visualize_filter_thresholds(filepath: str, action_threshold=0.5, state_diff_min=0.01)
⋮----
# 该函数用于专门测试extract_action_processer中数据清洗的效果，红点表示被剔除的数据
⋮----
action_mags = []
state_diffs = []
is_rejected = []
⋮----
# 解包 (假设 state 和 next_state 也是 dict)
⋮----
# if done: continue
⋮----
# 计算动作模长
a_vec = action_dict[agent_id]
a_mag = np.linalg.norm(a_vec)
# 计算位移模长
⋮----
s_diff = np.linalg.norm(s_next - s_curr)
⋮----
# 判定是否会被过滤 (逻辑同步 extract_action_processer)
rejected = (a_mag > action_threshold and s_diff < state_diff_min)
⋮----
# 转换为 numpy 以便绘图
action_mags = np.array(action_mags)
state_diffs = np.array(state_diffs)
is_rejected = np.array(is_rejected)
# --- 绘图 ---
⋮----
# 1. 散点图：展示动作与位移的关系
⋮----
# 绘制阈值线
⋮----
# 填充拒绝区域（左上角：高动作，低位移）
⋮----
# 使用示例
# visualize_filter_thresholds(f"sim_replay/{np.random.randint(0,1000)}.pkl", action_threshold=0.1, state_diff_min=0.001)
````

## File: vis/sim_controller.py
````python
class SimulationController
⋮----
def __init__(self, simulation, config=None)
⋮----
lower_model = self.config.get("lower_actor")
⋮----
ENV_NAME = "swarm_fast_single_agent"
config = (
⋮----
lower_model = os.path.abspath(lower_model)
⋮----
higher_model = self.config.get("higher_actor")
⋮----
ENV_NAME = "swarm_multi_agent"
⋮----
higher_model = os.path.abspath(higher_model)
⋮----
def _get_buffer_path(self)
def should_continue(self)
def step(self)
⋮----
transition = self.replay_buffer.buffer[self.step_count]
⋮----
active_agents = [a for a in self.engine.agents if not a.disabled]
obs_dict = {a.id: self.env.get_agent_observation(a) for a in active_agents}
⋮----
obs_dict = self.obs
action_dict = {}
⋮----
action = self.lower_model.compute_single_action(observation=obs, explore=False)
⋮----
action = action[:5]
⋮----
current_phys_state = self.engine._get_agent_data_struct()
all_done = all(dones.values()) if dones else False
buffer_action = action_dict if action_dict else np.zeros(5, dtype=np.float32)
⋮----
def get_info(self)
⋮----
render_data = self.engine.get_render_data()
reward_str = "N/A"
agents_info = []
⋮----
pos = agent['position']
````

## File: vis/smokevis.py
````python
class SmokeItem
⋮----
def __init__(self, area, panel_center, color=None)
def draw(self, painter: QtGui.QPainter)
⋮----
x = int(self.smoke_pos[0] + self.panel_center[0])
y = int(self.smoke_pos[1] + self.panel_center[1])
color_with_alpha = QtGui.QColor(200,200,200, alpha= 64)
````
