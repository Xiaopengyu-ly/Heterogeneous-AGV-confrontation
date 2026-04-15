import numpy as np
import copy

from comm.msg_pool import MsgPool
class TargetItem:
    """非合作目标的运动逻辑，可改造成移动障碍物或护送目标（无能力）"""
    def __init__(self, init_pos, vmax=60.0, accel=100.0, dT=0.2):
        self.position = np.array(init_pos, dtype=np.float64)
        self.velocity = np.array([5.0, -5.0], dtype=np.float64)
        self.v_max = float(vmax)
        self.accel = float(accel)
        self.dT = dT
    def update(self):
        """仅自动积分更新位置（无键盘输入）"""
        # 若需要更复杂行为（如随机机动），可在此扩展
        self.position = self.position + self.velocity * self.dT

# 定义关系类型 one-hot: [合作, 追, 逃]
REL_COOP   = np.array([1,0,0], dtype=int)
REL_CHASE  = np.array([0,1,0], dtype=int)
REL_ATTK   = np.array([0,0,1], dtype=int)
REL_NONE   = np.array([0,0,0], dtype=int)

'''  编写仿真环境模型  '''
class env_model:
    def init_agents(self, agents: list, com_tensor: np.ndarray, init_channel: list,
                    target_distance: dict, formation_structure: dict, goal_config: dict = None):
        """
        初始化智能体及其通信/任务关系。
        参数与原始代码完全一致。
        """
        self.agents = agents
        agent_ids = [agent.id for agent in agents]
        
        neighbors_dict = {}
        targets_dict = {}
        cannon_targets_dict = {}
        self.channel_dict = {}
        '''
            为形成对抗，在此处理，区分阵营，并解码追逃关系
                形成追逃关系字典，在目标信息给定时按字典进行分配
                把self.target写进agent里，并在agent里设置目标位置获取函数
        '''
        # 解析通信关系矩阵
        for i, row in enumerate(com_tensor):
            neighbors = []
            targets = []
            cannon_targets = []
            for j, rel_vec in enumerate(row):
                if i == j:
                    continue
                if np.array_equal(rel_vec, REL_COOP):
                    neighbors.append((agent_ids[j], 1))
                elif np.array_equal(rel_vec, REL_CHASE):
                    targets.append((agent_ids[j], 1))
                elif np.array_equal(rel_vec, REL_ATTK):
                    cannon_targets.append((agent_ids[j], 1))
            neighbors_dict[str(agent_ids[i])] = neighbors
            targets_dict[str(agent_ids[i])] = targets
            cannon_targets_dict[str(agent_ids[i])] = cannon_targets
            # print(agent_ids[i], ' ',targets_dict,' ',cannon_targets_dict)

        for i in range(len(init_channel)):
            self.channel_dict[str(agent_ids[i])] = init_channel[i]

        # 初始化每个 Agent
        # target_info = np.array([self.target.velocity, self.target.position])
        for agent in self.agents:
            agent.get_connect(self.msg_pool)
            agent.get_init_parameters(
                self.channel_dict[str(agent.id)],
                target_distance[str(agent.id)]
            )
            agent.get_net_neighbors(neighbors_dict, self.channel_dict, formation_structure, targets_dict, cannon_targets_dict)
            agent.get_grid_map(self.map_layers, self.grid_size)
        self.sim_copy = copy.deepcopy(self.__dict__)
        self.env_feedback['channel_dict'] = self.channel_dict

    def init_msgpool(self, msg_pool: MsgPool):
        self.msg_pool = msg_pool
        self.msg_pool.check()

    def smoke_area(self,agents_data):
        # 计算烟雾区域演化
        def smoke_update(smoke):
            alpha = 4 * self.smoke_radius / self.smoke_last_time ** 2
            radius = self.smoke_radius - alpha * ((smoke[1]-1) - self.smoke_last_time / 2) ** 2
            return (smoke[0], smoke[1] - 1, radius if radius >= 0 else 0)
        self.smoke = [smoke_update(x) for x in self.smoke if x[1] > 0]
        for agent in agents_data:
            if agent.get('SMOKE') and agent.get('disabled') == False:
                self.smoke.append((agent.get('position'), self.smoke_last_time, 0))
        self.map_layers.smoke = self.smoke
        self.env_feedback['smoke_zone'] = self.smoke

    def attack_results(self,agents_data):
        self.cannon_attk_results = []
        live_ids = []
        # 采集agent攻击方式（如果发射炮弹，以什么杀伤半径打击哪个位置）
        for agent in agents_data:
            if agent.get('disabled') is not True:
                if agent.get('ATTKlaunched') is True:
                    self.cannon_attk_results.append([agent.get('ATTKpos'),agent.get('ATTKradius')])
                live_ids.append(agent.get('id'))
        for items in self.cannon_attk_results:
            for agent in agents_data:
                if agent.get('disabled') is not True:
                    if items[0] is not None:
                        attk_error = agent.get('position') - items[0] 
                        # 如果是穿甲弹，杀伤半径内打中即死
                        if np.linalg.norm(attk_error) < items[1]:
                            if agent.get('id') in live_ids:
                                live_ids.remove(agent.get('id'))
                        '''破片弹的杀伤逻辑之后补充'''

        self.env_feedback['live_ids'] = live_ids

    def obs_sector_sampling(self, agents_data):
        obs_sector_dict = {}
        grid_size = self.grid_size
        H, W = self.grid_map.shape
        panel_center = np.array([W * grid_size * 0.5, H * grid_size * 0.5])
        origin_x, origin_y = (-panel_center[0], -panel_center[1])
        xi = 0.5 * np.pi / self.sector_num
        sector_edges = np.linspace(0, 2 * np.pi, self.sector_num + 1)
        
        for agent in agents_data:
            R = 100  # 参数未统一配置，别处也使用了该参数
            pos = agent.get('position')
            yaw = agent.get('angle')
            agent_id = agent.get('id')

            # ==========================================
            # 1. 提取静态障碍物坐标 (Static Obstacles)
            # ==========================================
            x_idx_global = int((pos[0] - origin_x) / grid_size)
            y_idx_global = int((pos[1] - origin_y) / grid_size)

            local_half = int(np.ceil(R / grid_size))
            x_start = max(0, x_idx_global - local_half)
            x_end   = min(W, x_idx_global + local_half + 1)
            y_start = max(0, y_idx_global - local_half)
            y_end   = min(H, y_idx_global + local_half + 1)

            local_grid = self.grid_map[y_start:y_end, x_start:x_end]
            obs_ys, obs_xs = np.where(local_grid == 1)

            if len(obs_xs) > 0:
                static_obs_x = (x_start + obs_xs) * grid_size + origin_x
                static_obs_y = (y_start + obs_ys) * grid_size + origin_y
            else:
                static_obs_x = np.array([])
                static_obs_y = np.array([])

            # ==========================================
            # 2. 提取友方集群坐标 (Dynamic Neighbors)
            # ==========================================
            neigh_info = agent.get('neigh_info', {})
            neigh_positions = neigh_info.get('position', {})
            
            neigh_x_list = []
            neigh_y_list = []
            for n_id, n_pos in neigh_positions.items():
                neigh_x_list.append(n_pos[0])
                neigh_y_list.append(n_pos[1])
                
            dynamic_obs_x = np.array(neigh_x_list)
            dynamic_obs_y = np.array(neigh_y_list)

            # ==========================================
            # 3. 坐标合并与距离计算 (Combine & Calculate)
            # ==========================================
            all_obs_x = np.concatenate([static_obs_x, dynamic_obs_x])
            all_obs_y = np.concatenate([static_obs_y, dynamic_obs_y])

            if len(all_obs_x) == 0:
                # 没有任何障碍物（静态或动态）
                obs_sector = np.full(self.sector_num, R)
            else:
                dx = all_obs_x - pos[0]
                dy = all_obs_y - pos[1]
                distances = np.hypot(dx, dy)
                mask = distances <= R
                
                if not np.any(mask):
                    obs_sector = np.full(self.sector_num, R)
                else:
                    dx, dy = dx[mask], dy[mask]
                    distances = distances[mask]
                    abs_angles = np.arctan2(dy, dx)
                    rel_angles = (abs_angles - yaw) % (2 * np.pi)

                    obs_sector = np.full(self.sector_num, R)
                    for i in range(self.sector_num):
                        # 扩大扇区范围：左右各扩展 xi
                        left_bound = sector_edges[i] - xi
                        right_bound = sector_edges[i + 1] + xi
                        
                        # 处理环形边界：将 rel_angles 复制一份 +2π 用于 wrap-around 检测
                        in_sec = (
                            ((rel_angles >= left_bound) & (rel_angles < right_bound)) |
                            ((rel_angles + 2 * np.pi >= left_bound) & (rel_angles + 2 * np.pi < right_bound))
                        )
                        if np.any(in_sec):
                            obs_sector[i] = min(obs_sector[i], np.min(distances[in_sec]))
                            
            obs_sector_dict[agent_id] = obs_sector
            
        self.env_feedback['obs_sector_dict'] = obs_sector_dict
 
    def env_update(self,agents_data):
        self.attack_results(agents_data)
        self.smoke_area(agents_data)
        self.obs_sector_sampling(agents_data)
        
        
    
