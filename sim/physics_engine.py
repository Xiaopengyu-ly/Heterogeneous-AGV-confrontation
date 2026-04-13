# sim/physics_engine.py
import numpy as np
import copy
from comm.msg_pool import MsgPool
from sim_env.env_model import env_model

class PhysicsEngine(env_model):
    def __init__(self, map_layers=None, grid_size=5, dT=0.1):
        super().__init__()
        # --- 1. 基础属性 ---
        if map_layers is None:
            raise ValueError("map_layers must be provided.")
        self.map_layers = map_layers
        self.grid_map = self.map_layers.obs_map
        self.smoke = self.map_layers.smoke
        self.d_spl_map = self.map_layers.down_sampled_map
        self.grid_size = grid_size
        self.dT = dT
        self.steps = 0
        
        # --- 环境参数 ---
        self.smoke_radius = 60
        self.smoke_last_time = 10
        self.sector_num = 36
        self.cannon_attk_results = []
        
        # --- 2. 状态容器 ---
        self.agents = []
        self.msg_pool = MsgPool()
        self.channel_dict = {}
        self.group_ids = {0: [], 1: []}
        
        # --- 3. 环境反馈信息 ---
        self.env_feedback = {
            'live_ids': [],
            'channel_dict': {},
            'obs_sector_dict': {},
            'smoke_zone': [],
        }
        
        # 快照容器
        self.initial_state = None 

    def init_msgpool(self, msg_pool: MsgPool):
        self.msg_pool = msg_pool
        self.msg_pool.check()
    
    def init_agents(self, agents: list, com_tensor: np.ndarray, init_channel: list,
                    target_distance: dict, formation_structure: dict, goal_config: dict = None):
        """
        重写 init_agents，实现更可控的初始化和快照保存
        """
        self.agents = agents
        agent_ids = [agent.id for agent in agents]
        
        # === [核心修复 1] 补全父类 env_model 所需的关键计数器 ===
        # env_update 往往依赖这些属性来遍历智能体，如果缺失会导致循环跳过，live_ids 变空
        self.agent_num = len(agents)
        self.blue_ids = self.group_ids.get(0, [])
        self.red_ids = self.group_ids.get(1, [])
        self.agents_id = agent_ids # 某些老版本 env_model 可能用这个名字

        # === [核心修复 2] 强制初始化 live_ids ===
        self.env_feedback['live_ids'] = list(agent_ids)

        neighbors_dict = {}
        targets_dict = {}
        cannon_targets_dict = {}
        self.channel_dict = {}

        # 1. 解析通信矩阵
        REL_COOP = np.array([1, 0, 0])
        REL_CHASE = np.array([0, 1, 0])
        REL_ATTK = np.array([0, 0, 1])

        for i, row in enumerate(com_tensor):
            neighbors = []
            targets = []
            cannon_targets = []
            for j, rel_vec in enumerate(row):
                if i == j: continue
                if np.array_equal(rel_vec, REL_COOP):
                    neighbors.append((agent_ids[j], 1))
                elif np.array_equal(rel_vec, REL_CHASE):
                    targets.append((agent_ids[j], 1))
                    # cannon_targets.append((agent_ids[j], 1)) # 加上该句就是又追又打
                elif np.array_equal(rel_vec, REL_ATTK):
                    cannon_targets.append((agent_ids[j], 1))
            
            sid = str(agent_ids[i])
            neighbors_dict[sid] = neighbors
            targets_dict[sid] = targets
            cannon_targets_dict[sid] = cannon_targets

        for i in range(len(init_channel)):
            self.channel_dict[str(agent_ids[i])] = init_channel[i]

        
        # 2. 初始化每个 Agent
        for agent in self.agents:
            agent.get_connect(self.msg_pool)
            agent.get_init_parameters(
                self.channel_dict[str(agent.id)],
                target_distance[f'{agent.id}']
            )
            agent.get_net_neighbors(
                neighbors_dict,
                self.channel_dict, 
                formation_structure, 
                targets_dict, 
                cannon_targets_dict
            )
            agent.get_grid_map(self.map_layers, self.grid_size)
        
        self.env_feedback['channel_dict'] = self.channel_dict
        
        # 3. 创建精准快照
        self.initial_state = {
            'agents': copy.deepcopy(self.agents),
            'group_ids': copy.deepcopy(self.group_ids),
            # 同时也保存父类属性，以防 reset 后丢失
            'env_model_attrs': {
                'agent_num': self.agent_num,
                'blue_ids': self.blue_ids,
                'red_ids': self.red_ids,
                'agents_id': self.agents_id
            },
            'env_feedback': copy.deepcopy(self.env_feedback),
            'channel_dict': copy.deepcopy(self.channel_dict),
            'smoke': copy.deepcopy(self.smoke)
        }

    def step_physics(self, controllers=None):
        agents_data = self._get_agent_data_struct() 
        self.env_update(agents_data)
        for agent in self.agents:
            agent.check_env(self.env_feedback)
            if controllers is not None and agent.id in controllers:
                action = controllers[agent.id]
                case = "RL_Actor" 
                agent.get_route_point(case, action)
            else:
                agent.get_route_point("A_star", np.array([0,0,0,0,0]))
            agent.update() 
        self.steps += 1

    def _get_agent_data_struct(self):
        agent_data = []
        for agent in self.agents:
            agent_data.append({
                'id': agent.id,
                'side': agent.side,
                'disabled': agent.disabled,
                'position': agent.position.copy(),
                'v' : agent.v,
                'w' : agent.w,
                'angle': agent.theta,
                't_id': agent.targets_id,
                'rpoint': agent.r_point,
                'p_pos': agent.p_pos,
                'sense_range': agent.sense_field,
                'sense_angle': agent.sense_angle,
                'ct_id': agent.cannon_targets_id,
                'WPangle': agent.cannon_theta,
                'ATTKtimes': agent.cannon_capacity - agent.cannon_remain,
                'ATTKpos': agent.attk_pos,
                'ATTKlaunched': agent.cannon_launched,
                'ATTKradius': agent.attk_radius,
                'SMOKE': agent.smoke_mission,
                'neigh_info' : agent.neighbors_info,
            })
        return agent_data

    def get_render_data(self):
        agent_data = self._get_agent_data_struct()
        env_data = {'SmokeArea': self.smoke}
        return {
            'agents': agent_data,
            'env': env_data,
            'grid_map': self.grid_map,
            'grid_size': self.grid_size,
        }

    def reset_engine(self):
        """
        精准重置：只恢复智能体和环境状态
        """
        if self.initial_state is not None:
            # 从快照恢复
            self.agents = copy.deepcopy(self.initial_state['agents'])
            self.group_ids = copy.deepcopy(self.initial_state['group_ids'])
            
            # [新增] 恢复 env_model 属性
            attrs = self.initial_state.get('env_model_attrs', {})
            for k, v in attrs.items():
                setattr(self, k, v)
                
            self.env_feedback = copy.deepcopy(self.initial_state['env_feedback'])
            self.channel_dict = copy.deepcopy(self.initial_state['channel_dict'])
            self.smoke = copy.deepcopy(self.initial_state['smoke'])
            
            for agent in self.agents:
                agent.get_connect(self.msg_pool)
        else:
            if len(self.agents) == 0:
                raise RuntimeError("No agents initialized! Please add agents before reset.")
        
        self.steps = 0