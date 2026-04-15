import numpy as np
import math
from gymnasium import spaces
from comm.msg_pool import MsgPool
from generate.generate_map import MapGenerator

'''
    [重构版] 数据管理组件
    负责 Get 逻辑
'''

class DataSystem:
    def __init__(self, agent):
        self.agent = agent

    def get_connect(self, pool: MsgPool):
        self.agent.msg_pool = pool

    def get_init_parameters(self, channel_id: int, target_distance: np.ndarray):
        self.agent.hit_rpoint = False
        self.agent.channel_id = channel_id
        self.agent.target_distance = target_distance
        self.agent.p_pos = self.agent.TARGET_POS + 20 * target_distance # 编队目标点
        self.agent.t_pos = self.agent.TARGET_POS # 编队中心点
        self.agent.obs_sector = [100 for _ in range(self.agent.sector_num)]

    def get_net_neighbors(self, neighbors_dict: dict, channel_dict: dict, formation_structure: dict, targets_dict: dict, cannon_targets_dict: dict):
        nodes = neighbors_dict[f"{self.agent.id}"]
        self.agent.neighbors_id = []
        self.agent.neighbors_info = {
            "weight": {},
            "position": {},
            "velo": {},
            "channelid": {},
            "formdist": {},
            'obssector': {}
        }
        for iter in range(len(nodes)):
            nid = nodes[iter][0]
            weight = nodes[iter][1]
            self.agent.neighbors_id.append(nid)
            self.agent.neighbors_info["weight"][f"{nid}"] = weight
            self.agent.neighbors_info["channelid"][f"{nid}"] = channel_dict[f"{nid}"]
            self.agent.neighbors_info["position"][f"{nid}"] = np.array([0, 0])
            self.agent.neighbors_info["velo"][f"{nid}"] = np.array([0, 0])
            self.agent.neighbors_info["formdist"][f"{nid}"] = formation_structure.get((self.agent.id, nid))
        
        # Targets Logic
        nodes = targets_dict[f"{self.agent.id}"]
        self.agent.targets_id = []
        self.agent.targets_info = {
            "weight": {},
            "position": {},
            "velo": {},
            "channelid": {},
            "formdist": {}
        }
        for iter in range(len(nodes)):
            nid = nodes[iter][0]
            weight = nodes[iter][1]
            self.agent.targets_id.append(nid)
            self.agent.targets_info["weight"][f"{nid}"] = weight
            self.agent.targets_info["channelid"][f"{nid}"] = channel_dict[f"{nid}"]
            self.agent.targets_info["position"][f"{nid}"] = self.agent.TARGET_POS.copy()
            self.agent.targets_info["velo"][f"{nid}"] = np.array([0, 0])
            self.agent.targets_info["formdist"][f"{nid}"] = formation_structure.get((self.agent.id, nid))
        self.agent.targets_id = self.agent.targets_id[0] if self.agent.targets_id else 0
        print(self.agent.id, ' CHASE,' , self.agent.targets_id, ' ,', self.agent.targets_info)
        
        # Cannon Targets Logic
        nodes = cannon_targets_dict[f"{self.agent.id}"]
        self.agent.cannon_targets_id = []
        self.agent.cannon_targets_info = {
            "weight": {},
            "position": {},
            "velo": {},
            "channelid": {},
            "formdist": {}
        }
        for iter in range(len(nodes)):
            nid = nodes[iter][0]
            weight = nodes[iter][1]
            self.agent.cannon_targets_id.append(nid)
            self.agent.cannon_targets_info["weight"][f"{nid}"] = weight
            self.agent.cannon_targets_info["channelid"][f"{nid}"] = channel_dict[f"{nid}"]
            self.agent.cannon_targets_info["position"][f"{nid}"] = self.agent.TARGET_POS.copy()
            self.agent.cannon_targets_info["velo"][f"{nid}"] = np.array([0, 0])
            self.agent.cannon_targets_info["formdist"][f"{nid}"] = formation_structure.get((self.agent.id, nid))
        self.agent.cannon_targets_id = self.agent.cannon_targets_id[0] if self.agent.cannon_targets_id else 0
        print(self.agent.id, ' ATTK,' , self.agent.cannon_targets_id, ' ,', self.agent.cannon_targets_info)

    def get_trajectory(self):
        return self.agent.trajectory

    def get_grid_map(self, map_layers: MapGenerator = None, grid_size: int = 1):
        self.agent.grid_size = grid_size
        self.agent.grid_map = map_layers.obs_map
        self.agent.down_sampled_map = map_layers.down_sampled_map
    
    def _angle_diff(self, a, b):
        diff = a - b
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def get_route_point(self, case: str = "RL_Actor", action : np.ndarray = np.array([0,0,0,0,0]) ):
        match case:
            case "mid":
                self.agent.r_point = (self.agent.t_pos + self.agent.position) / 2
                print(self.agent.r_point, "from agentpos", self.agent.position, "\n")
            case "RL_Actor":
                raw_state = action 
                self.agent.prev_r_point = self.agent.r_point
                self.agent.r_point = np.array([
                    raw_state[0],
                    raw_state[1],
                    raw_state[2],
                    raw_state[3],
                    raw_state[4]
                ])
        return self.agent.r_point