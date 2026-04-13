# RL_train/train_initialize.py
import numpy as np
import yaml

from sim_env.map_generator import MapGenerator
from agent.agent_core import Agent
from comm.msg_pool import MsgPool
from sim.physics_engine import PhysicsEngine
from RL_train.train_sim_core_lower import RLEnvAdapter
# from RL_train.swarm_pettingzoo_env import SwarmPettingZooEnv
def train_initialize(i):
    # 1. 读取配置文件
    with open("./version.yaml", "r") as f:
        version_config = yaml.safe_load(f)
    version = version_config["version"]["id"]
    config_path = f"E:/code/v3/version{version}/sim/config_data/config{i}.yaml" if isinstance(i, int) else i
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. 地图初始化
    width = config["map"]["width"]
    height = config["map"]["height"]
    grid_size = config["map"]["grid_size"]
    save_path = config["map"]["save_path"]

    map_layers = MapGenerator(width, height)
    map_layers.load_map(np.load(save_path[0]), np.load(save_path[1]))

    # 3. 创建物理引擎 (Phase 1 的类)
    engine = PhysicsEngine(map_layers=map_layers, grid_size=grid_size, dT=0.01)

    # 4. 初始化消息池
    msg_pool = MsgPool()
    engine.init_msgpool(msg_pool)

    # 5. 准备智能体列表
    agent_num = config["agents"]["num"]
    agent_id = config["agents"]["id"]
    pos = np.array(config["agents"]["pos"])
    agent_dT = config["agents"]["dT"]
    agent_side = np.array(config["agents"]["side"])
    theta = np.array(config["agents"]["theta"])
    agents = []
    for k in range(agent_num):
        agents.append(
            Agent(agent_id[k], pos[k], theta,
            agent_dT[k],
            agent_side[k]
        ))
        engine.group_ids[int(agent_side[k])].append(agent_id[k])
    
    # 初始化反馈信息中的 live_ids (防止第一帧报错)
    engine.env_feedback['live_ids'] = agent_id

    # 6. 解析关系与结构
    com_tensor = np.array(config["agents"]["com_tensor"])
    init_channel = msg_pool.channel_id[0:agent_num]
    target_distance = {k: np.array(v) for k, v in config["target_distance"].items()}
    formation_structure = {
        tuple(map(int, k.split("-"))): np.array(v)
        for k, v in config["formation_structure"].items()
    }

    # 7. 初始化引擎中的智能体
    engine.init_agents(
        agents,
        com_tensor,
        init_channel,
        target_distance,
        formation_structure
    )
    
    # 8. 创建 RL 环境适配器 "parl"为并行训练；"single"为单体训练
    env = RLEnvAdapter(engine, agent_id)
    # env = SwarmPettingZooEnv(engine, agent_id)
    return env

def main():
    # 简单的冒烟测试
    try:
        sim = train_initialize(0)
        print("RL Environment initialized successfully!")
    except Exception as e:
        print(f"Initialization failed: {e}")

if __name__ == "__main__":
    main()