# sim/sim_initialize.py
import numpy as np
import yaml

from generate.generate_map import MapGenerator
from agent.agent_core import Agent
from comm.msg_pool import MsgPool
from sim.physics_engine import PhysicsEngine
from sim.train_sim_core_lower import RLEnvAdapter

def sim_initialize(i=None):
    # 默认读取 config.yaml，但也支持传入具体路径或编号
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

    # 3. 创建物理引擎
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
            Agent(agent_id[k], pos[k], theta, # theta作为二维初始速度传入agent初始化
            agent_dT[k],
            agent_side[k]
        ))
        engine.group_ids[int(agent_side[k])].append(agent_id[k])
    
    engine.env_feedback['live_ids'] = agent_id

    # 6. 解析关系与结构
    com_tensor = np.array(config["agents"]["com_tensor"])
    init_channel = msg_pool.channel_id[0:agent_num]
    target_distance = {k: np.array(v) for k, v in config["target_distance"].items()}
    formation_structure = {
        tuple(map(int, k.split("-"))): np.array(v)
        for k, v in config["formation_structure"].items()
    }
    
    # 7. 初始化
    engine.init_agents(
        agents,
        com_tensor,
        init_channel,
        target_distance,
        formation_structure
    )
    
    # 8. 返回 RL 适配器, 仿真默认使用并行模式
    env = RLEnvAdapter(engine, agent_id)
    return env

def main():
    try:
        env = sim_initialize()
        print("Visualization environment initialized successfully.")
    except Exception as e:
        print(f"Init failed: {e}")

if __name__ == "__main__":
    main()