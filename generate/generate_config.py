# generate_config.py
import os
import numpy as np
import yaml
import random
from typing import List, Dict, Tuple

# 导入地图生成器
from generate.generate_map import MapGenerator

# ======================
# 1. 地图相关
# ======================
# map_params = [width, height, full_size_map_path, d_sample_map_path, map_Fixed, isBlank, d_sample_hw,obs_dense[0],obs_dense[1]]
def generate_or_load_map(map_params : list) -> np.ndarray:
    """生成或加载障碍物地图"""
    if map_params[4]:
        obs_map = np.load(map_params[2])
        d_spl_map = np.load(map_params[3])
        return obs_map, d_spl_map
    map_gen = MapGenerator(map_params[0], map_params[1], map_params[5], map_params[7], map_params[8])
    map_gen.generate_map(map_params[6])
    # grid_map = map_gen.layers["obs_map"]
    obs_map = map_gen.obs_map
    d_spl_map = map_gen.down_sampled_map
    np.save(map_params[2], obs_map)
    np.save(map_params[3], d_spl_map)
    print(f"✅ 地图已保存至: {map_params[2]} 、{map_params[3]}")
    return obs_map, d_spl_map

def get_random_positions(grid_map: np.ndarray, agent_num: int, red_num: int, grid_size: float, min_clearance: int = 5) -> np.ndarray:
    """在无障碍区域生成初始物理坐标，前 red_num 个在上半场，其余在下半场"""
    if red_num > agent_num:
        raise ValueError(f"逻辑错误：red_num ({red_num}) 不能大于 agent_num ({agent_num})！")

    blue_num = agent_num - red_num
    
    h, w = grid_map.shape
    center_h, center_w = h / 2.0, w / 2.0
    
    # 建立两个独立的备选库
    valid_cells_top = []
    valid_cells_bottom = []
    
    for r in range(min_clearance, h - min_clearance):
        for c in range(min_clearance, w - min_clearance):
            # 1. 检查中心安全禁区 (曼哈顿距离)
            if abs(r - center_h) + abs(c - center_w) <= 10:
                continue
                
            # 2. 过滤障碍点
            if grid_map[r, c] != 0:
                continue
                
            # 3. 过滤 Clearance 不足的区域
            window = grid_map[
                r - min_clearance : r + min_clearance + 1,
                c - min_clearance : c + min_clearance + 1
            ]
            if np.any(window == 1):
                continue
                
            # 4. 核心改动：按行号归类到上下半场
            if r >= center_h:
                valid_cells_top.append((r, c))
            else:
                valid_cells_bottom.append((r, c))
                
    # 分别进行容量安全校验
    if len(valid_cells_top) < red_num:
        raise RuntimeError(
            f"上半场合法格子不足！需要 {red_num} 个，仅有 {len(valid_cells_top)} 个。"
        )
    if len(valid_cells_bottom) < blue_num:
        raise RuntimeError(
            f"下半场合法格子不足！需要 {blue_num} 个，仅有 {len(valid_cells_bottom)} 个。"
        )
        
    # 分别独立无放回抽样
    selected_top = random.sample(valid_cells_top, red_num)
    selected_bottom = random.sample(valid_cells_bottom, blue_num)

    # 合并结果，保证前 red_num 个必定来自上半场
    selected = selected_top + selected_bottom

    # 转换为物理坐标
    positions = [
        [(c - w / 2.0) * grid_size, (r - h / 2.0) * grid_size]
        for r, c in selected
    ]
    
    return np.array(positions)

# ======================
# 2. 通信与关系张量
# ======================
REL_COOP = np.array([1, 0, 0], dtype=int)
REL_ATTK = np.array([0, 0, 1], dtype=int)
REL_CHASE = np.array([0, 1, 0], dtype=int)

def build_com_tensor(agent_num: int, agent_side: np.ndarray) -> np.ndarray:
    """构建通信/关系张量 (N, N, 3)"""
    com_tensor = np.zeros((agent_num, agent_num, 3), dtype=int)
    for i in range(agent_num):
        for j in range(agent_num):
            if i == j:
                continue
            if agent_side[i] == agent_side[j]:
                com_tensor[i, j] = REL_COOP
            else:
                com_tensor[i, j] = REL_ATTK
    return com_tensor

# ======================
# 3. 目标方向角 & target_distance
# ======================
def compute_target_angles(agent_side: np.ndarray, agent_id: List[int]) -> Dict[str, List[float]]:
    """为每个 agent 分配目标方向（同阵营均匀分布，敌对阵营对面）"""
    sides = np.unique(agent_side)
    side_agents = {side: [] for side in sides}
    for i, sid in enumerate(agent_side):
        side_agents[sid].append(agent_id[i])

    target_distance = {}
    for side, ids in side_agents.items():
        n = len(ids)
        for idx, aid in enumerate(ids):
            angle = 2 * np.pi * idx / n
            if side == 0:  # 假设 0 是蓝方，从对面出发
                angle += np.pi
            angle = angle % (2 * np.pi)
            dx = float(np.cos(angle))
            dy = float(np.sin(angle))
            target_distance[str(aid)] = [dx, dy]
    return target_distance


# ======================
# 4. 编队结构
# ======================
def build_formation_structure(
    agent_id: List[int],
    agent_side: np.ndarray,
    pos: np.ndarray,
    com_tensor: np.ndarray,
    x_sq: float = 10.0,
    y_sq: float = 10.0
) -> Dict[str, List[float]]:
    """仅对合作邻居构建编队偏移"""
    formation = {}
    agent_num = len(agent_id)
    id_to_idx = {aid: i for i, aid in enumerate(agent_id)}

    for i in range(agent_num):
        for j in range(agent_num):
            if i == j:
                continue
            if np.array_equal(com_tensor[i, j], REL_COOP):
                if agent_side[i] == 1:  # 仅蓝方有编队
                    dx = x_sq * np.sign(pos[j, 0] - pos[i, 0])
                    dy = y_sq * np.sign(pos[j, 1] - pos[i, 1])
                else:
                    dx = x_sq * np.sign(pos[j, 0] - pos[i, 0])
                    dy = y_sq * np.sign(pos[j, 1] - pos[i, 1])
                key = f"{agent_id[i]}-{agent_id[j]}"
                formation[key] = [float(dx), float(dy)]
    return formation


# ======================
# 5. Agent 配置分配（核心新增）
# ======================
def assign_agent_profiles(agent_side: np.ndarray) -> List[str]:
    """
    为每个 agent 分配配置 profile 名称。
    可根据 side 或随机选择预定义类型。
    """
    profiles = []
    for side in agent_side:
        if side == 0:
            # 红方
            # profile = random.choice(["blue_recon", "blue_comm"])
            profile = "water"
        else:
            # 蓝方
            # profile = random.choice(["red_fire", "red_cover"])
            profile = "land"
        profiles.append(profile)
    return profiles

# ======================
# 6. 主函数
# ======================
def generate_agent_config(i, rb_num : list = [1,0], obs_dense : list = [30,0.5]):
    # random.seed(42)
    # === 基础参数 ===
    width, height = 256, 256
    grid_size = int(3)
    d_sample_hw = np.array([32,32])
    side_num = {"red" : rb_num[0], "blue" : rb_num[1]}
    agent_num = sum(side_num.values())
    with open("./version.yaml", "r") as f:
        version_config = yaml.safe_load(f)
    version = version_config["version"]["id"]
    config_save_path = f"E:/code/v3/version{version}/sim/config_data/config{i}.yaml"
    full_size_map_path= f"E:/code/v3/version{version}/sim/map_data/grid_map{i}.npy"
    d_sample_map_path = f"E:/code/v3/version{version}/sim/map_data/d_spl_map{i}.npy"
    
    # === 1. 地图 ===
    map_Fixed = False
    isBlank = True if random.random() < 0.05 else False
    map_params = [width, height, full_size_map_path, d_sample_map_path, map_Fixed, isBlank, d_sample_hw,obs_dense[0],obs_dense[1]]
    obs_map, d_spl_map = generate_or_load_map(map_params)
    # === 2. Agent 基础属性 ===
    agent_dT = [0.02] * agent_num
    agent_id = random.sample(range(1, agent_num * 5), agent_num)
    pos = get_random_positions(d_spl_map, agent_num, side_num['red'], grid_size * width // d_sample_hw[0], min_clearance=2)
    theta = [[2*random.random()-1 for _ in range(2)] for _ in range(agent_num)]
    agent_side = np.array([0] * side_num["red"] + [1] * side_num["blue"])

    # === 3. 关系与通信 ===
    com_tensor = build_com_tensor(agent_num, agent_side)

    # === 4. 目标方向 ===
    target_distance = compute_target_angles(agent_side, agent_id)

    # === 5. 编队结构 ===
    formation_structure = build_formation_structure(agent_id, agent_side, pos, com_tensor)

    # === 6. 【新增】Agent 配置 profile 分配 ===
    agent_profiles = assign_agent_profiles(agent_side)

    # === 7. 构建完整配置字典 ===
    config = {
        "map": {
            "width": width,
            "height": height,
            "grid_size": grid_size,
            "d_sample_hw": d_sample_hw.tolist(),
            "save_path": [full_size_map_path, d_sample_map_path]
        },
        "agents": {
            "num": agent_num,
            "dT": agent_dT,
            "id": agent_id,
            "theta": theta,
            "pos": pos.tolist(),
            "side": agent_side.tolist(),
            "com_tensor": com_tensor.tolist(),
            "profiles": agent_profiles  # ← 新增字段！
        },
        "target_distance": target_distance,
        "formation_structure": formation_structure
    }

    # === 8. 保存 ===
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, indent=2, allow_unicode=True)

    print(f"✅ 配置文件已保存至: {config_save_path}")
    print("Agent Profiles:", dict(zip(agent_id, agent_profiles)))
    return agent_num


if __name__ == "__main__":
    num = generate_agent_config(0)
    print(f"智能体数量: {num}")