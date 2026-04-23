# monte_karlo_sampling.py
import os
import yaml
import numpy as np

# 设置随机种子（可选，便于复现）
# np.random.seed(42)

def sample_agent_profile(name: str):
    """为某一类智能体（如 'default'）生成随机配置 , 后续通过此代码配置实现异构"""
    config = {}
    # 辅助函数：采样后四舍五入到2位小数
    def rnd(x):
        return round(float(x), 2)
    # === 装备性能指标 ===
    config["v_max"] = rnd(np.random.uniform(100, 140))
    config["r_turn_min"] = rnd(np.random.uniform(20, 30))
    config["s_max"] = 10000  # 整数，不变

    config["sense_field"] = rnd(np.random.uniform(250, 350))
    config["sense_angle_deg"] = rnd(np.random.uniform(30, 45))
    config["sense_variance"] = rnd(np.random.uniform(0.1, 0.3))

    config["attack_range"] = int(np.random.uniform(150, 200))      # 整数
    config["cannon_w_max_deg"] = rnd(np.random.uniform(800, 1200))
    config["launch_delay"] = int(np.random.randint(8, 15))         # 整数
    config["num_per_launch"] = int(1)                                   # 固定整数
    config["attk_radius"] = rnd(np.random.uniform(10, 20))
    config["attk_variance"] = rnd(np.random.uniform(0.1, 0.3))
    config["cannon_capacity"] = int(np.random.randint(8, 12))      # 整数

    config["smoke_capacity"] = int(5)      # 整数
    config["reflective_surface"] = rnd(np.random.uniform(0.8, 1.2))
    config["exposed_area"] = rnd(2.25 * np.pi)     # ≈7.07

    config["decision_delay"] = int(np.random.randint(2, 5))        # 整数
    config["task_preference"] = int(np.random.choice([1, 2, 3, 4]))
    config["task_assignment"] = int(np.random.choice([0, 1]))
    config["weapon_assignment"] = int(np.random.choice([0, 1]))
    config["connect_dist"] = int(np.random.uniform(80, 120))       # 整数
    return config

def generate_agent_params(output_path="agent/agent_config.yaml", profiles=None):
    if profiles is None:
        profiles = ["default"]

    full_config = {}
    for name in profiles:
        print(f"Sampling config for '{name}'...")
        full_config[name] = sample_agent_profile(name)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 写入 YAML
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(full_config, f, allow_unicode=True, sort_keys=False, indent=2)

    print(f"✅ Agent config saved to: {output_path}")

if __name__ == "__main__":
    generate_agent_params(profiles=["default","water","land"])