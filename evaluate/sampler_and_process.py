import os
import glob
import pickle
import numpy as np
import yaml

# 导入你现有的模块
from sim.sim_initialize import sim_initialize
from generate.generate_config import generate_agent_config
from stable_baselines3 import SAC
from sim.sim_controller import SimulationController

def sampler(num_samples=100):
    """
    自动化蒙特卡洛仿真采样
    自动生成配置 -> 运行仿真 -> 导出 pkl 轨迹
    """
    print(f"=== 开始执行大规模仿真采样 (共 {num_samples} 局) ===")
    os.makedirs("sim_replay", exist_ok=True)
    os.makedirs("sim/config_data", exist_ok=True)
    
    # 尝试加载固化的底层策略 (SAC)，如果没有则后续可能需要使用启发式随机动作
    try:
        model = SAC.load("sac_policy")
        print(">>> 成功加载 SAC 策略模型 sac_policy.zip")
    except Exception as e:
        print(f">>> 警告：无法加载 sac_policy，将使用随机动作或启发式动作。({e})")
        model = None

    config_id = 0
    count = 0
    for i in range(num_samples):
        print(f"--- 正在运行第 {i+1}/{num_samples} 局仿真 ---")
        
        # 1. 本阶段效能评估，固定初始config配置，单纯评估内部各类概率性模型
        # 2. 初始化环境
        sim = sim_initialize(config_id)
        count += 1
        if count == 20:
            config_id = 1
            generate_agent_config(config_id)
            count = 0
        
        # 3. 配置控制器 (无 GUI 模式)
        config = {
            "case": 0,
            "max_steps": 300, # 限制每局最大步数，防止死锁
            "data_id": i,
            "buffer_capacity": 1500
        }
        controller = SimulationController(sim, config)
        
        # 注入模型
        if model is not None:
            # 修改 SimulationController 内部引用，让其能用到 SAC
            controller.lower_model = model 
            
        # 4. 无头执行仿真直到结束
        while controller.should_continue():
            # 控制器自动计算动作 -> 推进物理引擎 -> 存入 replay_buffer
            controller.step()
            
            # 如果到达最大步数或任务全部完成（all_done），should_continue 会返回 False
            
        # 5. 保存轨迹数据到 sim_replay/{i}.pkl
        save_path = f"sim_replay/{i}.pkl"
        controller.replay_buffer.save_buffer(save_path)
        print(f"    第 {i+1} 局完成，经历了 {controller.step_count} 步，轨迹已保存至 {save_path}。")
        
    print("=== 所有仿真采样结束 ===")


def data_processer_for_TwoTower_Mapping():
    """
    读取 sim_replay 目录下的 .pkl 仿真轨迹，提取双重映射模型所需的三元组数据集
    - P (装备性能): 提取 agent_config.yaml 中的 water/land 各 20 个参数
    - T (体系能力): 按红蓝阵营分开计算，仅计算到全灭或 MAX_SEQ_LEN
    - E (任务效能): 每局包含 2 个指标 (红方存活率, 蓝方存活率)
    """
    print("\n=== 开始提取双重映射模型数据集 ===")
    files_pattern = "*.pkl"
    source_dir = "sim_replay"
    dataset_path = "dataset/dual_mapping"
    
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    P_dataset = [] # 装备性能 (Static) -> shape: (N*2, 20)
    T_dataset = [] # 体系能力 (Time Series) -> shape: (N*2, MAX_SEQ_LEN, 2)
    E_dataset = [] # 任务效能 (Scalar) -> shape: (N, 2)
    
    MAX_SEQ_LEN = 300 # 僵持状态下的最大截断步数
    
    files = glob.glob(os.path.join(source_dir, files_pattern))
    print(f">>> 找到 {len(files)} 个待处理的仿真轨迹文件")
    
    # ==========================================
    # 预先读取 agent_config.yaml 获取红蓝双方装备参数 (20维)
    # ==========================================
    config_path = "agent/agent_config.yaml"
    red_params = np.zeros(20, dtype=np.float32)
    blue_params = np.zeros(20, dtype=np.float32)
    
    # 定义需要提取的 20 个装备性能指标 Key
    param_keys = [
        "v_max", "r_turn_min", "s_max", "sense_field", "sense_angle_deg", 
        "sense_variance", "attack_range", "cannon_w_max_deg", "launch_delay", "num_per_launch", 
        "attk_radius", "attk_variance", "cannon_capacity", "smoke_capacity", "reflective_surface", 
        "exposed_area", "decision_delay", "task_preference", "task_assignment", "weapon_assignment"
    ]
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as cfg_f:
            cfg = yaml.safe_load(cfg_f)
            water_cfg = cfg.get('water', {})
            land_cfg = cfg.get('land', {})
            # 红方对应 water，蓝方对应 land
            red_params = np.array([water_cfg.get(k, 0.0) for k in param_keys], dtype=np.float32)
            blue_params = np.array([land_cfg.get(k, 0.0) for k in param_keys], dtype=np.float32)
    else:
        print(f">>> 警告：未找到 {config_path}，使用全 0 参数占位。")

    for f_path in files:
        with open(f_path, 'rb') as f:
            buffer = pickle.load(f)
            
        if not buffer or len(buffer) < 5:
            continue
            
        initial_state = buffer[0][0]
        # side=0 认为是红方，side=1 认为是蓝方
        red_initial_count = sum(1 for a in initial_state if a.get('side', 0) == 0)
        blue_initial_count = sum(1 for a in initial_state if a.get('side', 0) == 1)
        
        # ==========================================
        # 1. 寻找真正的任务结束点 (task_steps)
        # ==========================================
        actual_len = len(buffer)
        task_steps = MAX_SEQ_LEN
        
        for step_idx in range(min(actual_len, MAX_SEQ_LEN)):
            state = buffer[step_idx][0]
            red_alive = sum(1 for a in state if a.get('side', 0) == 0 and not a.get('disabled', False))
            blue_alive = sum(1 for a in state if a.get('side', 0) == 1 and not a.get('disabled', False))
            
            # 如果某一方全军覆没，录入当前步数为结束点
            if red_alive == 0 or blue_alive == 0:
                task_steps = step_idx + 1
                break
                
        task_steps = min(task_steps, actual_len, MAX_SEQ_LEN)
        
        # ==========================================
        # 2. 计算【任务效能 E】
        # ==========================================
        # 截取任务结束帧，计算最终存活率
        final_state = buffer[task_steps - 1][0]
        red_alive_final = sum(1 for a in final_state if a.get('side', 0) == 0 and not a.get('disabled', False))
        blue_alive_final = sum(1 for a in final_state if a.get('side', 0) == 1 and not a.get('disabled', False))
        
        red_survival_rate = red_alive_final / red_initial_count if red_initial_count > 0 else 0.0
        blue_survival_rate = blue_alive_final / blue_initial_count if blue_initial_count > 0 else 0.0
        
        E_vec = np.array([red_survival_rate, blue_survival_rate, task_steps], dtype=np.float32)
        
        # ==========================================
        # 3. 按 task_steps 遍历计算【体系能力 T】
        # ==========================================
        T_red_seq = []
        T_blue_seq = []
        
        for step_idx in range(task_steps):
            transition = buffer[step_idx]
            state = transition[0]
            actions = transition[2] # dict: {agent_id: action_array}
            
            red_active = [a for a in state if a.get('side', 0) == 0 and not a.get('disabled', False)]
            blue_active = [a for a in state if a.get('side', 0) == 1 and not a.get('disabled', False)]
            
            # 计算红方指标 (距离 & 机动平滑度)
            red_dist = 0.0
            if len(red_active) > 1:
                dists = [np.linalg.norm(np.array(red_active[i]['position']) - np.array(red_active[j]['position'])) 
                         for i in range(len(red_active)) for j in range(i+1, len(red_active))]
                red_dist = np.mean(dists)
                
            red_mag = 0.0
            red_ids = [a['id'] for a in red_active]
            red_actions = [actions[aid][:2] for aid in red_ids if aid in actions and actions[aid] is not None]
            if red_actions:
                red_mag = np.mean([np.linalg.norm(act) for act in red_actions])
                
            T_red_seq.append([red_dist, red_mag])
            
            # 计算蓝方指标 (距离 & 机动平滑度)
            blue_dist = 0.0
            if len(blue_active) > 1:
                dists = [np.linalg.norm(np.array(blue_active[i]['position']) - np.array(blue_active[j]['position'])) 
                         for i in range(len(blue_active)) for j in range(i+1, len(blue_active))]
                blue_dist = np.mean(dists)
                
            blue_mag = 0.0
            blue_ids = [a['id'] for a in blue_active]
            blue_actions = [actions[aid][:2] for aid in blue_ids if aid in actions and actions[aid] is not None]
            if blue_actions:
                blue_mag = np.mean([np.linalg.norm(act) for act in blue_actions])
                
            T_blue_seq.append([blue_dist, blue_mag])
            
        # 转换为 Numpy 数组并 Padding 对齐到 MAX_SEQ_LEN
        T_red_seq = np.array(T_red_seq, dtype=np.float32)
        T_blue_seq = np.array(T_blue_seq, dtype=np.float32)
        
        if len(T_red_seq) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(T_red_seq)
            T_red_seq = np.pad(T_red_seq, ((0, pad_len), (0, 0)), 'constant')
            T_blue_seq = np.pad(T_blue_seq, ((0, pad_len), (0, 0)), 'constant')
            
        # ==========================================
        # 4. 组装数据 (分别将红蓝方压入)
        # ==========================================
        P_dataset.append(red_params)
        P_dataset.append(blue_params)
        
        T_dataset.append(T_red_seq)
        T_dataset.append(T_blue_seq)
        
        E_dataset.append(E_vec)

    # 最终保存
    if len(E_dataset) > 0:
        P_dataset = np.stack(P_dataset)
        T_dataset = np.stack(T_dataset)
        E_dataset = np.stack(E_dataset)
        
        print(f">>> 数据解析完毕！")
        print(f"    [P] 装备性能输入 Shape: {P_dataset.shape}")
        print(f"    [T] 体系能力时序 Shape: {T_dataset.shape}")
        print(f"    [E] 任务效能标签 Shape: {E_dataset.shape}")
        
        np.save(f"{dataset_path}_P.npy", P_dataset)
        np.save(f"{dataset_path}_T.npy", T_dataset)
        np.save(f"{dataset_path}_E.npy", E_dataset)
        print(f"=== 三元组数据集已成功导出至 {os.path.dirname(dataset_path)} 目录下 ===")
    else:
        print(">>> 错误：未提取到任何有效映射数据。")

if __name__ == "__main__":
    # # 1. 先进行 10 局的小规模采样验证
    # # 如果你已经有了 pkl 文件，可以将此行注释掉
    # sampler(num_samples=200)
    
    # 2. 数据处理与提取
    data_processer_for_TwoTower_Mapping()