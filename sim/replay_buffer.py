from collections import deque
import random
import pickle
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, obs, action, reward, next_state, next_obs, done):
        self.buffer.append((state, obs, action, reward, next_state, next_obs, done))
    def reset(self):
        self.buffer.clear()
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    def __len__(self):
        return len(self.buffer)
    def save_buffer(self, filepath: str = "sim/sim_replay/0.pkl"):
        # 将 deque 转为 list（pickle 可以处理，但 list 更通用）
        buffer_list = list(self.buffer)
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_list, f)
        print(f"Replay buffer saved to {filepath} with {len(buffer_list)} transitions.")
    def read_buffer(self, filepath: str = "sim/sim_replay/0.pkl"):
        with open(filepath, 'rb') as f:
            buffer_list = pickle.load(f)
        self.buffer = deque(buffer_list, maxlen=self.buffer.maxlen)
        print(f"Replay buffer loaded from {filepath} with {len(self.buffer)} transitions.")


    def extract_action_dataset(self, filepath: str = "sim/sim_replay/0.pkl", slice_len: int = 5):
        import pickle
        import numpy as np
        
        with open(filepath, 'rb') as f:
            buffer_list = pickle.load(f)

        all_action_samples = [] 
        agent_sequences = {} 

        print(f">>> 开始解析数据: {filepath}")
        ACTION_THRESHOLD = 0.1
        STATE_DIFF_MIN = 0.001
        
        # 内部工具函数：根据 agent_id 从状态列表中提取字典
        def get_agent_state(state_list, aid):
            for s in state_list:
                if s['id'] == aid:
                    return s
            return None

        for transition in buffer_list:
            # 修改变量名以避免歧义：action_data 可能是 dict 或者是 ndarray
            state, obs, action_data, reward, next_state, next_obs, done = transition
            
            if done:
                for aid in agent_sequences:
                    agent_sequences[aid] = []
                continue
            
            # 【核心修改：兼容单智能体(ndarray)和多智能体(dict)】
            if isinstance(action_data, dict):
                action_items = action_data.items()
            else:
                # 如果是单体训练，action 是 ndarray，我们从 state 中取出第一辆车的 ID 作为键
                agent_id = state[0]['id']
                action_items = [(agent_id, action_data)]

            for agent_id, action_vec in action_items:
                s_curr_dict = get_agent_state(state, agent_id)
                s_next_dict = get_agent_state(next_state, agent_id)
                
                if s_curr_dict is None or s_next_dict is None:
                    continue

                s_curr = np.array([s_curr_dict['position'][0], s_curr_dict['position'][1], s_curr_dict['angle']])
                s_next = np.array([s_next_dict['position'][0], s_next_dict['position'][1], s_next_dict['angle']])
                
                state_diff = np.linalg.norm(s_next - s_curr)
                # 截取物理引擎实际执行的单步 5 维动作
                executed_action = action_vec[:5]
                action_mag = np.linalg.norm(executed_action)
                
                if action_mag > ACTION_THRESHOLD and state_diff < STATE_DIFF_MIN:
                    agent_sequences[agent_id] = [] 
                    continue
                    
                if agent_id not in agent_sequences:
                    agent_sequences[agent_id] = []
                
                agent_sequences[agent_id].append(executed_action)
                
                # 序列长度达到 VQ-VAE 要求的 T=3
                if len(agent_sequences[agent_id]) == slice_len:
                    sample = np.array(agent_sequences[agent_id], dtype=np.float32)
                    all_action_samples.append(sample)
                    agent_sequences[agent_id].pop(0)

        if len(all_action_samples) == 0:
            return None
            
        final_data = np.stack(all_action_samples, axis=0)
        print(f">>> 提取完成，生成样本数: {final_data.shape[0]}, 每个样本形状: {final_data.shape[1:]}")
        return final_data


    def extract_dynamics_dataset(self, horizon_len: int,
                                   filepath: str = "sim/sim_replay/0.pkl", n_frames: int = 3):
        """
        提取 Forward Model 训练数据集 (新 MPPI 格式，无需 VQ-VAE)。

        输入: 多帧观测 obs (N_frames, 36, 6) + dynamics(2) + action 序列 (T, 5)
        目标: T 帧增量 [(36, 6)] — 统一帧的逐帧变化，对应 [t+1, ..., t+T]
        """
        import numpy as np
        import pickle

        T = horizon_len

        def build_unified_frame_from_obs(obs_dict):
            """从 obs dict 构建统一帧 (36, 6)"""
            from sim.obs_utils import build_unified_frame, goal_to_lidar_mask
            lidar_2d = np.array(obs_dict['lidar_2d'], dtype=np.float32)   # (5, 36)
            goal_dir = np.array(obs_dict['goal_dir'], dtype=np.float32)   # (2,)
            rou = goal_dir[0] * 200.0         # 反归一化
            etheta = goal_dir[1] * np.pi      # 反归一化
            goal_mask = goal_to_lidar_mask(rou, etheta)  # (36,)
            return build_unified_frame(lidar_2d, goal_mask)  # (36, 6)

        with open(filepath, 'rb') as f:
            buffer_list = pickle.load(f)

        dataset_obs = []        # (N, n_frames, 36, 6)
        dataset_dynamics = []   # (N, 2)
        dataset_actions = []    # (N, T, 5)  原生 action 序列
        dataset_deltas = []     # (N, T, 36, 6)  统一帧增量
        agent_caches = {}

        print(f">>> 解析 Forward Model 数据 (MPPI 格式): {filepath} (T={T})")
        for transition in buffer_list:
            state, obs, action_data, reward, next_state, next_obs, done = transition
            if done:
                agent_caches.clear()
                continue

            # 兼容单/多智能体
            if isinstance(action_data, dict):
                action_items = action_data.items()
                if 'lidar_2d' in obs:
                    obs_dict = {aid: obs for aid in action_data.keys()}
                else:
                    obs_dict = obs
            else:
                agent_id = state[0]['id']
                action_items = [(agent_id, action_data)]
                obs_dict = {agent_id: obs}

            for agent_id, action_vec in action_items:
                if agent_id not in agent_caches:
                    agent_caches[agent_id] = {'obs': [], 'actions': []}

                agent_caches[agent_id]['obs'].append(obs_dict[agent_id])
                agent_caches[agent_id]['actions'].append(action_vec[:5])

                # 需要 T+1 帧观测 (obs[0:T] for n_frames construction, obs[1:T+1] for delta)
                needed = max(T + 1, n_frames)
                if len(agent_caches[agent_id]['actions']) >= needed:
                    # --- A. 构建多帧观测 ---
                    obs_seq = agent_caches[agent_id]['obs'][:n_frames]
                    frames = [build_unified_frame_from_obs(o) for o in obs_seq]
                    obs_input = np.stack(frames, axis=0)  # (n_frames, 36, 6)

                    # --- B. Dynamics ---
                    dyn = np.array(obs_seq[-1]['dynamics'], dtype=np.float32)  # (2,)

                    # --- C. 原生 action 序列 ---
                    action_seq = np.array(
                        agent_caches[agent_id]['actions'][:T], dtype=np.float32
                    )  # (T, 5)

                    # --- D. T 帧增量: 统一帧 (36,6) 的逐帧变化 ---
                    deltas = []
                    prev_frame = build_unified_frame_from_obs(obs_seq[0])
                    for i in range(1, T + 1):
                        curr_frame = build_unified_frame_from_obs(
                            agent_caches[agent_id]['obs'][i])
                        deltas.append(curr_frame - prev_frame)
                        prev_frame = curr_frame

                    dataset_obs.append(obs_input)
                    dataset_dynamics.append(dyn)
                    dataset_actions.append(action_seq)
                    dataset_deltas.append(np.array(deltas, dtype=np.float32))

                    # --- E. 滑动 ---
                    agent_caches[agent_id]['obs'].pop(0)
                    agent_caches[agent_id]['actions'].pop(0)

        if len(dataset_obs) == 0:
            return None

        final_obs = np.array(dataset_obs, dtype=np.float32)       # (N, n_frames, 36, 6)
        final_dynamics = np.array(dataset_dynamics, dtype=np.float32)  # (N, 2)
        final_actions = np.array(dataset_actions, dtype=np.float32)    # (N, T, 5)
        final_deltas = np.array(dataset_deltas, dtype=np.float32)      # (N, T, 36, 6)

        print(f">>> 提取完成: Obs {final_obs.shape}, Actions {final_actions.shape}, "
              f"Dynamics {final_dynamics.shape}, Deltas {final_deltas.shape}")
        return final_obs, final_dynamics, final_actions, final_deltas


def lidar2d_to_distances(lidar_2d, max_range=100.0):
    """从 lidar_2d (n_bins, 36) 重建 36 维原始距离 (米)

    可用于 Forward Model 和 Latent MPC 的 CBF 碰撞检测。
    """
    n_bins, n_sectors = lidar_2d.shape
    bin_size = max_range / n_bins
    distances = np.full(n_sectors, max_range, dtype=np.float32)
    for s in range(n_sectors):
        for b in range(n_bins):
            if lidar_2d[b, s] > 0.5:
                distances[s] = b * bin_size
                break
    return distances


import numpy as np
import matplotlib.pyplot as plt
import pickle

def visualize_filter_thresholds(filepath: str, action_threshold=0.5, state_diff_min=0.01):
    # 该函数用于专门测试 extract_action_dataset 中数据清洗的效果，红点表示被剔除的数据
    with open(filepath, 'rb') as f:
        buffer_list = pickle.load(f)

    action_mags = []
    state_diffs = []
    is_rejected = []

    print(f">>> 正在分析文件: {filepath} ...")

    # 内部工具函数：根据 agent_id 从状态列表中提取字典，对齐底层结构
    def get_agent_state(state_list, aid):
        for s in state_list:
            if s['id'] == aid:
                return s
        return None

    for transition in buffer_list:
        # 解包 transition
        state_list, obs_dict, action_data, reward, next_state_list, next_obs_dict, done = transition
        
        # 【核心修改 1：兼容单智能体(ndarray)和多智能体(dict)】
        if isinstance(action_data, dict):
            action_items = action_data.items()
        else:
            # 单体训练时，action 是 ndarray，自动提取 agent_id
            agent_id = state_list[0]['id']
            action_items = [(agent_id, action_data)]

        for agent_id, action_vec in action_items:
            # 【核心修改 2：通过 agent_id 精准获取物理状态字典】
            s_curr_dict = get_agent_state(state_list, agent_id)
            s_next_dict = get_agent_state(next_state_list, agent_id)
            
            if s_curr_dict is None or s_next_dict is None:
                continue
                
            # 【核心修改 3：截取物理引擎实际执行的单步 5 维动作】
            executed_action = action_vec[:5]
            a_mag = np.linalg.norm(executed_action)
            
            # 计算位移模长
            s_curr = np.array([
                s_curr_dict['position'][0],
                s_curr_dict['position'][1],
                s_curr_dict['angle']
            ])
            s_next = np.array([
                s_next_dict['position'][0],
                s_next_dict['position'][1],
                s_next_dict['angle']
            ])
            s_diff = np.linalg.norm(s_next - s_curr)

            action_mags.append(a_mag)
            state_diffs.append(s_diff)
            
            # 判定是否会被过滤 (逻辑同步 extract_action_dataset)
            rejected = (a_mag > action_threshold and s_diff < state_diff_min)
            is_rejected.append(rejected)

    if not action_mags:
        print(">>> 警告：未提取到有效数据，无法绘图。")
        return

    # 转换为 numpy 以便绘图
    action_mags = np.array(action_mags)
    state_diffs = np.array(state_diffs)
    is_rejected = np.array(is_rejected)

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))
    
    # 1. 散点图：展示动作与位移的关系
    plt.scatter(action_mags[~is_rejected], state_diffs[~is_rejected], 
                c='blue', alpha=0.5, s=10, label='Accepted (Normal)')
    plt.scatter(action_mags[is_rejected], state_diffs[is_rejected], 
                c='red', alpha=0.8, s=20, marker='x', label='Rejected (Inconsistent)')

    # 绘制阈值线
    plt.axvline(x=action_threshold, color='green', linestyle='--', label='Action Threshold')
    plt.axhline(y=state_diff_min, color='orange', linestyle='--', label='Min State Diff')
    
    # 填充拒绝区域（右下角：高动作模长，低物理位移）
    max_action_mag = max(action_mags) if len(action_mags) > 0 else action_threshold * 2
    plt.fill_between([action_threshold, max_action_mag * 1.1], 0, state_diff_min, 
                     color='red', alpha=0.1, label='Rejected Zone')

    plt.xlabel('Action Magnitude (Input Signal Strength)')
    plt.ylabel('State Displacement (Physical Response)')
    plt.title(f'Consistency Check Analysis\nRejected: {np.sum(is_rejected)} / Total: {len(is_rejected)}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# # 使用示例
# visualize_filter_thresholds(f"sim_replay/{np.random.randint(0,100)}.pkl", action_threshold=0.1, state_diff_min=0.001)