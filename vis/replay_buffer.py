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
    def save_buffer(self, filepath: str = "sim_replay/0.pkl"):
        # 将 deque 转为 list（pickle 可以处理，但 list 更通用）
        buffer_list = list(self.buffer)
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_list, f)
        print(f"Replay buffer saved to {filepath} with {len(buffer_list)} transitions.")
    def read_buffer(self, filepath: str = "sim_replay/0.pkl"):
        with open(filepath, 'rb') as f:
            buffer_list = pickle.load(f)
        self.buffer = deque(buffer_list, maxlen=self.buffer.maxlen)
        print(f"Replay buffer loaded from {filepath} with {len(self.buffer)} transitions.")


    def extract_action_dataset(self, filepath: str = "sim_replay/0.pkl", slice_len: int = 5):
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


    def extract_dynamics_dataset(self, vq_model, Horizen_len: int, filepath: str = "sim_replay/0.pkl", seq_len: int = 3):
        import torch
        import pickle
        import numpy as np
        
        vq_model.eval()
        device = next(vq_model.parameters()).device
        
        POS_LIMIT = 500.0  
        ANG_LIMIT = np.pi  
        SKILL_NUM = 16

        def normalize_obs_dict(obs_d):
            """全面解析强化学习的 68 维 Dict 观测空间"""
            norm_lidar = np.array(obs_d['lidar'], dtype=np.float32)
            norm_semantic = np.array(obs_d['semantic'], dtype=np.float32)
            norm_prev_act = np.array(obs_d['prev_actions'], dtype=np.float32)
            
            norm_phy = np.zeros(3, dtype=np.float32)
            norm_phy[0] = obs_d['rel_goal'][0] / POS_LIMIT
            norm_phy[1] = obs_d['rel_goal'][1] / POS_LIMIT
            norm_phy[2] = obs_d['rel_goal'][2] / ANG_LIMIT
            
            norm_hg = np.zeros(9, dtype=np.float32)
            for i in range(3):
                norm_hg[i*3] = obs_d['history_goal'][i*3] / POS_LIMIT
                norm_hg[i*3+1] = obs_d['history_goal'][i*3+1] / POS_LIMIT
                norm_hg[i*3+2] = obs_d['history_goal'][i*3+2] / ANG_LIMIT
                
            return np.concatenate([norm_lidar, norm_phy, norm_semantic, norm_prev_act, norm_hg])

        with open(filepath, 'rb') as f:
            buffer_list = pickle.load(f)

        dataset_curr_obs = []   
        dataset_skills = []   
        dataset_actions = []
        dataset_future_obs = []  
        agent_caches = {} 

        print(f">>> 开始解析动力学数据: {filepath}")
        for transition in buffer_list:
            state, obs, action_data, reward, next_state, next_obs, done = transition
            if done:
                agent_caches.clear()
                continue
                
            # 【核心修复：兼容单智能体与多智能体，处理第一帧 reset 后未包裹的观测】
            if isinstance(action_data, dict):
                action_items = action_data.items()
                # 检查 obs 顶层键是否是 'lidar'，如果是，说明它是未包裹的单体观测（如第一帧）
                if 'lidar' in obs:
                    obs_dict = {aid: obs for aid in action_data.keys()}
                else:
                    obs_dict = obs
            else:
                agent_id = state[0]['id']
                action_items = [(agent_id, action_data)]
                # 若为单体，obs没有最外层的 agent_id 键，人工包上一层
                obs_dict = {agent_id: obs}

            for agent_id, action_vec in action_items:
                if agent_id not in agent_caches:
                    agent_caches[agent_id] = {'obs': [], 'actions': []}
                
                agent_caches[agent_id]['obs'].append(obs_dict[agent_id])
                # 只存入物理执行的 5 维动作
                agent_caches[agent_id]['actions'].append(action_vec[:5])

                if len(agent_caches[agent_id]['obs']) > Horizen_len:
                    # --- A. 获取 Skill ID ---
                    # 仅取前 seq_len (即 3步) 给 VQ-VAE 推理，防止维度崩溃
                    act_seq_for_vq = np.array(agent_caches[agent_id]['actions'][:seq_len], dtype=np.float32)
                    act_tensor = torch.from_numpy(act_seq_for_vq).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        z = vq_model.enc(act_tensor.view(1, -1))
                        _, _, indices = vq_model(act_tensor)
                        skill_id = indices.item()
                        
                    # 【核心修复】：将 VQ-VAE 的 [-1, 1] 潜变量线性映射到 [0, 1]
                    skill_z_np = z.cpu().numpy().flatten()
                    skill_z_np = (skill_z_np + 1.0) / 2.0  
                    skill_z_np = skill_z_np.tolist()
                    
                    # 拼接的 ID 本身就是 [0, 1] 范围的 (skill_id / SKILL_NUM)
                    skill_z_np.append(skill_id / SKILL_NUM)
                    
                    # 取出整个预测视界内的动作序列作为标签
                    full_act_seq = np.array(agent_caches[agent_id]['actions'][:Horizen_len], dtype=np.float32)

                    # --- B. 准备输入 (第 0 帧的全维度状态) ---
                    state_input = normalize_obs_dict(agent_caches[agent_id]['obs'][0])

                    # --- C. 准备目标标签 (未来残差序列) ---
                    future_sequence = []
                    for i in range(1, Horizen_len + 1):
                        f_state = normalize_obs_dict(agent_caches[agent_id]['obs'][i])
                        future_sequence.append(f_state - state_input)

                    dataset_curr_obs.append(state_input)
                    dataset_skills.append(skill_z_np)
                    dataset_actions.append(full_act_seq)
                    dataset_future_obs.append(np.array(future_sequence))

                    # --- E. 滑动窗口 ---
                    agent_caches[agent_id]['obs'].pop(0)
                    agent_caches[agent_id]['actions'].pop(0)

        if len(dataset_curr_obs) == 0:
            return None

        final_obs = np.array(dataset_curr_obs, dtype=np.float32)
        final_skills = np.array(dataset_skills, dtype=np.float32)
        final_actions = np.array(dataset_actions, dtype=np.float32)
        final_future_obs = np.array(dataset_future_obs, dtype=np.float32)

        print(f">>> 提取完成: 物理量纲已缩放, 组合状态维度: {final_obs.shape[1]}")
        return final_obs, final_skills, final_actions, final_future_obs


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