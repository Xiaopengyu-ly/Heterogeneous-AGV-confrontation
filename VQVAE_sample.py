import numpy as np
import os
import glob
import torch
from stable_baselines3 import SAC, PPO
from sim.sim_initialize import sim_initialize
from generate_config import generate_config
from vis.replay_buffer import ReplayBuffer
from RL_train.train_sim_core_lower import RLEnvAdapter  # 【修改1】引入环境适配器
from vis.sim_controller import SimulationController

def sampler():
    sample_iter = 200
    model = SAC.load("sac_policy_spirl", device='cpu')
    for iter in range(sample_iter):
        generate_config(iter)
        env = sim_initialize(iter)
        max_steps = 2000
        config = {
            "case": "sim_onceonly", 
            "max_steps": max_steps,
            "data_id": iter,
            "buffer_capacity": max_steps,
            "lower_actor": model,
            "use_latent_mpc" : True
        }
        sim_controller = SimulationController(env, config)
        while True:
            continue_flag = sim_controller.step()
            if not continue_flag:
                print(f"采样第 {iter} 轮结束")
                break

def data_processer_for_VQVAE():
    files_pattern = "*.pkl"
    source_dir = "sim_replay"
    action_dataset_path = "dataset/action_dataset.npy"
    os.makedirs(os.path.dirname(action_dataset_path), exist_ok=True)
    
    action_dataset_list = []
    buffer = ReplayBuffer(capacity=2000)
    
    if os.path.exists(source_dir):
        files = glob.glob(os.path.join(source_dir, files_pattern))
        print(f">>> 找到 {len(files)} 个数据文件")
        
        for f in files:
            action_dump = buffer.extract_action_dataset(f)
            if action_dump is not None and len(action_dump) > 0:
                action_dataset_list.append(action_dump)
        
        if len(action_dataset_list) > 0:
            final_action_dataset = np.concatenate(action_dataset_list, axis=0)
            print(f">>> 最终数据集形状: {final_action_dataset.shape}")
            np.save(action_dataset_path, final_action_dataset)
            print(f">>> 数据集已保存至: {action_dataset_path}")
        else:
            print(">>> 错误：未提取到任何有效样本，未保存文件。")

def data_processer_for_TwoTower():
    from VQVAE_skill_generate import SoftVQVAE

    files_pattern = "*.pkl"
    source_dir = "sim_replay"
    dataset_path = "dataset/dynamics_dataset"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # 【修改5】关键配置：将技能长度对齐到底层 Action Chunking 的 Horizon 长度
    CONFIG = {
        'T': 5,             # 技能长度 (Horizon = 3)
        'H': 10,            # 预测视界 (未来 10 步的轨迹)
        'latent_dim': 4,
        'num_skills': 16,
        'model_path_ae': 'vqvae_skills.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")

    curr_obs_dataset, skills_dataset = [], []
    action_dataset, future_trajectory_dataset = [], []

    buffer = ReplayBuffer(capacity=2000)

    vq_model = SoftVQVAE(
        seq_len=CONFIG['T'], 
        action_dim = 5,      # 拆分后的单步动作维度
        latent_dim=CONFIG['latent_dim'], 
        num_skills=CONFIG['num_skills']
    ).to(device)

    if os.path.exists(CONFIG['model_path_ae']):
        print(">>> 加载现有 VQ-VAE...")
        vq_model.load_state_dict(torch.load(CONFIG['model_path_ae'], map_location=device))
    else:
        print(">>> 暂无可用 VQ-VAE 模型")
        return False
    
    if os.path.exists(source_dir):
        files = glob.glob(os.path.join(source_dir, files_pattern))
        print(f">>> 找到 {len(files)} 个数据文件")
        
        for f in files:
            dump = buffer.extract_dynamics_dataset(vq_model, CONFIG['H'], f, CONFIG['T'])
            if dump is not None and len(dump) > 0 :
                curr_obs, skills, actions, future_trajectory = dump
                curr_obs_dataset.append(curr_obs)
                skills_dataset.append(skills)
                action_dataset.append(actions)
                future_trajectory_dataset.append(future_trajectory)

        if len(skills_dataset) > 0:
            curr_obs_dataset = np.concatenate(curr_obs_dataset, axis=0)
            skills_dataset = np.concatenate(skills_dataset, axis=0)
            action_dataset = np.concatenate(action_dataset, axis=0)
            future_trajectory_dataset = np.concatenate(future_trajectory_dataset, axis=0)
            
            print(f">>> 最终数据集形状: Obs {curr_obs_dataset.shape}, Skill {skills_dataset.shape}, Actions {action_dataset.shape}, Trajectory {future_trajectory_dataset.shape}")
            np.save(f"{dataset_path}_obs.npy", curr_obs_dataset)
            np.save(f"{dataset_path}_skills.npy", skills_dataset)
            np.save(f"{dataset_path}_actions.npy", action_dataset)
            np.save(f"{dataset_path}_trajectorys.npy", future_trajectory_dataset)
            print(f">>> 数据集已保存至: {dataset_path} ")
        else:
            print(">>> 错误：未提取到任何有效样本。")

if __name__ == "__main__":
    # # 1. 仿真数据采样
    # sampler()

    # # 2. VQVAE数据预处理
    # data_processer_for_VQVAE()

    # 3. 双塔正向预测模型数据预处理 
    data_processer_for_TwoTower()