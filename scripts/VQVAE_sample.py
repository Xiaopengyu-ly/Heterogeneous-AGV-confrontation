import numpy as np
import os
import glob
import torch
from stable_baselines3 import SAC, PPO
from sim.sim_initialize import sim_initialize
from generate.generate_config import generate_agent_config
from sim.replay_buffer import ReplayBuffer
from sim.train_sim_core import RLEnvAdapter  # 【修改1】引入环境适配器
from sim.sim_controller import SimulationController

def sampler(sample_num : int = 10, policy_path : str = "models/policies/sac_policy", rb_num : list = [1,0], obs_dense = [30,0.5], use_latent_mpc = False):
    model = SAC.load(policy_path, device='cpu')
    for iter in range(sample_num):
        generate_agent_config(iter, rb_num, obs_dense, use_latent_mpc)
        env = sim_initialize(iter)
        max_steps = 2000
        config = {
            "case": "sim_onceonly", 
            "max_steps": max_steps,
            "data_id": iter,
            "buffer_capacity": max_steps,
            "lower_actor": model,
            "use_latent_mpc" : use_latent_mpc  # 关键设置，用于技能提取不需要开启latent mpc
        }
        sim_controller = SimulationController(env, config)
        while True:
            continue_flag = sim_controller.step()
            if not continue_flag:
                print(f"采样第 {iter} 轮结束")
                break

def data_processer_for_VQVAE(slice_len : int = 10):
    files_pattern = "*.pkl"
    source_dir = "sim/sim_replay"
    action_dataset_path = "dataset/action_dataset.npy"
    os.makedirs(os.path.dirname(action_dataset_path), exist_ok=True)
    
    action_dataset_list = []
    buffer = ReplayBuffer(capacity=2000)
    
    if os.path.exists(source_dir):
        files = glob.glob(os.path.join(source_dir, files_pattern))
        print(f">>> 找到 {len(files)} 个数据文件")
        
        for f in files:
            action_dump = buffer.extract_action_dataset(f , slice_len)
            if action_dump is not None and len(action_dump) > 0:
                action_dataset_list.append(action_dump)
        
        if len(action_dataset_list) > 0:
            final_action_dataset = np.concatenate(action_dataset_list, axis=0)
            print(f">>> 最终数据集形状: {final_action_dataset.shape}")
            np.save(action_dataset_path, final_action_dataset)
            print(f">>> 数据集已保存至: {action_dataset_path}")
            return final_action_dataset
        else:
            print(">>> 错误：未提取到任何有效样本，未保存文件。")

def data_processer_for_TwoTower(skill_len : int = 10, horizon_len : int = 10):
    """
    生成 Forward Model 训练数据集。
    要求: skill_len == horizon_len (动作编码长度 = 预测视界)

    产出:
      dataset/dynamics_dataset_obs.npy         — (N, 256)  单帧观测
      dataset/dynamics_dataset_skills.npy      — (N, 5)    VQ-VAE skill 编码
      dataset/dynamics_dataset_trajectorys.npy — (N, T, 38) T 帧增量
    """
    from models.vqvae.VQVAE_skill_generate import SoftVQVAE
    assert skill_len == horizon_len, \
        f"skill_len({skill_len}) must equal horizon_len({horizon_len})"

    files_pattern = "*.pkl"
    source_dir = "sim/sim_replay"
    dataset_path = "dataset/dynamics_dataset"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    T = skill_len
    CONFIG = {
        'T': T,
        'latent_dim': 4,
        'num_skills': 16,
        'model_path_ae': 'models/vqvae/vqvae_skills.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")

    obs_dataset, skills_dataset, deltas_dataset = [], [], []
    buffer = ReplayBuffer(capacity=2000)

    vq_model = SoftVQVAE(
        seq_len=T, action_dim=5,
        latent_dim=CONFIG['latent_dim'], num_skills=CONFIG['num_skills']
    ).to(device)

    if os.path.exists(CONFIG['model_path_ae']):
        print(">>> 加载现有 VQ-VAE...")
        vq_model.load_state_dict(torch.load(CONFIG['model_path_ae'], map_location=device))
    else:
        print(">>> 暂无可用 VQ-VAE 模型")
        return False

    if os.path.exists(source_dir):
        files = glob.glob(os.path.join(source_dir, files_pattern))
        print(f">>> 找到 {len(files)} 个数据文件 (T={T})")

        for f in files:
            dump = buffer.extract_dynamics_dataset(vq_model, T, f, T)
            if dump is not None and len(dump) > 0:
                curr_obs, skills, deltas = dump
                obs_dataset.append(curr_obs)
                skills_dataset.append(skills)
                deltas_dataset.append(deltas)

        if len(skills_dataset) > 0:
            obs_dataset = np.concatenate(obs_dataset, axis=0)
            skills_dataset = np.concatenate(skills_dataset, axis=0)
            deltas_dataset = np.concatenate(deltas_dataset, axis=0)

            print(f">>> 最终数据集: Obs {obs_dataset.shape}, "
                  f"Skill {skills_dataset.shape}, Deltas {deltas_dataset.shape}")
            np.save(f"{dataset_path}_obs.npy", obs_dataset)
            np.save(f"{dataset_path}_skills.npy", skills_dataset)
            np.save(f"{dataset_path}_trajectorys.npy", deltas_dataset)
            print(f">>> 数据集已保存至: {dataset_path}_*")
            return obs_dataset
        else:
            print(">>> 错误：未提取到任何有效样本。")

if __name__ == "__main__":
    # # 1. 仿真数据采样
    # sampler()

    # # 2. VQVAE数据预处理
    # data_processer_for_VQVAE()

    # 3. 双塔正向预测模型数据预处理 
    data_processer_for_TwoTower()