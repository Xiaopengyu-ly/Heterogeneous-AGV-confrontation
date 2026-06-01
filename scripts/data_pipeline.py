import numpy as np
import os
import glob
from stable_baselines3 import SAC
from sim.sim_initialize import sim_initialize
from generate.generate_config import generate_agent_config
from sim.replay_buffer import ReplayBuffer
from sim.sim_controller import SimulationController


def sample_sac_rollouts(sample_num: int = 10, policy_path: str = "models/policies/sac_policy",
                         rb_num: list = [1, 0], obs_dense: list = [30, 0.5],
                         use_latent_mppi: bool = False):
    """Run SAC policy rollouts and save replay buffers to disk."""
    model = SAC.load(policy_path, device='cpu')
    for iteration in range(sample_num):
        generate_agent_config(iteration, rb_num, obs_dense, use_latent_mppi)
        env = sim_initialize(iteration)
        max_steps = 2000
        config = {
            "case": "sim_onceonly",
            "max_steps": max_steps,
            "data_id": iteration,
            "buffer_capacity": max_steps,
            "lower_actor": model,
            "use_latent_mppi": use_latent_mppi
        }
        sim_controller = SimulationController(env, config)
        while True:
            continue_flag = sim_controller.step()
            if not continue_flag:
                print(f"采样第 {iteration} 轮结束")
                break


def build_forward_model_dataset(horizon_len: int = 10, n_frames: int = 3):
    """
    从 replay buffer 提取 Forward Model 训练数据集。

    产出:
      dataset/dynamics_dataset_obs.npy       — (N, n_frames, 36, 6)  多帧观测
      dataset/dynamics_dataset_dynamics.npy  — (N, 2)                硬件参数
      dataset/dynamics_dataset_actions.npy   — (N, T, 5)             原生 action 序列
      dataset/dynamics_dataset_trajectorys.npy — (N, T, 36, 6)      统一帧增量
    """
    files_pattern = "*.pkl"
    source_dir = "sim/sim_replay"
    dataset_path = "dataset/dynamics_dataset"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    T = horizon_len

    obs_dataset, dyn_dataset, act_dataset, deltas_dataset = [], [], [], []
    buffer = ReplayBuffer(capacity=2000)

    if os.path.exists(source_dir):
        files = glob.glob(os.path.join(source_dir, files_pattern))
        print(f">>> 找到 {len(files)} 个数据文件 (T={T}, n_frames={n_frames})")

        for f in files:
            dump = buffer.extract_dynamics_dataset(T, f, n_frames)
            if dump is not None and len(dump) > 0:
                curr_obs, curr_dyn, curr_act, curr_deltas = dump
                obs_dataset.append(curr_obs)
                dyn_dataset.append(curr_dyn)
                act_dataset.append(curr_act)
                deltas_dataset.append(curr_deltas)

        if len(act_dataset) > 0:
            obs_dataset = np.concatenate(obs_dataset, axis=0)
            dyn_dataset = np.concatenate(dyn_dataset, axis=0)
            act_dataset = np.concatenate(act_dataset, axis=0)
            deltas_dataset = np.concatenate(deltas_dataset, axis=0)

            print(f">>> 最终数据集: Obs {obs_dataset.shape}, "
                  f"Actions {act_dataset.shape}, Dynamics {dyn_dataset.shape}, "
                  f"Deltas {deltas_dataset.shape}")
            np.save(f"{dataset_path}_obs.npy", obs_dataset)
            np.save(f"{dataset_path}_dynamics.npy", dyn_dataset)
            np.save(f"{dataset_path}_actions.npy", act_dataset)
            np.save(f"{dataset_path}_trajectorys.npy", deltas_dataset)
            print(f">>> 数据集已保存至: {dataset_path}_*")
            return obs_dataset
        else:
            print(">>> 错误：未提取到任何有效样本。")


if __name__ == "__main__":
    build_forward_model_dataset()
