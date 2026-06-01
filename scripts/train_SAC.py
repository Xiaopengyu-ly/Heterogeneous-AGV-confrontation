# train_agent.py
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from PyQt5.QtWidgets import QApplication

from vis.controlled_window import ControlledVisWindow
from sim.sim_initialize import sim_initialize
from sim.train_sim_core import UnifiedFeatureExtractor
from generate.generate_config import generate_agent_config
from generate.generate_agents import generate_agent_params
import sys
import time
import os
import glob
import shutil

def make_env(i):
    def _init():
        generate_agent_params(profiles=["default"])
        generate_agent_config(i)
        sim = sim_initialize(i)
        return sim
    return _init

def train_agent(env_nums : int = 9, steps : int = 100000, iter : int = 1, policy_path : str = "models/policies/sac_policy"):
    # print("✅ 检查单个环境...")
    # sim = sim_initialize("./sim/config.yaml")
    # check_env(sim)
    # print("Action space type:", type(sim.action_space))
    # print("Action space:", sim.action_space)

    # 一般取num_envs <= 15 在计算机性能范围内
    # total_timesteps= 30000 左右能训练得到目标导引能力
    # 在不同地图切换下，训练 iteration 轮
    # 需通过generate_config.py保证训练时环境中只有一个agent（测试时不限制个数）
    iteration = iter
    for iter in range(iteration):
        num_envs = env_nums
        print(f"🧠 启动多进程训练（{num_envs} 个环境）...")
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        env = VecMonitor(env, filename=f"./logs/monitor{iter}.csv")
        # case = 1 if iter == 0 else 1
        case = 0
        # if os.path.exists(policy_path) and case == 1:
        #     model = SAC.load(policy_path, env=env)
        #     print(f"🔄 加载模型 '{policy_path}'，继续训练")
        # else:
        #     case = 0
        try:
            if case == 0:
                print("🆕 从头开始训练（case=0）")
                # 使用 SAC，仅保留 SAC 支持的参数
                model = SAC(
                    "MultiInputPolicy",
                    env,
                    verbose = 1,
                    learning_rate = 3e-4,
                    gamma = 0.99,
                    ent_coef = "auto_0.2",
                    train_freq = 1,
                    gradient_steps = 1,
                    buffer_size = 10000,
                    learning_starts = 1000,
                    policy_kwargs = dict(
                        features_extractor_class = UnifiedFeatureExtractor,
                        features_extractor_kwargs = dict(features_dim=128),
                        net_arch = dict(pi=[256, 256], qf=[256, 256]),
                    ),
                )
                reset_timesteps = True
            else:
                model = SAC.load(policy_path, env=env)
                print(f"🔄 加载模型 '{policy_path}'，继续训练")
                model.replay_buffer.pos = 0
                reset_timesteps = False
            model.learn(
                total_timesteps = steps,   # 至少 total_timesteps 步 ，200000一轮课程比较稳定
                progress_bar = True,
                log_interval = 20000,          # 每 log_interval 步打印一次
                reset_num_timesteps=reset_timesteps
            )
            model.save(policy_path)
            print(f"💾 模型已保存为 {policy_path} ")
            env.close()
            del env
        except:
            env.close()
            del env
            raise(RuntimeError())
        time.sleep(2)

def clean_dir(clean_mode : str = "configmap"):
    if clean_mode == "configmap":
        # 清理子文件夹中的匹配文件
        targets = ["config*", "grid_map*", "d_spl_map*"]
        dirs = ["sim/config_data","sim/map_data"]
        for dir in dirs:
            if os.path.exists(dir):
                for pattern in targets:
                    files = glob.glob(os.path.join(dir, pattern))
                    for f in files:
                        try:
                            os.remove(f)
                            print(f"Deleted: {f}")
                        except OSError as e:
                            print(f"Error deleting {f}: {e}")
    elif clean_mode == "all":
        # 清理子文件夹中的匹配文件
        targets = ["*"]
        dirs = ["sim/config_data","sim/map_data","sim/sim_replay"]
        for dir in dirs:
            if os.path.exists(dir):
                for pattern in targets:
                    files = glob.glob(os.path.join(dir, pattern))
                    for f in files:
                        try:
                            os.remove(f)
                            print(f"Deleted: {f}")
                        except OSError as e:
                            print(f"Error deleting {f}: {e}")

# scripts/train_SAC.py

def test_and_vis(policy_path: str, config_id: int = 0):
    print("🧪 加载统一 SAC 策略并启动可视化 ...")
    env = sim_initialize(config_id)

    lower_model = SAC.load(policy_path)

    app = QApplication(sys.argv)
    max_steps = 2000

    config = {
        "case": "sim_onceonly",
        "max_steps": max_steps,
        "data_id": 0,
        "buffer_capacity": max_steps,
        "lower_actor": lower_model,
    }

    window = ControlledVisWindow(env, config)
    window.show()
    sys.exit(app.exec_())


