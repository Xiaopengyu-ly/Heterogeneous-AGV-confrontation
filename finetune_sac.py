# finetune_sac.py
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 直接导入你原有的初始化工具
from RL_train.train_initialize import train_initialize
from generate_config import generate_config

def make_env(i):
    def _init():
        # 1. 生成每个子进程独立的配置文件
        generate_config(i)
        # 2. 调用专门针对 RL 训练的初始化函数，返回 RLEnvAdapter
        env = train_initialize(i)
        return env
    return _init

def online_finetune():
    num_envs = 12 # 保持你习惯的 12 个并行环境
    print(f"🧠 启动 SPiRL 多进程微调（{num_envs} 个并行环境）...")
    
    # 构建并行环境
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecMonitor(env, filename="./logs/finetune_monitor.csv")

    # 确定 BC 模型路径
    model_path = "sac_policy_bc"
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 BC 权重文件: {model_path}")

    print(f"🔄 加载行为克隆模型 '{model_path}' 进行 Offline-to-Online 微调...")
    
    # ==========================================
    # ★ SPiRL 核心：覆盖原有的 SAC 超参数 ★
    # ==========================================
    custom_objects = {
        # 1. 降低学习率，保护 BC 注入的“肌肉记忆”，只微调纠偏能力
        "learning_rate": 3e-5,  
        # 2. 重新开启探索（BC 模型通常 std 极小，必须给它试错的空间）
        "ent_coef": "auto_0.1", 
    }
    
    # 加载模型并替换为新的环境与超参
    model = SAC.load(model_path, env=env, custom_objects=custom_objects)
    
    # 清空 Replay Buffer，防止之前 BC 或旧的回放数据干扰真实的 online RL
    model.replay_buffer.reset() 
    
    # 设置定时保存机制
    os.makedirs("./finetuned_models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./finetuned_models/",
        name_prefix="sac_finetuned"
    )

    print(">>> 🚀 开始在真实物理反馈中注入“韧性”...")
    
    try:
        # 微调训练轮数不需要从头训练那么多
        model.learn(
            total_timesteps=300_000, 
            callback=checkpoint_callback,
            progress_bar=True,
            log_interval=10,
            reset_num_timesteps=False # 保持步数累计
        )
        
        # 最终保存为 spirl 模型
        model.save("sac_policy_spirl")
        print("💾 微调彻底完成！模型已保存为 sac_policy_spirl.zip")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被手动中断，正在保存当前进度...")
        model.save("sac_policy_spirl_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    online_finetune()