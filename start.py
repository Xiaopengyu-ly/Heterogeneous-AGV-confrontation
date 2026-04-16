import os
import numpy as np

from generate import generate_agents,generate_config
from scripts.train_SAC import train_agent, test_and_vis, clean_dir
from scripts.VQVAE_sample import sampler, data_processer_for_VQVAE, data_processer_for_TwoTower
from scripts.behavior_clone import train_aligned_bc, inject_to_sac
from models.policies.finetune_sac import sac_finetune
from models.predictors.agent_dyn_predictor import train_forward_model,visualize_imagined_trajectories
from models.vqvae.VQVAE_skill_generate import train_soft_vqvae,visualize_continuous_interpolation

def main():
    config = {
        "rb_num" : [2,0],
        "obs_dense" : [20,0.5],
        "sac_path_primitive" : "models/policies/sac_policy",
        "sac_path_behavior_clone" : "models/policies/sac_policy_bc",
        "sac_path_finetuned" : "models/policies/sac_policy_finetuned",
        "sac_train_env_nums" : 12,
        "sac_train_steps" : 100000,
        "sac_train_iters" : 1,
        "test_config_id"  : 0,
        "sample_num" : 10,
        "vqvae_slice_len" : 10,
        "predict_horizen" : 10,
    }

    # # 操作流程
    # # 1、 配置智能体参数
    # generate_agents.generate_agent_config()

    # # 2、 训练强化学习策略，分阶段训练
    # # 1) 训练单体SAC -> 
    # # 训练单体SAC时，随机生成的样本，默认"rb_num" = [1,0], "obs_dense" = [30,0.5] (因为SB3仅支持单体训练)
    # train_agent(config.get("sac_train_env_nums",12), config.get("sac_train_steps",10000), config.get("sac_train_iters",1)) 

    # # 2) 批量仿真采样 -> 制作VQVAE技能数据集 —> 训练VQVAE技能提取模型 -> 进行VQVAE技能克隆  -> sac微调
    # sampler(config.get("sample_num",10), config.get("sac_path_primitive","models/policies/sac_policy"),
    #         config.get("rb_num",[1,0]), config.get("obs_dense", [30,0.5]))
    # clean_dir("configmap")
    # action_data = data_processer_for_VQVAE(config.get("vqvae_slice_len",5))
    # print(f">>> 成功加载动作数据集，形状: {action_data.shape}")
    # vqvae_model = train_soft_vqvae(action_data, config.get("vqvae_slice_len",5))
    # visualize_continuous_interpolation(vqvae_model, config.get("vqvae_slice_len",5))
    # train_aligned_bc()
    # inject_to_sac()
    # sac_finetune(config.get("sac_train_env_nums",12), config.get("sac_train_steps",10000), config.get("sac_path_finetuned","models/policies/sac_policy_finetuned"))

    # # 3) 制作正向预测数据集 -> 训练正向预测模型 -> 
    # obs_data = data_processer_for_TwoTower(config.get("vqvae_slice_len",5), config.get("predict_horizen",5))
    # forward_model = train_forward_model(config.get("predict_horizen",5))
    # visualize_imagined_trajectories(forward_model, obs_data[np.random.randint(0, len(obs_data)):][:1], config.get("predict_horizen",5), config.get("vqvae_slice_len",5))
    
    # 4) 训练 latent MPC 参数（基于多智能体MPC代价函数）
    
    
    # # 3、配置用于可视化测试的地图障碍密度、红蓝个体数量
    generate_config.generate_config(config.get("test_config_id",0), config.get("rb_num",[1,0]), config.get("obs_dense", [30,0.5]))
    # 测试时开启latent MPC
    test_and_vis(config.get("sac_path_finetuned","models/policies/sac_policy_spirl"))


if __name__ == "__main__":
    main()
    

