import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from generate.generate_agents import *
from generate.generate_config import *
from scripts.train_SAC import *
from scripts.VQVAE_sample import *
from models.predictors.agent_dyn_predictor import *
from models.vqvae.VQVAE_skill_generate import *

def main():
    config = {
        "rb_num" : [1,0],
        "obs_dense" : [20,0.5],
        "l_mpc" : True,
        "sac_path" : "models/policies/sac_policy",
        "vae_path": "models/vae/action_vae_pretrained.pt",
        "sac_train_env_nums" : 12,
        "sac_train_steps" : 300000,
        "sac_train_iters" : 1,
        "test_config_id"  : 0,
        "sample_num" : 400,
        "vqvae_slice_len" : 3,        # VQ-VAE 技能长度 = Forward Model 预测视界
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(config['device'])
    base_dir = os.path.abspath(".")
    T = config['vqvae_slice_len']

    # # =========================================================================
    # # 阶段 1 — 专家轨迹 SAC 训练
    # #   产出: models/policies/sac_policy.zip
    # # =========================================================================
    # generate_agent_params(profiles=["default"])
    # train_agent(
    #     env_nums=config.get("sac_train_env_nums", 12),
    #     steps=config.get("sac_train_steps", 500_000),
    #     iter=config.get("sac_train_iters", 1),
    #     policy_path=config["sac_path"],
    # )
    # # =========================================================================
    # # 阶段 1.5 — sac 部署测试
    # # =========================================================================
    # generate_agent_config(config.get("test_config_id", 0),
    #                       config.get("rb_num", [1, 0]),
    #                       config.get("obs_dense", [30, 0.5]),
    #                       config.get("l_mpc", True))
    # test_and_vis(config.get("sac_path", "models/policies/sac_policy"))

    # # =========================================================================
    # # 阶段 2 — VQ-VAE 技能发现
    # #   产出: models/vqvae/vqvae_skills.pth
    # # =========================================================================
    # sampler(config.get("sample_num", 100),
    #         config.get("sac_path", "models/policies/sac_policy"),
    #         config.get("rb_num", [1, 0]),
    #         config.get("obs_dense", [30, 0.5]),
    #         config.get("l_mpc", False))
    # data_processer_for_VQVAE(T)
    # clean_dir("configmap")
    # action_data = np.load("dataset/action_dataset.npy")
    # print(f">>> 动作数据集: {action_data.shape}")
    # train_soft_vqvae(action_data, T)

    # # =========================================================================
    # # 阶段 3 — Forward Model 数据集 + 训练 + 可视化
    # #   产出: dataset/dynamics_dataset_*.npy  +  models/predictors/forward_model.pth
    # # =========================================================================
    # data_processer_for_TwoTower(T, T)   # seq_len == horizon_len
    # train_forward_model(T)

    # # 训练后可视化: 随机抽取一个样本绘制热力图对比
    # from models.predictors.agent_dyn_predictor import plot_forward_predictions
    # from models.vqvae.VQVAE_skill_generate import SoftVQVAE
    # from models.vae.action_vae import ActionVAE

    # forward_model = ForwardPredictor(horizon=T).to(device)
    # forward_model.load_state_dict(torch.load("models/predictors/forward_model.pth", map_location=device))
    # forward_model.eval()

    # vq_model = SoftVQVAE(seq_len=T, action_dim=5, latent_dim=4, num_skills=16).to(device)
    # vq_model.load_state_dict(torch.load("models/vqvae/vqvae_skills.pth", map_location=device))
    # vq_model.eval()

    # # 注册 skill → ActionVAE 嵌入 (分析性 history_goal 更新所需)
    # vae_path = "models/vae/action_vae_pretrained.pt"
    # if os.path.exists(vae_path):
    #     vae_ckpt = torch.load(vae_path, map_location=device)
    #     action_vae = ActionVAE().to(device)
    #     if 'model_state_dict' in vae_ckpt:
    #         action_vae.load_state_dict(vae_ckpt['model_state_dict'])
    #     else:
    #         action_vae.load_state_dict(vae_ckpt)
    #     action_vae.eval()
    #     forward_model.register_skill_action_embeddings(vq_model, action_vae)
    #     print(f"  ✓ ActionVAE 已加载并注册 skill→action 嵌入")
    # else:
    #     print(f"  ⚠ ActionVAE 未找到: {vae_path}")

    # obs_data = np.load("dataset/dynamics_dataset_obs.npy")
    # skill_data = np.load("dataset/dynamics_dataset_skills.npy")
    # tgt_data = np.load("dataset/dynamics_dataset_trajectorys.npy")

    # for idx in np.random.choice(len(obs_data), size=3, replace=False):
    #     plot_forward_predictions(forward_model, vq_model, obs_data, skill_data, tgt_data,
    #                               horizon=T, sample_idx=idx)
    # plt.show()

    # # =========================================================================
    # # 阶段 4 — Latent MPC 部署测试
    # #   需要: VQ-VAE + Forward Model
    # # =========================================================================
    generate_agent_config(config.get("test_config_id", 0),
                          config.get("rb_num", [1, 0]),
                          config.get("obs_dense", [30, 0.5]),
                          config.get("l_mpc", True))
    test_and_vis(config.get("sac_path", "models/policies/sac_policy"))

    # clean_dir("all")

if __name__ == "__main__":
    main()
