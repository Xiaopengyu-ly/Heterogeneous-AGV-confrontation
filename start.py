import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from generate.generate_agents import *
from generate.generate_config import *
from scripts.train_SAC import *
from scripts.data_pipeline import *
from models.predictors.agent_dyn_predictor import *


def main():
    config = {
        "rb_num": [1, 0],
        "obs_dense": [20, 0.5],
        "l_mppi": True,
        "sac_path": "models/policies/sac_policy",
        "sac_train_env_nums": 12,
        "sac_train_steps": 300000,
        "sac_train_iters": 1,
        "test_config_id": 0,
        "sample_num": 500,
        "forward_horizon": 5,         # Forward Model 预测视界
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    device = torch.device(config['device'])
    base_dir = os.path.abspath(".")
    T = config['forward_horizon']

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
    #                       config.get("l_mppi", True))
    # test_and_vis(config.get("sac_path", "models/policies/sac_policy"))

    # # =========================================================================
    # # 阶段 2 — SAC 采样 + Forward Model 数据集生成
    # #   产出: dataset/dynamics_dataset_*.npy 
    # # =========================================================================
    # sample_sac_rollouts(config.get("sample_num", 100),
    #                     config.get("sac_path", "models/policies/sac_policy"),
    #                     config.get("rb_num", [1, 0]),
    #                     config.get("obs_dense", [30, 0.5]),
    #                     False)
    # clean_dir("configmap")
    # build_forward_model_dataset(T)

    # # =========================================================================
    # # 阶段 3 — Forward Model 训练 + 可视化
    # #   产出: models/predictors/forward_model.pth
    # # =========================================================================
    # train_forward_model(T)

    # # 训练后可视化: 随机抽取样本绘制热力图对比
    # from models.predictors.agent_dyn_predictor import plot_forward_predictions

    # forward_model = ForwardPredictor(horizon=T).to(device)
    # forward_model.load_state_dict(torch.load("models/predictors/forward_model.pth", map_location=device))
    # forward_model.eval()

    # obs_data = np.load("dataset/dynamics_dataset_obs.npy")
    # act_data = np.load("dataset/dynamics_dataset_actions.npy")
    # tgt_data = np.load("dataset/dynamics_dataset_trajectorys.npy")
    # dyn_path = "dataset/dynamics_dataset_dynamics.npy"
    # if os.path.exists(dyn_path):
    #     dyn_data = np.load(dyn_path)
    # else:
    #     dyn_data = np.zeros((len(obs_data), 2), dtype=np.float32)

    # for idx in np.random.choice(len(obs_data), size=3, replace=False):
    #     plot_forward_predictions(forward_model, obs_data, dyn_data, act_data, tgt_data,
    #                               horizon=T, sample_idx=idx)
    # plt.show()

    # =========================================================================
    # 阶段 4 — MPPI 部署测试
    #   需要: Forward Model
    # =========================================================================
    # generate_agent_config(config.get("test_config_id", 0),
    #                       config.get("rb_num", [1, 0]),
    #                       config.get("obs_dense", [30, 0.5]),
    #                       config.get("l_mppi", True))
    # test_and_vis(config.get("sac_path", "models/policies/sac_policy"))

    # clean_dir("all")


if __name__ == "__main__":
    main()
