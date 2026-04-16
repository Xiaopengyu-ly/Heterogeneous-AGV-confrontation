# agent/planning/agent_PNC.py
import torch
import numpy as np
import os
from models.vqvae.VQVAE_skill_generate import SoftVQVAE
from models.predictors.agent_dyn_predictor import ForwardPredictor
from agent.planning.latent_mpc_search import LatentMPCPlanner

class AgentPNC:
    """
    智能体规划与控制中心 (Planning and Control)
    解耦了所有与模型加载、策略推理相关的逻辑。
    """
    def __init__(self, config):
        self.config = config
        self.lower_model = config.get("lower_actor")
        self.use_latent_mpc = config.get("use_latent_mpc", True)
        
        # 初始化 Latent MPC 相关的深度学习组件
        if self.use_latent_mpc:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_skills = 16
            self.horizon = 10
            self.POS_LIMIT = 500.0
            self.ANG_LIMIT = np.pi

            print(">>> [AgentPNC] 正在初始化 Latent MPC 组件 (VQ-VAE + Forward Model)...")
            
            # 1. 加载 VQ-VAE 提取技能码本
            vqvae_path = self.config.get("vqvae_path", "vqvae_skills.pth")
            self.vq_model = SoftVQVAE(seq_len=10, action_dim=5, latent_dim=4, num_skills=self.num_skills).to(self.device)
            if os.path.exists(vqvae_path):
                self.vq_model.load_state_dict(torch.load(vqvae_path, map_location=self.device))
            self.vq_model.eval()

            with torch.no_grad():
                # 映射码本到 [0, 1] 防止越界警告
                codebook = self.vq_model.vq.embedding.weight.data
                codebook = (codebook + 1.0) / 2.0 
                ids = torch.arange(self.num_skills).float().to(self.device).unsqueeze(1) / self.num_skills
                self.skill_vecs = torch.cat([codebook, ids], dim=1) 

            # 2. 加载双塔动力学预测模型
            forward_path = self.config.get("forward_path", "forward_model.pth")
            self.forward_model = ForwardPredictor(horizon=self.horizon).to(self.device)
            if os.path.exists(forward_path):
                self.forward_model.load_state_dict(torch.load(forward_path, map_location=self.device))
            self.forward_model.eval()
            
            # 3. 实例化重构后的 MPC Planner
            self.mpc_planner = LatentMPCPlanner(
                forward_model=self.forward_model,
                skill_vecs=self.skill_vecs,
                device=self.device,
                num_skills=self.num_skills,
                pos_limit=self.POS_LIMIT,
                ang_limit=self.ANG_LIMIT
            )
            print(">>> [AgentPNC] Latent MPC 模块已激活！")

    def compute_actions(self, obs_dict):
        """
        根据观测数据，执行推理并返回所有智能体的动作字典。
        """
        action_dict = {}
        if self.lower_model is not None:
            for agent_id, obs in obs_dict.items():
                # --- Latent MPC 寻优并注入高阶意图 ---
                if self.use_latent_mpc:
                    best_skill = self.mpc_planner.search_best_skill(obs)
                    obs['semantic'] = best_skill  # 覆盖 semantic 预留位

                # --- 底层模型决策 ---
                if callable(getattr(self.lower_model, "compute_single_action", None)):
                    action = self.lower_model.compute_single_action(observation=obs, explore=False)
                elif callable(getattr(self.lower_model, "predict", None)):
                    action, _states = self.lower_model.predict(obs, deterministic=True)
                    action = action[:5] # 截取前 5 维用于物理执行
                else:
                    raise ValueError("Unknown lower_model type.")
                    
                action_dict[agent_id] = action
        else:
            # 如果没有加载模型，默认返回 None
            for agent_id in obs_dict.keys():
                action_dict[agent_id] = None
                
        return action_dict