# agent/planning/agent_PNC.py
import torch
import numpy as np
import os
from abc import ABC, abstractmethod
from models.vqvae.VQVAE_skill_generate import SoftVQVAE
from models.predictors.agent_dyn_predictor import ForwardPredictor
from agent.planning.latent_mpc_search import LatentMPCPlanner

# ==========================================
# 策略适配器 (Strategy / Adapter Pattern)
# ==========================================
class BasePolicyAdapter(ABC):
    @abstractmethod
    def get_action(self, obs: dict) -> np.ndarray:
        pass

class SB3PolicyAdapter(BasePolicyAdapter):
    """适配 Stable Baselines 3 模型 (如 PPO, SAC)"""
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: dict) -> np.ndarray:
        action, _states = self.model.predict(obs, deterministic=True)
        return action[:5]  # 截取物理执行维度

class CustomActorAdapter(BasePolicyAdapter):
    """适配自研 / 自定义接口的模型"""
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: dict) -> np.ndarray:
        return self.model.compute_single_action(observation=obs, explore=False)

def create_policy_adapter(model) -> BasePolicyAdapter:
    """策略工厂：根据模型特征返回对应适配器"""
    if model is None:
        return None
    if callable(getattr(model, "compute_single_action", None)):
        return CustomActorAdapter(model)
    elif callable(getattr(model, "predict", None)):
        return SB3PolicyAdapter(model)
    else:
        raise ValueError(f"无法为该模型创建适配器: {type(model)}")

# ==========================================
# 规划与控制中心
# ==========================================
class AgentPNC:
    def __init__(self, config):
        self.config = config
        # 初始化解耦后的策略适配器
        self.lower_actor_adapter = create_policy_adapter(config.get("lower_actor"))
        self.use_latent_mpc = config.get("use_latent_mpc", True)
        
        if self.use_latent_mpc:
            self.device = torch.device('cpu')
            self.num_skills = 16
            self.horizon = 10
            self.POS_LIMIT = 500.0
            self.ANG_LIMIT = np.pi

            print(">>> [AgentPNC] 正在初始化 Latent MPC 组件 (VQ-VAE + Forward Model)...")
            
            # 1. 加载 VQ-VAE
            vqvae_path = self.config.get("vqvae_path", "vqvae_skills.pth")
            self.vq_model = SoftVQVAE(seq_len=10, action_dim=5, latent_dim=4, num_skills=self.num_skills).to(self.device)
            if os.path.exists(vqvae_path):
                self.vq_model.load_state_dict(torch.load(vqvae_path, map_location=self.device))
            self.vq_model.eval()

            with torch.no_grad():
                codebook = self.vq_model.vq.embedding.weight.data
                codebook = (codebook + 1.0) / 2.0 
                ids = torch.arange(self.num_skills).float().to(self.device).unsqueeze(1) / self.num_skills
                self.skill_vecs = torch.cat([codebook, ids], dim=1) 

            # 2. 加载动力学预测模型
            forward_path = self.config.get("forward_path", "forward_model.pth")
            self.forward_model = ForwardPredictor(horizon=self.horizon).to(self.device)
            if os.path.exists(forward_path):
                self.forward_model.load_state_dict(torch.load(forward_path, map_location=self.device))
            self.forward_model.eval()
            
            # 3. 实例化 MPC Planner
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
        """逻辑主轴：清晰表达每个智能体的意图注入与动作推断"""
        action_dict = {}
        
        if self.lower_actor_adapter is None:
            return {agent_id: None for agent_id in obs_dict.keys()}

        for agent_id, obs in obs_dict.items():
            if self.use_latent_mpc:
                best_skill = self.mpc_planner.search_best_skill(obs)
                # 显式占用五维语义观测空间，专用于向底层控制框架映射 skill-id
                obs['semantic'] = best_skill  

            # 适配器隐蔽了所有内部差异，提供清晰的推理接口
            action_dict[agent_id] = self.lower_actor_adapter.get_action(obs)
                
        return action_dict