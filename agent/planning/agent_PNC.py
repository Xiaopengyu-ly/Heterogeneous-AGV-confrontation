# agent/planning/agent_PNC.py
import numpy as np
from abc import ABC, abstractmethod

# ==========================================
# 策略适配器 (Strategy / Adapter Pattern) 
# ==========================================
class BasePolicyAdapter(ABC):
    @abstractmethod
    def get_action(self, obs: dict) -> np.ndarray:
        pass

class SB3PolicyAdapter(BasePolicyAdapter):
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: dict) -> np.ndarray:
        action, _states = self.model.predict(obs, deterministic=True)
        return action[:5]

class CustomActorAdapter(BasePolicyAdapter):
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: dict) -> np.ndarray:
        return self.model.compute_single_action(observation=obs, explore=False)

def create_policy_adapter(model) -> BasePolicyAdapter:
    if model is None:
        return None
    if callable(getattr(model, "compute_single_action", None)):
        return CustomActorAdapter(model)
    elif callable(getattr(model, "predict", None)):
        return SB3PolicyAdapter(model)
    else:
        raise ValueError(f"无法为该模型创建适配器: {type(model)}")

# ==========================================
# 纯粹的控制中心 (PNC)
# ==========================================
class AgentPNC:
    def __init__(self, config):
        self.config = config
        self.lower_actor_adapter = create_policy_adapter(config.get("lower_actor"))

    def compute_actions(self, obs_dict):
        """
        此时 MPC 寻优已经由智能体自身的 agent.behavior.task_allocate_model(obs) 完成，
        这里只负责执行底层强化学习策略。
        """
        action_dict = {}
        
        if self.lower_actor_adapter is None:
            return {agent_id: None for agent_id in obs_dict.keys()}

        for agent_id, obs in obs_dict.items():
            # obs 中的 'semantic' 维度已经在调用 PNC 之前，被 task_allocate_model 修改过了
            action_dict[agent_id] = self.lower_actor_adapter.get_action(obs)
                
        return action_dict