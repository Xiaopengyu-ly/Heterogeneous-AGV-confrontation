# agent/planning/agent_PNC.py
import numpy as np
from abc import ABC, abstractmethod

# 全局 PNC 步数计数器
_PNC_STEP = 0
_PNC_PRINT_INTERVAL = 1  # 与 MPC 诊断对齐

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
        global _PNC_STEP, _PNC_PRINT_INTERVAL
        _PNC_STEP += 1
        do_print = (_PNC_STEP % _PNC_PRINT_INTERVAL == 0)

        action_dict = {}

        if self.lower_actor_adapter is None:
            return {agent_id: None for agent_id in obs_dict.keys()}

        for agent_id, obs in obs_dict.items():
            action = self.lower_actor_adapter.get_action(obs)
            action_dict[agent_id] = action

            if do_print:
                skill_vec = obs.get('semantic', None)
                if skill_vec is not None:
                    skill_id = int(round(skill_vec[-1] * 16))  # semantic=[z0,z1,z2,z3, id/16]
                else:
                    skill_id = -1
                v, w = action[0], action[1]
                act_mag = np.linalg.norm(action[:2])
                # print(f"{'':14s}[ACT] agent={agent_id} skill={skill_id:02d} | v={v:+.2f} w={w:+.2f} mag={act_mag:.3f} | a3={action[2]:.3f} a4={action[3]:.3f} a5={action[4]:.3f}")

        return action_dict