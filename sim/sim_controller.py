# vis/sim_controller.py
from sim.replay_buffer import ReplayBuffer
import numpy as np
from agent.planning.agent_PNC import AgentPNC # 引入新的解耦模块

class SimulationController:
    def __init__(self, simulation, config=None):
        self.env = simulation
        # 兼容处理
        if hasattr(simulation, 'engine'):
            self.engine = simulation.engine
        elif hasattr(simulation, 'sim'):
            self.engine = simulation.sim
        elif hasattr(simulation, 'marl_env'):
            self.engine = simulation.marl_env.engine
            self.env = simulation.marl_env
        else:
            self.engine = simulation
        
        self.config = config or {}
        self.case = self.config.get("case")
        self.max_steps = self.config.get("max_steps", 300)
        self.data_id = self.config.get("data_id", 0)
        
        # ★ 将复杂的模型加载与推理委托给 AgentPNC ★
        self.pnc_controller = AgentPNC(self.config)

        self.buffer_path_template = f"sim/sim_replay/{self.data_id}.pkl"
        self.replay_buffer = ReplayBuffer(capacity=self.config.get("buffer_capacity", 10000))
        
        self.step_count = 0
        self.done = False
        
        if self.case == "replay_sim":
            self.replay_buffer.read_buffer(self._get_buffer_path())
        else:
            # 实时仿真模式
            self.obs, _ = self.env.reset()
            self.prev_phys_state = self.engine._get_agent_data_struct()

    def _get_buffer_path(self):
        return f"sim_replay/{self.data_id}.pkl"

    def should_continue(self):
        if self.case == "replay_sim":
            return self.step_count < len(self.replay_buffer)
        else:
            return not self.done and self.step_count < self.max_steps

    def step(self):
        if not self.should_continue():
            return False

        if self.case == "replay_sim":
            # --- 回放模式 ---
            transition = self.replay_buffer.buffer[self.step_count]
            agents_data_state, agent_obs, action_defalut, _, _, _,_ = transition
            if len(agents_data_state) == len(self.engine.agents):
                for i, a_data in enumerate(agents_data_state):
                    self.engine.agents[i].position = a_data['position']
                    self.engine.agents[i].theta = a_data['angle']
                    self.engine.agents[i].r_point = a_data.get('rpoint')
                    self.engine.agents[i].attk_pos = a_data.get('ATTKpos')
                    self.engine.agents[i].cannon_theta = a_data.get('WPangle')
            self.step_count += 1
            return True

        else: 
            # --- 实时仿真模式 ---
            if self.step_count == 0:
                 active_agents = [a for a in self.engine.agents if not a.disabled]
                 obs_dict = {a.id: self.env.get_agent_observation(a) for a in active_agents}
            else:
                 obs_dict = self.obs 
            
            # 1. 策略推理（委托给 PNC 模块执行）
            action_dict = self.pnc_controller.compute_actions(obs_dict)
            
            # 2. 环境步进
            next_obs_dict, rewards, dones, truncated, info = self.env.step(action_dict)
            current_phys_state = self.engine._get_agent_data_struct()
            all_done = all(dones.values()) if dones else False

            # 3. 将实时数据推送到 Buffer
            buffer_action = action_dict if action_dict else np.zeros(5, dtype=np.float32)
            self.replay_buffer.push(
                state=self.prev_phys_state,   
                obs=self.obs,     
                action=buffer_action, 
                reward=rewards,
                next_state=current_phys_state, 
                next_obs=next_obs_dict,
                done=all_done
            )
            
            # 4. 更新控制器状态
            self.obs = next_obs_dict
            self.prev_phys_state = current_phys_state
            self.step_count += 1
            self.done = all_done

            # 5. 检查结束
            if self.done or self.step_count >= self.max_steps:
                self.replay_buffer.save_buffer(self.buffer_path_template)
                return False
            return True

    def get_info(self):
        render_data = self.engine.get_render_data()
        reward_str = "N/A"
        agents_info = []
        for agent in render_data['agents']:
            pos = agent['position']
            agents_info.append({
                'id': agent['id'],
                'pos': pos,
                'attk': agent.get('ATTKtimes', 0),
                'alive': not agent.get('disabled'),
                'v' : agent.get('v'),
                'w' : agent.get('w'),
            })

        return {
            'step': self.step_count,
            'reward': reward_str,
            'done': self.done,
            'agents': agents_info,
            'render_data': render_data
        }