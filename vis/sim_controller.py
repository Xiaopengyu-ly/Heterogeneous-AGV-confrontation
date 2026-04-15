# vis/sim_controller.py
from vis.replay_buffer import ReplayBuffer
import numpy as np
import os
import torch
from agent.planning.latent_mpc_search import latent_mpc_search


class SimulationController:
    def __init__(self, simulation, config=None):
        self.env = simulation
        # 兼容处理：如果是 Adapter，取出内部 Engine；如果是 Engine，直接用
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
        
        # 底层 RL 模型 (现在它是由 BC 注入后的 SAC)
        lower_model = self.config.get("lower_actor")
        self.lower_model = lower_model if lower_model is not None else None

        # =========================================================
        # ★ Latent MPC 核心组件初始化 ★
        # =========================================================
        self.use_latent_mpc = self.config.get("use_latent_mpc", True)
        if self.use_latent_mpc:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_skills = 16
            self.horizon = 3
            self.POS_LIMIT = 500.0
            self.ANG_LIMIT = np.pi

            print(">>> 正在初始化 Latent MPC 组件 (VQ-VAE + Forward Model)...")
            
            # 1. 加载 VQ-VAE 提取技能码本
            from models.vqvae.VQVAE_skill_generate import SoftVQVAE
            vqvae_path = self.config.get("vqvae_path", "vqvae_skills.pth")
            self.vq_model = SoftVQVAE(seq_len=3, action_dim=5, latent_dim=4, num_skills=self.num_skills).to(self.device)
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
            from models.predictors.agent_dyn_predictor import ForwardPredictor 
            forward_path = self.config.get("forward_path", "forward_model.pth")
            self.forward_model = ForwardPredictor(horizon=self.horizon).to(self.device)
            if os.path.exists(forward_path):
                self.forward_model.load_state_dict(torch.load(forward_path, map_location=self.device))
            self.forward_model.eval()
            print(">>> Latent MPC 模块已激活！")

        self.buffer_path_template = f"sim_replay/{self.data_id}.pkl"
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
            
            action_dict = {}
            if self.lower_model is not None:
                # --- 底层模型决策 ---
                for agent_id, obs in obs_dict.items():
                    
                    # =============================================
                    # ★ 执行 Latent MPC，将高阶意图注入给底层 SAC ★
                    # =============================================
                    if self.use_latent_mpc:
                        best_skill = latent_mpc_search(self, obs)
                        # 覆盖 semantic 预留位，实现“意识夺舍”
                        obs['semantic'] = best_skill

                    if callable(getattr(self.lower_model, "compute_single_action", None)):
                        action = self.lower_model.compute_single_action(observation=obs, explore=False)
                    elif callable(getattr(self.lower_model, "predict", None)):
                        action, _states = self.lower_model.predict(obs, deterministic=True)
                        # SAC 输出是15维(Action Chunking)，我们只截取第一步的 5 维用于物理执行
                        action = action[:5] 
                    else:
                        raise ValueError("Unknown lower_model type.")
                    action_dict[agent_id] = action
            else:
                for agent_id in obs_dict.keys():
                    action_dict[agent_id] = None
            
            # 1. 环境步进
            next_obs_dict, rewards, dones, truncated, info = self.env.step(action_dict)
            
            # 2. 获取当前物理状态用于存储
            current_phys_state = self.engine._get_agent_data_struct()
            all_done = all(dones.values()) if dones else False

            # 3. 将实时数据推送到 Buffer
            buffer_action = action_dict if action_dict else np.zeros(5, dtype=np.float32)
            
            self.replay_buffer.push(
                state=self.prev_phys_state,   
                obs = self.obs,     
                action=buffer_action, 
                reward=rewards,
                next_state=current_phys_state, 
                next_obs= next_obs_dict,
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