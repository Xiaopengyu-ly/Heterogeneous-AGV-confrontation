# sim/sim_controller.py
import numpy as np
from sim.replay_buffer import ReplayBuffer


class SimulationController:
    """统一 SAC 仿真控制器 — 支持单层模式 + MAPPO 两层模式 + 回放."""

    def __init__(self, simulation, config=None):
        self.env = simulation
        self.engine = simulation.engine if hasattr(simulation, 'engine') else simulation

        self.config = config or {}
        self.case = self.config.get("case")
        self.max_steps = self.config.get("max_steps", 300)
        self.data_id = self.config.get("data_id", 0)

        self.lower_actor = self.config.get("lower_actor")
        self.mappo_actor = self.config.get("mappo_actor")
        self._mappo_chase_cache = {}   # {agent_id: [Agent]} 存活敌军+友军
        self._mappo_attack_cache = {}  # {agent_id: [Agent]} 存活敌军

        self.buffer_path_template = f"sim/sim_replay/{self.data_id}.pkl"
        self.replay_buffer = ReplayBuffer(capacity=self.config.get("buffer_capacity", 10000))

        self.step_count = 0
        self.done = False

        if self.case != "replay_sim":
            self.obs, _ = self.env.reset()
            self.prev_phys_state = self.engine._get_agent_data_struct()

    # ------------------------------------------------------------------
    def should_continue(self):
        if self.case == "replay_sim":
            return self.step_count < len(self.replay_buffer)
        return not self.done and self.step_count < self.max_steps

    # ------------------------------------------------------------------
    def _get_mappo_obs_and_cache(self):
        """构建 MAPPO 极坐标观测 — MultiDiscrete 版本."""
        obs_dict = {}
        max_enemies = 3
        max_allies = 2
        max_sense_dist = 300.0

        for agent in self.engine.agents:
            if agent.side != 0 or agent.disabled:
                continue

            all_enemies, all_allies = [], []
            alive_enemies, alive_allies = [], []

            for other in self.engine.agents:
                if other.id == agent.id:
                    continue
                dx = other.position[0] - agent.position[0]
                dy = other.position[1] - agent.position[1]
                dist = np.hypot(dx, dy)
                rel_angle = (np.arctan2(dy, dx) - agent.theta + np.pi) % (2 * np.pi) - np.pi

                entry = {"agent": other, "dist": dist, "angle": rel_angle}
                if other.side != agent.side:
                    all_enemies.append(entry)
                    if not other.disabled:
                        alive_enemies.append(entry)
                else:
                    all_allies.append(entry)
                    if not other.disabled:
                        alive_allies.append(entry)

            all_enemies.sort(key=lambda x: x["dist"])
            all_allies.sort(key=lambda x: x["dist"])
            alive_enemies.sort(key=lambda x: x["dist"])
            alive_allies.sort(key=lambda x: x["dist"])

            self._mappo_chase_cache[agent.id] = (
                [e["agent"] for e in alive_enemies] + [a["agent"] for a in alive_allies]
            )
            self._mappo_attack_cache[agent.id] = [e["agent"] for e in alive_enemies]

            obs_array = np.zeros((max_enemies * 3 + max_allies * 3 + 4,), dtype=np.float32)
            idx = 0
            for i in range(max_enemies):
                if i < len(all_enemies):
                    e = all_enemies[i]
                    obs_array[idx:idx+3] = [
                        np.clip(e["dist"] / max_sense_dist, 0., 1.),
                        e["angle"] / np.pi,
                        1.0 if not e["agent"].disabled else -1.0,
                    ]
                else:
                    obs_array[idx:idx+3] = [1.0, 0.0, -1.0]
                idx += 3

            for i in range(max_allies):
                if i < len(all_allies):
                    a = all_allies[i]
                    obs_array[idx:idx+3] = [
                        np.clip(a["dist"] / max_sense_dist, 0., 1.),
                        a["angle"] / np.pi,
                        1.0 if not a["agent"].disabled else -1.0,
                    ]
                else:
                    obs_array[idx:idx+3] = [1.0, 0.0, 1.0]
                idx += 3

            # 自身状态
            obs_array[idx]   = agent.smoke_remain / max(agent.smoke_capacity, 1)
            obs_array[idx+1] = agent.cannon_remain / max(agent.cannon_capacity, 1)
            obs_array[idx+2] = getattr(agent, "v", 0.0) / max(agent.v_max, 1)
            obs_array[idx+3] = 1.0 if not agent.disabled else -1.0

            obs_dict[str(agent.id)] = obs_array
        return obs_dict, (self._mappo_chase_cache, self._mappo_attack_cache)

    # ------------------------------------------------------------------
    def step(self):
        if not self.should_continue():
            return False

        # ---- 回放模式 ----
        if self.case == "replay_sim":
            transition = self.replay_buffer.buffer[self.step_count]
            agents_data_state, _, _, _, _, _, _ = transition
            if len(agents_data_state) == len(self.engine.agents):
                for i, a_data in enumerate(agents_data_state):
                    self.engine.agents[i].position = a_data['position']
                    self.engine.agents[i].theta = a_data['angle']
                    self.engine.agents[i].r_point = a_data.get('rpoint')
                    self.engine.agents[i].attk_pos = a_data.get('ATTKpos')
                    self.engine.agents[i].cannon_theta = a_data.get('WPangle')
            self.step_count += 1
            return True

        # ---- MAPPO 两层模式 ----
        if self.mappo_actor is not None:
            # MAPPO 顶层决策: 每 25 步分配 chase/attack/smoke
            if self.step_count % 25 == 0:
                obs_dict, (chase_cache, attack_cache) = self._get_mappo_obs_and_cache()
                for aid_str, obs in obs_dict.items():
                    action = self.mappo_actor.compute_single_action(
                        obs, policy_id="red_shared_policy"
                    )
                    agent = next((a for a in self.engine.agents if str(a.id) == aid_str), None)
                    if agent is None or agent.disabled:
                        continue

                    chase_idx, attack_idx, smoke_flag = int(action[0]), int(action[1]), int(action[2])

                    # chase: 设置 t_pos
                    chase_list = chase_cache.get(agent.id, [])
                    if chase_idx == 0 or chase_idx > len(chase_list):
                        agent.t_pos = agent.position.copy()
                    else:
                        agent.t_pos = chase_list[chase_idx - 1].position.copy()

                    # attack: 设置 cannon_targets_id
                    attack_list = attack_cache.get(agent.id, [])
                    if attack_idx == 0 or attack_idx > len(attack_list):
                        agent.cannon_targets_id = 0
                    else:
                        target = attack_list[attack_idx - 1]
                        agent.cannon_targets_id = target.id
                        agent.cannon_targets_info["position"][target.id] = target.position.copy()
                        chan = self.engine.channel_dict.get(str(target.id))
                        if chan is not None:
                            agent.cannon_targets_info["channelid"][target.id] = chan

                    # smoke
                    if smoke_flag == 1 and agent.smoke_remain > 0:
                        agent.smoke_mission = True

            # 统一推理 + 物理步进 (MPC 或 SAC)
            action_dict = {}
            for agent in self.engine.agents:
                if agent.disabled:
                    continue
                obs = self.env.get_agent_observation(agent)
                if getattr(agent, 'use_latent_mpc', False):
                    _, ll_action = agent.behavior_system.get_mpc_action(obs)
                elif self.lower_actor is not None:
                    ll_action, _ = self.lower_actor.predict(obs, deterministic=True)
                else:
                    continue
                action_dict[agent.id] = ll_action

            current_phys_state = self.engine._get_agent_data_struct()
            next_obs_dict, rewards, dones, truncated, info = self.env.step(action_dict)

            # 录轨迹 (与单层路径一致)
            all_done = all(dones.values()) if dones else False
            buffer_action = action_dict if action_dict else np.zeros(5, dtype=np.float32)
            self.replay_buffer.push(
                state=self.prev_phys_state,
                obs=self.obs,
                action=buffer_action,
                reward=rewards,
                next_state=current_phys_state,
                next_obs=next_obs_dict,
                done=all_done,
            )

            self.obs = next_obs_dict
            self.prev_phys_state = current_phys_state
            self.step_count += 1

            red_alive = sum(1 for a in self.engine.agents if a.side == 0 and not a.disabled)
            blue_alive = sum(1 for a in self.engine.agents if a.side == 1 and not a.disabled)
            if red_alive == 0 or blue_alive == 0:
                self.done = True

            if self.done or self.step_count >= self.max_steps:
                self.replay_buffer.save_buffer(self.buffer_path_template)
                return False
            return True

        # ---- 单层推理模式 (MPC 或 SAC) ----
        action_dict = {}
        for agent in self.engine.agents:
            if agent.disabled:
                continue
            obs = self.env.get_agent_observation(agent)
            if getattr(agent, 'use_latent_mpc', False):
                _, ll_action = agent.behavior_system.get_mpc_action(obs)
            elif self.lower_actor is not None:
                ll_action, _ = self.lower_actor.predict(obs, deterministic=True)
            else:
                continue
            action_dict[agent.id] = ll_action

        next_obs_dict, rewards, dones, truncated, info = self.env.step(action_dict)

        current_phys_state = self.engine._get_agent_data_struct()
        all_done = all(dones.values()) if dones else False

        buffer_action = action_dict if action_dict else np.zeros(5, dtype=np.float32)
        self.replay_buffer.push(
            state=self.prev_phys_state,
            obs=self.obs,
            action=buffer_action,
            reward=rewards,
            next_state=current_phys_state,
            next_obs=next_obs_dict,
            done=all_done,
        )

        self.obs = next_obs_dict
        self.prev_phys_state = current_phys_state
        self.step_count += 1
        self.done = all_done

        if self.done or self.step_count >= self.max_steps:
            self.replay_buffer.save_buffer(self.buffer_path_template)
            return False
        return True

    # ------------------------------------------------------------------
    def get_info(self):
        render_data = self.engine.get_render_data()
        agents_info = []
        for agent in render_data['agents']:
            agents_info.append({
                'id': agent['id'],
                'pos': agent['position'],
                'attk': agent.get('ATTKtimes', 0),
                'alive': not agent.get('disabled'),
                'v': agent.get('v'),
                'w': agent.get('w'),
            })

        return {
            'step': self.step_count,
            'reward': "N/A",
            'done': self.done,
            'agents': agents_info,
            'render_data': render_data,
        }
