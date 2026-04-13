import numpy as np
import random

'''
    [重构版] 状态监测组件
    负责 Check 逻辑
'''
class CheckSystem:
    def __init__(self, agent):
        self.agent = agent

    def check_hit(self):
        # 访问 self.agent.r_point 等
        if self.agent.r_point is not None:
            distance_to_r_point = np.sqrt((self.agent.position[0] - self.agent.r_point[0])**2 + (self.agent.position[1] - self.agent.r_point[1])**2)
            if not self.agent.hit_rpoint and distance_to_r_point <= self.agent.miss_dist:
                self.agent.hit_rpoint = True
                self.agent.r_point = None

    def check_env(self, env_feedback : dict ):
        # 毁伤效果检定
        live_id = env_feedback['live_ids']
        if self.agent.id not in live_id:
            self.agent.disabled = True
            return
        # 侦察/火力打击目标切换
        channel_dict = env_feedback['channel_dict']
        if self.agent.cannon_targets_id not in live_id and not self.agent.cannon_targets_id == 0: 
            self.agent.cannon_targets_id = 0
            self.agent.attk_pos = None
            for id in live_id:
                if not (id == self.agent.id or id == 0 or id in self.agent.neighbors_id):
                    self.agent.cannon_targets_id = id
                    # 新目标由通信获得初始信息
                    self.agent.cannon_targets_info["channelid"][f"{self.agent.cannon_targets_id}"] = channel_dict[f'{self.agent.cannon_targets_id}']
                    # 【注意】这里调用 agent 的接口，agent 会自动转发给 comm_system
                    if self.agent.msg_pool is not None:
                        self.agent.broadcast_msg(self.agent.msg_pool)
                        self.agent.recieve_msg(self.agent.msg_pool)
                    self.agent.send_count = 0
        # 烟雾区域反馈
        self.agent.smoke_zones = env_feedback['smoke_zone']

        # 障碍物相对速度求解
        curr_sector = env_feedback['obs_sector_dict'].get(self.agent.id)
        prev_sector = getattr(self.agent, 'obs_sector', curr_sector) # 防止第一帧没有 prev_sector
        d_phi = 2 * np.pi / self.agent.sector_num
        
        obs_v_sector = []
        obs_vt_sector = []
        
        # 模板匹配参数配置
        search_range = 2  # 搜索范围：在上一帧寻找的偏移量范围 (例如前后各找 2 个扇区)
        window_size = 1   # 模板半宽：1 代表取当前扇区及前后各 1 个，共 3 个扇区构成特征模板
        
        for i in range(self.agent.sector_num):
            if curr_sector[i] < 100 and prev_sector[i] < 100:
                # 1. 径向速度计算 (你原有的逻辑，正代表远离)
                v_r = (curr_sector[i] - prev_sector[i]) / self.agent.dT
                
                # =======================================================
                # 2. 一维特征模板匹配 (Block Matching) 计算切向运动
                # =======================================================
                best_shift = 0
                min_error = float('inf')
                
                # 提取当前帧的局部形状模板 (需处理 360 度环形越界)
                curr_patch = [curr_sector[(i + d) % self.agent.sector_num] for d in range(-window_size, window_size + 1)]
                
                # 在上一帧的邻域内滑动，寻找绝对误差和 (SAD) 最小的位置
                for shift in range(-search_range, search_range + 1):
                    error = 0
                    for d in range(-window_size, window_size + 1):
                        prev_idx = (i + shift + d) % self.agent.sector_num
                        error += abs(curr_patch[d + window_size] - prev_sector[prev_idx])
                    
                    if error < min_error:
                        min_error = error
                        best_shift = shift
                
                # 质量控制：如果最佳匹配误差过大，说明是新出现的障碍物或发生遮挡，拒绝输出切向速度
                match_threshold = 5.0 * (2 * window_size + 1) # 允许平均每个扇区有 5m 的形变误差
                if min_error > match_threshold:
                    v_t_obs = 0.0
                else:
                    # 运动学换算：
                    # 若 shift 为负，说明上一帧障碍物在更小的索引处，障碍物发生了逆时针(正角度)旋转。
                    # 所以角速度 w_obs = -shift * d_phi / dT
                    w_obs = -best_shift * d_phi / self.agent.dT
                    v_t_obs = curr_sector[i] * w_obs
                    
                # =======================================================
                # 3. 剥离双轮差速小车自身旋转，获取环境真实切向速度
                # =======================================================
                prev_w = getattr(self.agent, 'w', 0.0) # 需确保你的 agent 中保存了上一步下发的真实角速度 w
                v_t_env = v_t_obs + curr_sector[i] * prev_w
                
            else:
                v_r = 0.0
                v_t_env = 0.0
                
            obs_v_sector.append(v_r)
            obs_vt_sector.append(v_t_env)
            
        # 状态更新
        self.agent.obs_v_sector = obs_v_sector
        self.agent.obs_vt_sector = obs_vt_sector
        # print(np.array(obs_vt_sector))
        self.agent.obs_sector = curr_sector

    def check_rtPlan(self):
        # 判断路径规划必要性
        '''
            确保路径规划只在一种情况下被调用：
            车辆视线/未来行驶路径上被静态障碍物遮挡，对动态障碍物（一般为其他车辆）不使用A*规划路径
        '''
        r_target = self.position - self.t_pos
        dist_target = np.linalg.norm(r_target)
        if len(self.local_obstacles) > 0:
            obs_array = np.array(self.local_obstacles)
            r_obs = self.position - obs_array
            for index in range(np.size(r_obs, 0)):
                dist_obs = np.linalg.norm(r_obs[index, :])
                shade_angle = np.dot(r_obs[index, :], r_target) / (dist_target * dist_obs)
                if dist_target - dist_obs >= 5 and shade_angle >= 0.7:
                    self.rtPlanFlag = True
                    # print("agent ", self.id, "pos at", self.position, "need rtplan \n")