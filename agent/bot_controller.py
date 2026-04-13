# bot_controller.py
import numpy as np

def normalize_angle(angle):
    """将角度归一化到 [-π, π]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def guidance_with_obstacle_avoidance(agent, ref_point):
    """
    基于牵引点坐标，返回期望控制量 [v, w]
    """
    if agent.disabled:
        return 0.0, 0.0
    v_des, w_des, angle_err = feedback_control(agent, ref_point)
    v_safe, w_safe = safety_filter(agent, v_des, w_des, angle_err)
    return v_safe,w_safe

# def feedback_control(agent, ref_point):
#     # 向参考点进行Oriolo状态反馈控制
#     # --- 1. 计算本体坐标系下的误差---
#     # 位置误差: reference - current
#     # 牵引点为五维状态信息，[e_x, e_y, e_theta, v_r, w_r]
#     e_x = ref_point[0]
#     e_y = ref_point[1]
#     e_theta = ref_point[2]
#     v_r = ref_point[3]
#     omega_r = ref_point[4]
#     # --- 2. Oriolo 型名义控制律 ---
#     k1 = 4.0   # 原 0.5，放大 8 倍：不仅靠 vr 带动，主要靠位置误差拉动
#     k2 = 10.0  # 原 1.0，放大 10 倍：强力纠偏
#     k3 = 15.0  # 原 2.0，放大 7.5 倍：强力对正航向
#     v_nom = v_r * np.cos(e_theta) + k1 * e_x
#     omega_nom = omega_r + k2 * v_r * e_y + k3 * v_r * np.sin(e_theta)
#     return v_nom, omega_nom



def feedback_control(agent, ref_point):
    """
    结合 SAC 引导点与扇区人工势场 (APF) 的平滑跟踪控制器 (纯本体坐标系优化版)
    输入 ref_point 为五维状态信息: [e_x, e_y, e_theta, v_r, w_r]
    """
    # --- 1. 解析 SAC 引导点参数 ---
    e_x = ref_point[0]
    e_y = ref_point[1]
    e_theta = ref_point[2]
    v_r = ref_point[3]
    omega_r = ref_point[4]
    
    # 直接在本体坐标系下获取 SAC 期望的相对引导角 phi
    angle_err = np.arctan2(e_y, e_x)
    
    # --- 4. 生成名义控制量 (v_nom, omega_nom) ---
    k1 = 4.0 
    v_nom = v_r * np.cos(e_theta) + k1 * e_x 
    k_w = 5.0  
    omega_nom = omega_r + k_w * angle_err 
    
    return v_nom, omega_nom, angle_err


def safety_filter(agent, v_ref, w_ref, angle_err):
    # print(agent.obs_sector)
    v_constraints = []
    v_constraints.append(v_ref)
    v_constraints.append(np.float32(agent.v_max))
    angle_fix = 0
    CBF_dist = 10.0
    APF_dist = 20.0
    # 正值代表障碍物正在朝车体靠近。后续可用多帧雷达数据的统计差分来更新此数组。
    if not hasattr(agent, 'obs_v_sector'):
        agent.obs_v_sector = np.zeros(agent.sector_num)

    if np.all(agent.obs_sector == 100):
        # 扇区还未检测到障碍物
        v_safe = min(v_constraints)    
        v_safe = max(0, v_safe)
        dv = v_safe - v_ref
        # 【新增代码 1】将线速度修正量记录到 agent 实例中
        agent.dv = dv

        eps = agent.v_min
        if v_safe > eps:
            w_safe = min(w_ref, v_safe / agent.r_turn_min)  # 保持当前朝向
        else:
            w_safe = min(w_ref, eps / agent.r_turn_min)
        dw = w_safe - w_ref
        # 【新增代码 2】将角速度修正量记录到 agent 实例中
        agent.dw = dw
        
        return v_safe, w_safe
    else:
        for i in range(agent.sector_num):
            center_angle = agent.sector_center[i]
            gamma = 30
            d_min = 5
            tau = 0.001
            K_cbf = 1 / (1 - gamma * tau)
            dist = agent.obs_sector[i] - d_min
            obs_v = agent.obs_v_sector[i]

            # ==========================================================
            # 改进 3: 基于 TTC (碰撞时间) 与侧向绕行的 APF
            # ==========================================================
            cos_theta = np.cos(center_angle)
            TTC_max = 1.0
            if agent.obs_sector[i] < APF_dist:
                # 1. 基础距离权重 (保证底线排斥，防止静态贴d身时失去斥力)
                weight_dist = (1.0 - dist / APF_dist) ** 2
                
                # 2. 动态 TTC 权重 (提前应对高速逼近的威胁)
                v_approach = v_ref * cos_theta  + obs_v
                if v_approach > 0.1 and dist > 0:
                    ttc = dist / v_approach
                    weight_ttc = (1.0 - min(ttc, TTC_max) / TTC_max) ** 2
                else:
                    weight_ttc = 0.0
                    
                # 综合权重：取距离威胁和动态威胁中的最大值
                weight = max(weight_dist, weight_ttc)
                
                obs_angle_local = center_angle
                
                # 3. 斥力方向：恢复 180 度主排斥，防止绕圈和靠近
                rep_angle_local = obs_angle_local + np.pi
                
                # 4. 侧滑偏置：引入温和的侧向力打破死锁 (偏转 30 度)，不再使用激进的 90 度
                # 如果障碍物在左侧，向右侧滑 (-pi/6)；障碍物在右侧，向左侧滑 (+pi/6)
                slide_bias = -np.pi/6 if obs_angle_local > 0 else np.pi/6
                rep_angle_local += slide_bias
                
                # 计算角度差并累加
                angle_diff = np.arctan2(np.sin(rep_angle_local - angle_fix), 
                                        np.cos(rep_angle_local - angle_fix))
                angle_fix += 1.5 * weight * angle_diff
                
            if np.cos(center_angle) > 0 and agent.obs_sector[i] < CBF_dist:
                # dist = agent.obs_sector[i] - d_min
                # cbf 显式速度约束
                if agent.prev_r_point is None:
                    v_sector = gamma * (dist) / np.cos(center_angle) 
                else:
                    # v_sector = K_cbf * (tau * (agent.r_point[3] - agent.prev_r_point[3]) / agent.dT + gamma * (dist) / np.cos(center_angle) )  
                    v_sector = (K_cbf * (tau * (agent.r_point[3] - agent.prev_r_point[3]) / agent.dT + gamma * dist) - obs_v) / cos_theta             
                v_constraints.append(v_sector)

        v_safe = min(v_constraints)    
        v_safe = max(0, v_safe)
        dv = v_safe - v_ref
        # 【新增代码 1】将线速度修正量记录到 agent 实例中
        agent.dv = dv

        k_w = 5.0 
        w_fix = w_ref - k_w * angle_err + k_w * np.arctan2(np.sin(angle_err + angle_fix), np.cos(angle_err + angle_fix))

        eps = agent.v_min
        if v_safe > eps:
            w_safe = min(w_fix, v_safe / agent.r_turn_min)  # 保持当前朝向
        else:
            w_safe = min(w_fix, eps / agent.r_turn_min)
        dw = w_safe - w_ref
        # 【新增代码 2】将角速度修正量记录到 agent 实例中
        agent.dw = dw

        return v_safe, w_safe
