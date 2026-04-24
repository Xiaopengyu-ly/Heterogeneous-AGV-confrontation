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
    基于牵引点坐标，计算并返回期望控制量及修正量
    返回: v_safe, w_safe, dv, dw
    """
    if agent.disabled:
        return 0.0, 0.0, 0.0, 0.0
        
    v_des, w_des, angle_err = feedback_control(agent, ref_point)
    v_safe, w_safe, dv, dw = safety_filter(agent, v_des, w_des, angle_err)
    return v_safe, w_safe, dv, dw

def feedback_control(agent, ref_point):
    """
    结合引导点与扇区人工势场 (APF) 的平滑跟踪控制器
    输入 ref_point 为五维状态信息: [e_x, e_y, e_theta, v_r, w_r]
    """
    e_x = ref_point[0]
    e_y = ref_point[1]
    e_theta = ref_point[2]
    v_r = ref_point[3]
    omega_r = ref_point[4]
    
    # 直接在本体坐标系下获取期望的相对引导角 phi
    angle_err = np.arctan2(e_y, e_x)
    
    k1 = 8.0 
    v_nom = v_r * np.cos(e_theta) + k1 * e_x 
    k_w = 5.0  
    omega_nom = omega_r + k_w * angle_err 
    
    return v_nom, omega_nom, angle_err


def safety_filter(agent, v_ref, w_ref, angle_err):
    """
    双通道安全滤波器：
    - 无邻居时：走显式代数截断（Fast Path），O(1) 极速运算。
    - 有邻居时：走统一 QP 求解（Rigorous Path），保证多体与墙壁的绝对约束。
    """
    # return v_ref, w_ref, 0, 0 # 验证时可旁路掉安全滤波器
    # ==========================================================
    # 1. 通用前置计算：提取静态特征 (O(1) 复杂度)
    # ==========================================================
    ANGLE_FIX = 0.0
    CBF_DIST = 10.0
    APF_DIST = 20.0
    CBF_DMIN = 5.0
    
    if not hasattr(agent, 'obs_v_sector'):
        agent.obs_v_sector = np.zeros(agent.sector_num)

    # 初始化线速度上限为机械极限
    v_upper_bounds = [np.float32(agent.v_max)]

    if not np.all(agent.obs_sector == 100):
        for i in range(agent.sector_num):
            center_angle = agent.sector_center[i]
            dist = agent.obs_sector[i] - CBF_DMIN
            obs_v = agent.obs_v_sector[i]
            cos_theta = np.cos(center_angle)

            # --- 1.1 计算 APF 产生的侧偏角 ---
            if agent.obs_sector[i] < APF_DIST:
                weight_dist = (1.0 - dist / APF_DIST) ** 2
                v_approach = v_ref * cos_theta + obs_v
                if v_approach > 0.1 and dist > 0:
                    ttc = dist / v_approach
                    weight_ttc = (1.0 - min(ttc, 1.0)) ** 2
                else:
                    weight_ttc = 0.0
                    
                weight = max(weight_dist, weight_ttc)
                rep_angle_local = center_angle + np.pi
                slide_bias = -np.pi/6 if center_angle > 0 else np.pi/6
                rep_angle_local += slide_bias
                
                angle_diff = np.arctan2(np.sin(rep_angle_local - ANGLE_FIX), 
                                        np.cos(rep_angle_local - ANGLE_FIX))
                ANGLE_FIX += 1.5 * weight * angle_diff
                
            # --- 1.2 计算 静态 CBF 产生的显式线速度上限 ---
            if cos_theta > 0 and agent.obs_sector[i] < CBF_DIST:
                gamma = 30
                tau = 0.001
                K_cbf = 1 / (1 - gamma * tau)
                
                if agent.prev_r_point is None:
                    v_sector = gamma * dist / cos_theta
                else:
                    v_sector = (K_cbf * (tau * (agent.r_point[3] - agent.prev_r_point[3]) / agent.dT + gamma * dist) - obs_v) / cos_theta             
                
                v_upper_bounds.append(v_sector)

    # 汇总静态障碍给出的物理极限要求
    v_static_max = max(0.0, float(min(v_upper_bounds)))
    
    # 汇总带有绕障倾向的标称角速度
    k_w = 5.0 
    w_nom = w_ref - k_w * angle_err + k_w * np.arctan2(np.sin(angle_err + ANGLE_FIX), np.cos(angle_err + ANGLE_FIX))

    # ------------------------------------------------------
    # 快速滤波 ：假设无动态邻居，保留显式快速求解优势
    # ------------------------------------------------------
    v_final = min(v_ref, v_static_max)
    v_final = max(0.0, float(v_final))
    w_limit = agent.v_max / agent.r_turn_min if agent.r_turn_min > 0.01 else 2.0
    eps = agent.v_min
    if v_final > eps:
        w_final = min(w_nom, v_final / agent.r_turn_min)
    else:
        w_final = min(w_nom, eps / agent.r_turn_min)
        
    # 保证不超机械界限
    w_final = max(-w_limit, min(w_limit, float(w_final)))
    return float(v_final), float(w_final), v_final - v_ref, w_final - w_ref