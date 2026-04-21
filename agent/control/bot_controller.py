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

# ==========================================================
# 定义明确的约束函数 (摒弃 lambda 表达式，防止闭包晚期绑定问题)
# ==========================================================
def static_cbf_constraint(u, v_max_limit):
    """
    静态障碍物 CBF 约束公式: v <= v_max_limit
    SciPy 格式要求 fun(u) >= 0，因此转换为: v_max_limit - v >= 0
    u[0] 是线速度 v
    """
    return v_max_limit - u[0]

def dynamic_cbf_constraint(u, dx_body, dy_body, h, L_offset, gamma_dyn):
    """
    动态多体 CBF 约束公式: L_f h + L_g h * u + gamma * h >= 0
    展开为: 2 * dx * v + 2 * L_offset * dy * w + gamma_dyn * h >= 0
    u[0] 是 v, u[1] 是 w
    """
    return 2.0 * dx_body * u[0] + 2.0 * L_offset * dy_body * u[1] + gamma_dyn * h


import numpy as np
from scipy.optimize import minimize

def static_cbf_constraint(u, v_max_limit):
    """静态障碍物线速度上限约束"""
    return v_max_limit - u[0]

def dynamic_cbf_constraint(u, dx_body, dy_body, h, L_offset, gamma_dyn):
    """动态邻居前向偏置点 CBF 避碰约束"""
    return 2.0 * dx_body * u[0] + 2.0 * L_offset * dy_body * u[1] + gamma_dyn * h

def safety_filter(agent, v_ref, w_ref, angle_err):
    """
    双通道安全滤波器：
    - 无邻居时：走显式代数截断（Fast Path），O(1) 极速运算。
    - 有邻居时：走统一 QP 求解（Rigorous Path），保证多体与墙壁的绝对约束。
    """
    
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

    # # ==========================================================
    # # 2. 判断是否需要启动 QP 求解器，正在考虑把此QP转移到顶层MPC实现
    # # ==========================================================
    # has_neighbors = agent.neighbors_id is not [] and len(agent.neighbors_id) > 0
    
    # if has_neighbors:
    #     # ------------------------------------------------------
    #     # 严谨通道 (Rigorous Path)：遇到邻居，启动联合 QP
    #     # ------------------------------------------------------
    #     L_OFFSET = agent.r_point[0]    # 前向偏置距离
    #     D_SAFE = 3       # 动态避碰的安全距离 必须大于 CBF_DMIN *2
    #     GAMMA_DYN = 2.0   # 动态避碰 CBF 增益
        
    #     u_nom = np.array([v_ref, w_nom])
    #     def objective(u):
    #         return 0.5 * ((u[0] - u_nom[0])**2 + (u[1] - u_nom[1])**2)
            
    #     constraints = []
        
    #     # 硬约束 1: 静态避障要求的最大线速度
    #     constraints.append({
    #         'type': 'ineq',
    #         'fun': static_cbf_constraint,
    #         'args': (v_static_max,)
    #     })
        
    #     # 硬约束 2: 其他动态智能体的排斥超平面
    #     theta_i = agent.theta 
    #     px_i = agent.position[0]+ L_OFFSET * np.cos(theta_i)
    #     py_i = agent.position[1]+ L_OFFSET * np.sin(theta_i)
        
    #     for nbr in agent.neighbors_id:
    #         theta_j = np.atan2(agent.neighbors_info["velo"][f"{nbr}"][1],agent.neighbors_info["velo"][f"{nbr}"][0])
    #         px_j = agent.neighbors_info["position"][f"{nbr}"][0] + L_OFFSET * np.cos(theta_j)
    #         py_j = agent.neighbors_info["position"][f"{nbr}"][1] + L_OFFSET * np.sin(theta_j)
            
    #         dx_global = px_i - px_j
    #         dy_global = py_i - py_j
            
    #         D_ij_square = dx_global**2 + dy_global**2
    #         if D_ij_square < CBF_DIST**2:
    #             # 转到本车坐标系
    #             dx_body = dx_global * np.cos(theta_i) + dy_global * np.sin(theta_i)
    #             dy_body = -dx_global * np.sin(theta_i) + dy_global * np.cos(theta_i)

    #             h_val = D_ij_square - D_SAFE**2
    #             constraints.append({
    #                 'type': 'ineq',
    #                 'fun': dynamic_cbf_constraint,
    #                 'args': (dx_body, dy_body, h_val, L_OFFSET, GAMMA_DYN)
    #             })
                
    #     bounds = ((0.0, float(agent.v_max)), (-w_limit, w_limit))
    #     res = minimize(objective, u_nom, method='SLSQP', bounds=bounds, constraints=constraints)
        
    #     if res.success:
    #         v_final, w_final = res.x
    #     else:
    #         # 死锁时兜底停车
    #         v_final, w_final = 0.0, 0.0

    return float(v_final), float(w_final), v_final - v_ref, w_final - w_ref