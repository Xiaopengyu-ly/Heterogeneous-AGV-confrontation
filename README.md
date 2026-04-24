# Heterogeneous AGV Confrontation (异构无人车集群对抗仿真与强化学习平台)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)
![Framework](https://img.shields.io/badge/PyTorch-Stable_Baselines3-orange.svg)

本项目是一个专为异构无人车（AGV）集群设计的轻量级仿真、可视化与算法验证软件。系统支持随机环境生成与初始配置，内置红蓝对抗逻辑、复杂的传感器模型及物理约束。

不仅适用于传统的路径规划与避障算法验证，项目更深度集成了**多智能体强化学习 (MARL)** 与 **模型预测控制 (MPC)** 框架，提供从底层状态检测到高层“技能发现”的完整工具链，为未来的 Sim2Real (ROS2) 部署奠定了高度模块化的基础。

## ✨ 核心特性 (Key Features)

* **高度解耦的架构设计**：采用外观模式重构的 Agent 实体，将能力严格划分为通信 (`Comm`)、检测 (`Check`)、数据感知 (`Data`) 与行为决策 (`Behavior`) 四大子系统。引擎、智能体与规划控制模块分离解耦。
* **分层强化学习与技能表征**：集成了基于 VQ-VAE 的连续动作离散化与技能提取 (Skill Discovery)，支持将隐空间技能输入到下游策略中。
* **前沿规划与控制算法集成**：
  * 支持 **SAC、MAPPO** 等主流强化学习算法 (基于 Stable-Baselines3 / RLlib)。
  * 内置 **Latent MPC** 与动力学前向预测模型 (Forward World Model)，实现隐空间下的安全协同规划。
  * 预留 **Control Barrier Functions (CBF)** 安全过滤器接口，确保物理级的刚性防碰撞边界。
* **复杂的实战对抗机制**：支持火炮塔追踪、攻击范围判定、视野感知衰减以及烟雾弹战术释放等高级博弈逻辑。
* **高性能与可视化**：提供基于 PyQt5 的全景/受控视角可视化面板，支持未来关键模块向 C++ (pybind11) 迁移以进一步加速采样。
* **绝对物理安全边界 (Strict Safety Filters)**： 在 Latent MPC 求解器中，将 Control Barrier Functions (CBF) 作为严格的硬约束（Hard Constraints）而非传统 RL 的软惩罚项（Soft Penalty）进行求解，确保异构集群在极端密集障碍物下的绝对物理防碰撞安全。
* **语义观测空间**： 针对分层强化学习架构进行了针对性优化。在观测空间中专门预留了 5 维的“语义”向量，用于无缝对接 VQ-VAE 提取的离散 skill_id 代码本；同时在智能体属性中解耦了终点目标与动态编队站位点（p_pos），避免了上下层规划逻辑的冲突。

## 📂 项目结构 (Project Structure)

```text
Heterogeneous-AGV-confrontation/
├── agent/        # 智能体核心定义 (架构解耦：PNC、安全过滤器、各类系统组件)
├── comm/         # 进程间/智能体间消息池与通信模拟
├── dataset/      # 仿真采集的动作、观测与轨迹数据集
├── generate/     # 地图、障碍物与智能体参数自动化生成配置
├── models/       # 核心算法模型 (SAC策略、动力学预测模型、VQ-VAE结构)
├── scripts/      # 训练流脚本 (SAC训练、行为克隆、VQ-VAE采样)
├── sim/          # 物理引擎底层与仿真控制器
├── vis/          # 基于 PyQt5 的可视化系统 (面板、烟雾、UI)
└── start.py      # 系统一键启动入口与 Pipeline 编排
```

##  🚀 快速开始 (Quick Start)
##  1. 环境依赖
```Bash
pip install -r requirements.txt
```
*  **核心依赖**: torch, numpy, stable-baselines3, pyqt5, pyyaml
##  2. 运行工作流 (start.py)

项目的入口文件 start.py 编排了从参数生成到模型可视化的完整生命周期。
```Bash
# 调整 start.py 中的 config 参数后运行
python start.py
```
你可以通过取消注释相应的代码块来执行不同的阶段：

* 配置初始化: 生成水陆异构智能体的默认参数与地图配置。

* 阶段一：基础策略训练: 开启多环境并行，训练单体 SAC 策略以获取基础导航能力。

* 阶段二：高级技能提取与微调:

    * 批量仿真收集海量轨迹样本。

    * 训练 VQ-VAE 提取离散技能 (Skills)。

    * 执行行为克隆 (Behavior Cloning) 并对齐特征。

    * 结合技能约束进行 SAC 二次微调 (Finetune)。

* 阶段三：世界模型预测: 训练前向动力学模型 (Forward Model)，实现对未来视野的轨迹预测。

* 阶段四：对抗评估: 加载微调后的模型与 Latent MPC 参数，启动带有红蓝对抗与障碍物的可视化界面进行对抗测试。


##  ⚙️ 智能体参数化 (Agent Customization)
    所有智能体属性均可通过 YAML 配置（通过 agent_config.yaml 或自动化脚本注入），支持自定义：

    动力学: v_max, r_turn_min (最小转弯半径), s_max

    感知与通信: sense_field, sense_angle, connect_dist, 烟雾衰减系数

    火力与对抗: attack_range, cannon_capacity, smoke_capacity, 任务分配权重

##  🔮 未来规划 (Future Work)
    [ ] Sim2Real: 接入 ROS2 框架，将控制指令规范化下发至真实的履带/轮式平台。

    [ ] 算力优化: 将底层频繁调用的碰撞检测与环境状态机重构为 C++ 扩展，大幅提升 RL 采样吞吐量。