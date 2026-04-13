from PyQt5.QtWidgets import QApplication
from stable_baselines3 import SAC,PPO
from vis.controlled_window import ControlledVisWindow
from sim.sim_initialize import sim_initialize
import sys
from generate_config import generate_config


def main():
    data_id = 0
    generate_config(data_id)
    sim = sim_initialize(data_id)
    model = SAC.load("sac_policy_spirl")
    app = QApplication(sys.argv)
    max_steps = 2000
    config = {
        "case": "sim_onceonly", #"sim_onceonly", # "replay_sim"
        "max_steps": max_steps,
        "data_id": data_id,
        "buffer_capacity": max_steps,
        "lower_actor": model,
        "use_latent_mpc" : True
    }
    # 创建窗口（此时 QApplication 已存在）
    window = ControlledVisWindow(sim, config)
    window.show()
    # 启动事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()