# agent/agent_loader.py
import yaml
import os
import numpy as np

def load_agent_config(config_name="default", config_path = None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_all = yaml.safe_load(f)
    
    if config_name not in config_all:
        raise ValueError(f"Config '{config_name}' not found in {config_path}")
    
    config = config_all[config_name]
    
    # 自动转换角度单位
    if "sense_angle_deg" in config:
        config["sense_angle"] = np.deg2rad(config.pop("sense_angle_deg"))
    if "cannon_w_max_deg" in config:
        config["cannon_w_max"] = np.deg2rad(config.pop("cannon_w_max_deg"))
    
    return config