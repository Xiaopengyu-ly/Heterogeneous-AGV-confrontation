import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from stable_baselines3 import SAC
import os

# ================= 0. 全局配置 =================
CONFIG = {
    'batch_size': 1024,
    'lr': 1e-3,
    'epochs': 70,
    # 严格对齐: history_goal(9) + lidar(36) + prev_act(15) + rel_goal(3) + semantic/skill(5) = 68
    'input_dim': 68,      
    'output_dim': 15,       # Action Chunking (3步 * 5维)
    'model_save_path': 'models/policies/pure_bc_mlp_aligned.pth',
    'sac_base_path': 'models/policies/sac_policy_spirl.zip',      # 初始 SAC 模型
    'sac_save_path': 'models/policies/sac_policy_bc.zip',   # 注入后的部署模型
    'dataset_obs': 'dataset/dynamics_dataset_obs.npy',
    'dataset_skills': 'dataset/dynamics_dataset_skills.npy',
    'dataset_actions': 'dataset/dynamics_dataset_actions.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(CONFIG['device'])

# ================= 1. 对齐后的 MLP 网络 =================
class PureBCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PureBCModel, self).__init__()
        # 严格对接 SB3 默认 SAC 架构 (无 LayerNorm)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# ================= 2. 训练函数 =================
def train_aligned_bc():
    print(">>> 正在加载数据并进行字母顺序重组 (对齐 SB3 FlattenExtractor)...")
    obs_raw = np.load(CONFIG['dataset_obs']).astype(np.float32)
    skills_raw = np.load(CONFIG['dataset_skills']).astype(np.float32)
    actions_seq = np.load(CONFIG['dataset_actions']).astype(np.float32)
    
    # 提取前 3 步动作并展平为 15 维
    targets = actions_seq[:, :3, :].reshape(-1, 15)

    # 简单加权防止全速直线导致的过拟合
    weights = np.ones(len(targets), dtype=np.float32)
    weights[targets[:, 0] < 0.95] *= 5.0
    weights[np.abs(targets[:, 2]) > 0.5] *= 2.0

    # ================= 【核心替换与重组】 =================
    # obs_raw 原始顺序:
    # 0:36 (lidar), 36:39 (rel_goal), 39:44 (原semantic/空位), 44:59 (prev_actions), 59:68 (history_goal)
    lidar        = obs_raw[:, 0:36]
    rel_goal     = obs_raw[:, 36:39]
    # 原有的 dummy semantic 被直接抛弃，用真实提取的 skills_raw 顶替
    prev_actions = obs_raw[:, 44:59]
    history_goal = obs_raw[:, 59:68]

    # SB3 字母排序: history_goal -> lidar -> prev_actions -> rel_goal -> semantic(skill)
    aligned_obs = np.concatenate([
        history_goal,
        lidar,
        prev_actions,
        rel_goal,
        skills_raw  # <--- 技能向量完美嵌入 semantic 预留位
    ], axis=1)

    dataset = TensorDataset(torch.from_numpy(aligned_obs), 
                            torch.from_numpy(targets),
                            torch.from_numpy(weights))
    
    train_size = int(0.9 * len(dataset))
    train_db, val_db = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_db, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_db, batch_size=CONFIG['batch_size'])

    model = PureBCModel(CONFIG['input_dim'], CONFIG['output_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    print(f">>> 启动 BC 架构训练 (Input: {CONFIG['input_dim']}, Output: {CONFIG['output_dim']})...")
    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        for b_obs, b_target, b_weight in train_loader:
            b_obs, b_target, b_weight = b_obs.to(device), b_target.to(device), b_weight.to(device)
            pred = model(b_obs)
            
            mse_loss = (F.mse_loss(pred, b_target, reduction='none') * b_weight.view(-1, 1)).mean()
            direction_penalty = torch.mean(F.relu(-pred[:, 2] * b_target[:, 2]))
            loss = mse_loss + 0.5 * direction_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for b_obs, b_target, _ in val_loader:
                    b_obs, b_target = b_obs.to(device), b_target.to(device)
                    val_loss += F.mse_loss(model(b_obs), b_target).item()
            avg_val = val_loss / len(val_loader)
            print(f"Epoch {epoch+1:3d} | Val MSE: {avg_val:.6f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), CONFIG['model_save_path'])

    print(f">>> BC 训练完成。")

# ================= 3. 注入函数 =================
def inject_to_sac():
    print(f"\n>>> 正在进行架构对齐注入...")
    mlp_net = PureBCModel(CONFIG['input_dim'], CONFIG['output_dim']).to(device)
    mlp_net.load_state_dict(torch.load(CONFIG['model_save_path']))
    
    # 极简注入：因为有 semantic 预留位，直接读取原 SAC 模型即可，无需 DummyEnv
    print(f">>> 加载原生 SAC 模型: {CONFIG['sac_base_path']}")
    sac_model = SAC.load(CONFIG['sac_base_path'], device=device)
    actor = sac_model.policy.actor

    mlp_linears = [m for m in mlp_net.modules() if isinstance(m, nn.Linear)]
    sac_latent_linears = [m for m in actor.latent_pi if isinstance(m, nn.Linear)]
    
    with torch.no_grad():
        # 1. 注入 Latent 部分 (256->256)
        for i in range(2):
            sac_latent_linears[i].weight.copy_(mlp_linears[i].weight)
            sac_latent_linears[i].bias.copy_(mlp_linears[i].bias)
            print(f"  - 注入 SAC latent_pi 层 {i} (维度: {mlp_linears[i].weight.shape})")

        # 2. 注入 Mu 部分 (输出层 256->15)
        actor.mu.weight.copy_(mlp_linears[2].weight)
        actor.mu.bias.copy_(mlp_linears[2].bias)
        print(f"  - 注入 SAC mu 层 (维度: {mlp_linears[2].weight.shape})")

        # 3. 初始化标准差 (置信度设高，让 SAC 退化为确定性策略)
        actor.log_std.weight.data.fill_(0.0)
        actor.log_std.bias.data.fill_(-2.0)

    sac_model.save(CONFIG['sac_save_path'])
    print(f">>> 注入成功！带有 Skill 语义的新模型已保存为: {CONFIG['sac_save_path']}")

if __name__ == "__main__":
    if not os.path.exists(CONFIG['model_save_path']):
        train_aligned_bc()
    inject_to_sac()