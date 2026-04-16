import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt

# ================= 0. 全局配置 =================
CONFIG = {
    'num_skills': 16,   
    'batch_size': 256,  # 1024     
    'lr': 5e-4,
    'epochs': 10,
    'model_path': 'models/predictors/forward_model.pth',
    'dataset_obs': 'dataset/dynamics_dataset_obs.npy',
    'dataset_skills': 'dataset/dynamics_dataset_skills.npy',
    'dataset_targets': 'dataset/dynamics_dataset_trajectorys.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(CONFIG['device'])
print(f"Using device: {device}")

# ================= 1. 辅助模块 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

def circular_gradient_loss(pred_lidar, target_lidar):
    """计算 Lidar 角度维度的循环梯度损失"""
    pred_grad = pred_lidar - torch.roll(pred_lidar, shifts=1, dims=-1)
    target_grad = target_lidar - torch.roll(target_lidar, shifts=1, dims=-1)
    return nn.functional.mse_loss(pred_grad, target_grad)

# ================= 2. 双塔 Transformer 模型 =================
class ForwardPredictor(nn.Module):
    # 【修正2】适应全新的 68 维观测：lidar=36, 剩余辅助特征 aux_dim=32 (3+5+15+9)
    def __init__(self, lidar_dim=36, aux_dim=32, skill_dim=5, hidden_dim=128, horizon=20):
        super().__init__()
        self.horizon = horizon
        self.state_dim = lidar_dim + aux_dim  # 总维度 68
        
        # --- 左塔：当前状态编码器 ---
        self.lidar_encoder = nn.Sequential(
            nn.Unflatten(1, (1, lidar_dim)),
            nn.Conv1d(1, 32, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * lidar_dim, hidden_dim)
        )
        self.aux_net = nn.Linear(aux_dim, 128)
        
        self.skill_proj = nn.Sequential(
            nn.Linear(skill_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        
        self.f0_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 128 + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # --- 右塔与解码器 ---
        self.traj_input_proj = nn.Linear(self.state_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=horizon)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.traj_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=4)
        self.output_head = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, lidar, aux, skill_vecs, target_traj=None):
        l_f = self.lidar_encoder(lidar)
        p_f = self.aux_net(aux)
        s_f = self.skill_proj(skill_vecs) 
        
        f0 = self.f0_fusion(torch.cat([l_f, p_f, s_f], dim=1))
        
        if self.training and target_traj is not None:
            x_t = self.traj_input_proj(target_traj)
            x_t = self.pos_enc(x_t)
            h_t = self.traj_transformer(x_t)
            f_traj = h_t.mean(dim=1)
            
            recon_input = f_traj.unsqueeze(1).repeat(1, self.horizon, 1)
            recon_input = self.pos_enc(recon_input)
            recon_out = self.output_head(self.decoder_transformer(recon_input))
            return f0, f_traj, recon_out
        else:
            pred_input = f0.unsqueeze(1).repeat(1, self.horizon, 1)
            pred_input = self.pos_enc(pred_input)
            pred_out = self.output_head(self.decoder_transformer(pred_input))
            return pred_out

# ================= 3. 训练流程 =================
def train_forward_model(predict_horizen : int = 5):
    obs = np.load(CONFIG['dataset_obs'])
    skills = np.load(CONFIG['dataset_skills'])
    targets = np.load(CONFIG['dataset_targets'])

    print(f"Dataset shapes - Obs: {obs.shape}, Skills: {skills.shape}, Targets: {targets.shape}")

    l_t = torch.FloatTensor(obs[:, :36]).to(device)
    p_t = torch.FloatTensor(obs[:, 36:]).to(device)  # 自动切分出后 32 维辅特征
    k_t = torch.FloatTensor(skills).to(device)
    t_t = torch.FloatTensor(targets).to(device)

    full_dataset = TensorDataset(l_t, p_t, k_t, t_t)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    # 动态传入 Horizon
    model = ForwardPredictor(horizon = predict_horizen).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = torch.amp.GradScaler()

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for l_b, p_b, k_b, t_b in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                f0, f_traj, recon_traj = model(l_b, p_b, k_b, target_traj=t_b)
                
                loss_align = nn.functional.mse_loss(f0, f_traj.detach())
                
                diff_mask = (t_b.abs() > 1e-4).float()
                weight = 1.0 + 5.0 * diff_mask
                mse_recon = (nn.functional.mse_loss(recon_traj, t_b, reduction='none') * weight).mean()
                loss_grad = circular_gradient_loss(recon_traj[:, :, :36], t_b[:, :, :36])
                
                pred_out = model(l_b, p_b, k_b) 
                loss_pred = nn.functional.mse_loss(pred_out, t_b)

                total_loss = loss_align + mse_recon + 1.0 * loss_grad + loss_pred

            scaler.scale(total_loss).backward()
            scaler.scale(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()

        scheduler.step()
        if (epoch+1) % 4 == 0:
            model.eval()
            val_mse = 0
            with torch.no_grad():
                for l_b, p_b, k_b, t_b in val_loader:
                    v_preds = model(l_b, p_b, k_b)
                    val_mse += nn.functional.mse_loss(v_preds, t_b).item()
            print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss/len(train_loader):.6f} | Val MSE: {val_mse/len(val_loader):.6f}")

    torch.save(model.state_dict(), CONFIG['model_path'])
    return model

# ================= 4. 可视化与主函数 =================
def visualize_imagined_trajectories(forward_model, test_obs, predict_horizen = 5, vq_len = 5, vq_path='models/vqvae/vqvae_skills.pth', num_skills=16):
    forward_model.eval()
    device = next(forward_model.parameters()).device

    from models.vqvae.VQVAE_skill_generate import SoftVQVAE 
    vq_model = SoftVQVAE(seq_len=vq_len, action_dim=5, latent_dim=4, num_skills=num_skills).to(device)
    if os.path.exists(vq_path):
        vq_model.load_state_dict(torch.load(vq_path, map_location=device))
        vq_model.eval()
    else:
        print(f"Error: VQ-VAE weights not found at {vq_path}")
        return

    lidar_input = torch.FloatTensor(test_obs[:, :36]).to(device).repeat(num_skills, 1)
    phy_input = torch.FloatTensor(test_obs[:, 36:]).to(device).repeat(num_skills, 1)

    # 3. 构造 5 维技能向量
    with torch.no_grad():
        # 提取 4 维码本向量 (num_skills, 4)，此时范围是 [-1, 1]
        codebook_weights_raw = vq_model.vq.embedding.weight.data
        
        # 【核心修复】：将码本向量映射到 [0, 1] 以通过 SAC 的 semantic Box(0, 1) 校验
        codebook_weights = (codebook_weights_raw + 1.0) / 2.0
        
        # 构造 1 维归一化 ID (num_skills, 1)，范围是 [0, 1]
        skill_ids = torch.arange(num_skills).float().to(device).unsqueeze(1)
        norm_ids = skill_ids / num_skills
        
        # 拼接成 (16, 5) 向量，完全符合 (0, 1) 范围要求
        skill_vecs = torch.cat([codebook_weights, norm_ids], dim=1)

    with torch.no_grad():
        deltas = forward_model(lidar_input, phy_input, skill_vecs).cpu().numpy()

    current_state = test_obs[0]
    preds_abs = current_state[np.newaxis, np.newaxis, :] + deltas

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    colors = plt.cm.jet(np.linspace(0, 1, num_skills))
    
    # --- 左图：XY 平面轨迹 ---
    for i in range(num_skills):
        # 注意：在最新的 normalize_obs_dict 中，物理量 ex, ey 的索引刚好在 36 和 37，保持不变
        ax1.plot(preds_abs[i, :, 36], preds_abs[i, :, 37], 
                 color=colors[i], marker='o', markersize=3, alpha=0.7)
        ax1.text(preds_abs[i, -1, 36], preds_abs[i, -1, 37], f"S{i}", fontsize=10)
    
    ax1.scatter(current_state[36], current_state[37], c='black', marker='X', s=100, label='Current Pos')
    ax1.set_title(f"Imagined Future Trajectories (Horizon={predict_horizen})")
    ax1.set_xlabel("Relative X (Normalized)")
    ax1.set_ylabel("Relative Y (Normalized)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- 右图：雷达剖面变形预测 ---
    sample_id = np.random.randint(0, num_skills)
    step = predict_horizen - 1  # 观察最后一帧的预测雷达
    ax2.plot(current_state[:36], 'b-', label='Current Lidar Scan', lw=2)
    
    pred_lidar = np.clip(preds_abs[sample_id, step, :36], 0, 1)
    ax2.plot(pred_lidar, 'r--', label=f'Predicted Lidar (Skill {sample_id}, Step={step+1})', lw=2)
    
    ax2.set_title(f"Lidar Profile Prediction Detail")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # 动态传入 Horizon
    model = ForwardPredictor(horizon=5).to(device)
    if os.path.exists(CONFIG['model_path']):
        print(">>> Loading model...")
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    else:
        model = train_forward_model()

    obs_samples = np.load(CONFIG['dataset_obs'])
    # 随机取一个样本进行测试
    visualize_imagined_trajectories(model, obs_samples[np.random.randint(0, len(obs_samples)):][:1])

if __name__ == "__main__":
    main()