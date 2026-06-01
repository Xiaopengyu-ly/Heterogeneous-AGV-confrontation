import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt

# ================= 0. 全局配置 =================
CONFIG = {
    'batch_size': 256,
    'lr': 5e-4,
    'epochs': 20,
    'model_path': 'models/predictors/forward_model.pth',
    'dataset_obs': 'dataset/dynamics_dataset_obs.npy',
    'dataset_actions': 'dataset/dynamics_dataset_actions.npy',
    'dataset_targets': 'dataset/dynamics_dataset_trajectorys.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(CONFIG['device'])
print(f"Using device: {device}")

# 观测结构常量 (与 sim/obs_utils.py 对齐)
N_SECTORS = 36
SECTOR_ANGLE = 2 * np.pi / N_SECTORS
LIDAR_MAX_RANGE = 100.0
N_FRAMES = 3
PATCH_DIM = 6             # 5 lidar bins + 1 goal
OUT_DIM = N_SECTORS * PATCH_DIM  # 216
GOAL_DIM = 2              # auxiliary goal_dir prediction


# ================= 1. 辅助模块 =================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal positions."""
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


# ================= 2. AdaLN 模块 =================
class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization: action(5) → scale + shift"""
    def __init__(self, d_model, cond_dim=5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.GELU(),
            nn.Linear(32, d_model * 2)
        )
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, x, cond):
        γ, β = self.cond_mlp(cond).chunk(2, dim=-1)
        if γ.dim() == 2:
            γ = γ.unsqueeze(1)
            β = β.unsqueeze(1)
        return γ * self.norm(x) + β


# ================= 3. ViT Encoder =================
class ViTEncoder(nn.Module):
    def __init__(self, d_model=128, n_frames=N_FRAMES, n_sectors=N_SECTORS,
                 patch_dim=PATCH_DIM, n_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_sectors = n_sectors

        self.patch_embed = nn.Sequential(nn.Linear(patch_dim, d_model), nn.GELU())
        self.angular_pe = nn.Parameter(torch.randn(1, n_sectors, d_model) * 0.02)
        self.temporal_pe = PositionalEncoding(d_model, max_len=n_frames)
        self.dynamics_proj = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, obs_frames, dynamics):
        B, N, S, F = obs_frames.shape
        x = obs_frames.reshape(B, N * S, F)
        x = self.patch_embed(x).reshape(B, N, S, -1)
        x = x + self.angular_pe.unsqueeze(1)
        temporal_pe = self.temporal_pe.pe[:N]
        x = x + temporal_pe.view(1, N, 1, -1)
        x = x.reshape(B, N * S, self.d_model)
        dyn_embed = self.dynamics_proj(dynamics).unsqueeze(1)
        x = torch.cat([dyn_embed, x], dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:, 0, :]  # CLS output


# ================= 4. AdaLN Decoder =================
class AdaLNTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, cond_dim=5, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.adaln1 = AdaLayerNorm(d_model, cond_dim)
        self.adaln2 = AdaLayerNorm(d_model, cond_dim)
        self.adaln3 = AdaLayerNorm(d_model, cond_dim)

    def forward(self, tgt, memory, action_cond, tgt_mask=None):
        x = self.adaln1(tgt, action_cond) if action_cond is not None else tgt
        tgt2 = self.self_attn(x, x, x, attn_mask=tgt_mask)[0]
        tgt = tgt + tgt2
        x = self.adaln2(tgt, action_cond) if action_cond is not None else tgt
        tgt2 = self.cross_attn(x, memory, memory)[0]
        tgt = tgt + tgt2
        x = self.adaln3(tgt, action_cond) if action_cond is not None else tgt
        tgt2 = self.ffn(x)
        tgt = tgt + tgt2
        return tgt


class AdaLNTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, cond_dim, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AdaLNTransformerDecoderLayer(d_model, nhead, dim_feedforward, cond_dim, dropout)
            for _ in range(n_layers)])

    def forward(self, tgt, memory, action_cond, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, action_cond, tgt_mask)
        return tgt


# ================= 5. ForwardPredictor =================
class ForwardPredictor(nn.Module):
    """
    MPPI Forward Model: ViT Encoder + AdaLN-conditioned Transformer Decoder.

    Input:
      - obs_frames: (B, N_frames, 36, 6)  多帧统一观测
      - dynamics:   (B, 2)                 硬件参数
      - actions:    (B, T, 5)               T 步原生 action 序列

    Output:
      - frame_deltas: (B, T, 36, 6)        T 步统一帧增量 (与输入帧格式一致)
      - goal_deltas:  (B, T, 2)            辅助 goal_dir 增量 (用于 MPPI cost)
    """

    def __init__(self, n_frames=N_FRAMES, n_sectors=N_SECTORS, patch_dim=PATCH_DIM,
                 d_model=128, horizon=10, cond_dim=5,
                 n_enc_layers=3, n_dec_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.n_sectors = n_sectors
        self.patch_dim = patch_dim

        self.vit_encoder = ViTEncoder(
            d_model=d_model, n_frames=n_frames, n_sectors=n_sectors,
            patch_dim=patch_dim, n_layers=n_enc_layers, n_heads=n_heads, dropout=dropout)

        self.decoder = AdaLNTransformerDecoder(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            cond_dim=cond_dim, n_layers=n_dec_layers, dropout=dropout)

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_token, std=0.02)

        self.delta_embed = nn.Linear(OUT_DIM, d_model)

        # Output heads: frame (36×6) + auxiliary goal_dir (2)
        self.frame_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, OUT_DIM))
        self.goal_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, GOAL_DIM))

        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(horizon, horizon) * float('-inf'), diagonal=1))

    def forward(self, obs_frames, dynamics, actions, gt_deltas=None):
        """
        obs_frames: (B, N_frames, 36, 6)
        dynamics:   (B, 2)
        actions:    (B, T, 5)
        gt_deltas:  (B, T, 36, 6) or None

        Returns:
          frame_deltas: (B, T, 36, 6)
          goal_deltas:  (B, T, 2)
        """
        B = obs_frames.shape[0]
        T = actions.shape[1]

        scene_repr = self.vit_encoder(obs_frames, dynamics)
        memory = scene_repr.unsqueeze(1)
        action_cond = actions  # (B, T, 5)

        if self.training and gt_deltas is not None:
            # ---- Teacher forcing ----
            prev_deltas_flat = gt_deltas[:, :-1, :].reshape(B, T - 1, OUT_DIM)
            tgt_input = torch.cat([
                self.start_token.expand(B, 1, -1),
                self.delta_embed(prev_deltas_flat)
            ], dim=1)
            h = self.decoder(tgt_input, memory, action_cond,
                             tgt_mask=self.causal_mask[:T, :T])
            frame_out = self.frame_head(h).view(B, T, self.n_sectors, self.patch_dim)
            goal_out = self.goal_head(h)
            return frame_out, goal_out
        else:
            # ---- Autoregressive inference ----
            tgt = self.start_token.expand(B, 1, -1)
            frame_list, goal_list = [], []
            for t in range(T):
                mask = self.causal_mask[:t+1, :t+1]
                cond_t = action_cond[:, t:t+1, :]
                h = self.decoder(tgt, memory, cond_t, tgt_mask=mask)
                delta_t = self.frame_head(h[:, -1:, :])
                goal_t = self.goal_head(h[:, -1:, :])
                frame_list.append(delta_t.view(B, 1, self.n_sectors, self.patch_dim))
                goal_list.append(goal_t)
                embed_t = self.delta_embed(delta_t)
                tgt = torch.cat([tgt, embed_t], dim=1)
            return (torch.cat(frame_list, dim=1),
                    torch.cat(goal_list, dim=1))


# ================= 6. 训练 =================
def train_forward_model(predict_horizon: int = 10):
    obs = np.load(CONFIG['dataset_obs'])          # (N, N_frames, 36, 6)
    actions = np.load(CONFIG['dataset_actions'])   # (N, T, 5)
    targets = np.load(CONFIG['dataset_targets'])   # (N, T, 36, 6)

    print(f"Dataset shapes - Obs: {obs.shape}, Actions: {actions.shape}, Targets: {targets.shape}")

    actions = actions[:, :predict_horizon, :]
    targets = targets[:, :predict_horizon, :, :]

    dynamics_path = CONFIG.get('dataset_dynamics', 'dataset/dynamics_dataset_dynamics.npy')
    if os.path.exists(dynamics_path):
        dynamics = np.load(dynamics_path)
    else:
        dynamics = np.zeros((len(obs), 2), dtype=np.float32)

    obs_t = torch.FloatTensor(obs).to(device)
    dyn_t = torch.FloatTensor(dynamics).to(device)
    act_t = torch.FloatTensor(actions).to(device)
    tgt_t = torch.FloatTensor(targets).to(device)

    full_dataset = TensorDataset(obs_t, dyn_t, act_t, tgt_t)
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    ROLLOUT_WEIGHT = 0.5
    GOAL_WEIGHT = 10.0  # auxiliary goal loss weight

    model = ForwardPredictor(horizon=predict_horizon).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = torch.amp.GradScaler()

    print(f"{'Epoch':>5} | {'Train':>8} | {'TF_frame':>10} {'TF_goal':>8} | "
          f"{'AR_frame':>10} {'AR_goal':>8} | {'AR@T':>10}")

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for b_obs, b_dyn, b_act, b_tgt in train_loader:
            B, T = b_tgt.shape[0], b_tgt.shape[1]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type):
                # ---- Teacher Forcing ----
                pred_tf, goal_tf = model(b_obs, b_dyn, b_act, gt_deltas=b_tgt)

                # Per-patch frame MSE (TF)
                frame_mse_tf = nn.functional.mse_loss(pred_tf, b_tgt)

                # Auxiliary goal: extract gt goal deltas from frame
                # Use the last frame's goal column to estimate goal change
                gt_goal = frame_to_goal_dir_delta(b_tgt)  # (B, T, 2)
                goal_mse_tf = nn.functional.mse_loss(goal_tf, gt_goal)
                loss_tf = frame_mse_tf + GOAL_WEIGHT * goal_mse_tf

                # ---- Autoregressive Rollout ----
                (pred_ar, goal_ar) = model(b_obs, b_dyn, b_act, gt_deltas=None)

                # Cumulative absolute frame
                pred_abs = torch.cumsum(pred_ar, dim=1)
                gt_abs = torch.cumsum(b_tgt, dim=1)

                time_weights = torch.linspace(0.5, 1.5, steps=T, device=device)
                tw = time_weights.view(1, T, 1, 1)
                frame_mse_ar = (nn.functional.mse_loss(
                    pred_abs, gt_abs, reduction='none') * tw).mean()

                gt_goal_abs = torch.cumsum(gt_goal, dim=1)
                pred_goal_abs = torch.cumsum(goal_ar, dim=1)
                tw_g = time_weights.view(1, T, 1)
                goal_mse_ar = (nn.functional.mse_loss(
                    pred_goal_abs, gt_goal_abs, reduction='none') * tw_g).mean()

                loss_ar = frame_mse_ar + GOAL_WEIGHT * goal_mse_ar
                total_loss = loss_tf + ROLLOUT_WEIGHT * loss_ar

            scaler.scale(total_loss).backward()
            scaler.scale(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()

        scheduler.step()

        # ---- 验证 ----
        if (epoch + 1) % 4 == 0:
            model.eval()
            tf_frame_rmse = 0.0
            ar_frame_rmse = torch.zeros(predict_horizon, device=device)
            ar_abs_T = 0.0
            n_val = 0
            with torch.no_grad():
                for b_obs, b_dyn, b_act, b_tgt in val_loader:
                    # TF
                    p_tf, _ = model(b_obs, b_dyn, b_act, gt_deltas=b_tgt)
                    tf_frame_rmse += torch.sqrt(
                        nn.functional.mse_loss(p_tf, b_tgt)).item()

                    # AR
                    p_ar, _ = model(b_obs, b_dyn, b_act, gt_deltas=None)
                    for s in range(predict_horizon):
                        ar_frame_rmse[s] += torch.sqrt(
                            nn.functional.mse_loss(p_ar[:, s, :, :], b_tgt[:, s, :, :]))

                    p_abs = torch.cumsum(p_ar, dim=1)
                    t_abs = torch.cumsum(b_tgt, dim=1)
                    ar_abs_T += torch.sqrt(
                        nn.functional.mse_loss(p_abs[:, -1, :, :], t_abs[:, -1, :, :])).item()
                    n_val += 1

            tf_frame_rmse /= n_val
            ar_frame_rmse /= n_val
            ar_abs_T /= n_val

            print(f"{epoch+1:5d} | {epoch_loss/len(train_loader):8.4f} | "
                  f"{tf_frame_rmse:10.6f} {'':>8} | "
                  f"{ar_frame_rmse.mean().item():10.6f} {'':>8} | "
                  f"{ar_abs_T:10.6f} | "
                  f"AR@1:{ar_frame_rmse[0].item():.6f} AR@T:{ar_frame_rmse[-1].item():.6f}")

    torch.save(model.state_dict(), CONFIG['model_path'])
    return model


def frame_to_goal_dir_delta(frame_deltas):
    """
    Estimate goal_dir delta from 36×6 frame changes.
    Uses the goal column (index 5) peak shift to infer goal direction change.

    frame_deltas: (B, T, 36, 6)
    Returns: (B, T, 2) — estimated [Δrou, Δetheta]
    """
    B, T = frame_deltas.shape[:2]
    device = frame_deltas.device
    goal_col = frame_deltas[..., 5]  # (B, T, 36)
    peak = goal_col.argmax(dim=-1).float()  # (B, T)
    d_etheta = (peak / 36.0 * 2 - 1) * 0.1  # small-scale direction change
    d_rou = goal_col.max(dim=-1).values * 0.1
    return torch.stack([d_rou, d_etheta], dim=-1)


# ================= 7. 帧→距离/目标 提取 (供 MPPI 使用) =================
def frame_to_lidar_dist(frame):
    """
    Convert absolute frame (..., 36, 6) to normalized lidar distances (..., 36).
    Columns 0-4 are lidar bin occupancies (5 bins × 20m each).
    Threshold > 0.5 to find first occupied bin.

    Returns: (..., 36) values in [0, 1] (normalized by LIDAR_MAX_RANGE)
    """
    lidar_bins = torch.clamp(frame[..., :5], 0.0, 1.0)  # (..., 36, 5)
    # Transpose to (..., 5, 36) — bins first
    lidar_2d = lidar_bins.transpose(-2, -1)  # (..., 5, 36)
    n_bins = lidar_2d.shape[-2]
    bin_size = LIDAR_MAX_RANGE / n_bins  # 20m
    occupied = (lidar_2d > 0.5).float()  # (..., 5, 36)
    # For each sector, argmax over bins gives first occupied bin index
    # If no bin is occupied, argmax defaults to 0 → distance = 0 (near)
    first_bin = occupied.argmax(dim=-2).float()  # (..., 36)  values 0..4
    # Handle sectors with no obstacle: set to max
    any_occupied = occupied.sum(dim=-2)  # (..., 36)
    first_bin = torch.where(any_occupied > 0, first_bin,
                            torch.full_like(first_bin, n_bins - 1))
    lidar_dist = first_bin * bin_size / LIDAR_MAX_RANGE
    return lidar_dist


def frame_to_goal_dir(frame):
    """
    Extract normalized goal_dir from the goal column of a frame.
    frame: (..., 36, 6)
    Returns: (..., 2) [rou/200, etheta/pi]
    """
    goal_col = frame[..., 5]  # (..., 36)
    peak_idx = goal_col.argmax(dim=-1).float()  # (...)
    # Sector 0..35 → angle -π..π
    etheta_raw = (peak_idx / 36.0) * 2 * np.pi - np.pi
    etheta = etheta_raw / np.pi  # normalize to [-1, 1]
    rou = goal_col.max(dim=-1).values * 0.5  # rough estimate, capped
    return torch.stack([rou, etheta], dim=-1)


# ================= 8. 可视化 =================
def plot_forward_predictions(forward_model, obs_samples, dyn_samples, act_samples, tgt_samples,
                              horizon=10, sample_idx=0):
    """
    可视化: 初始帧 + T 步预测帧的 36×6 热力图 (GT vs Pred).
    上半部分 = GT, 下半部分 = Pred. 每列是一步.
    """
    forward_model.eval()
    dev = next(forward_model.parameters()).device

    obs_t = torch.FloatTensor(obs_samples[sample_idx:sample_idx+1]).to(dev)
    dyn_t = torch.FloatTensor(dyn_samples[sample_idx:sample_idx+1]).to(dev)
    act_t = torch.FloatTensor(act_samples[sample_idx:sample_idx+1, :horizon]).to(dev)
    gt_deltas = tgt_samples[sample_idx:sample_idx+1, :horizon, :, :]

    with torch.no_grad():
        pred_deltas, _ = forward_model(obs_t, dyn_t, act_t)
    pred_deltas = pred_deltas.cpu().numpy()  # (1, T, 36, 6)

    # Get initial frame from observation (last frame = current)
    init_frame = obs_samples[sample_idx, -1, :, :]  # (36, 6)

    # Cumsum to get absolute frames
    gt_abs = np.cumsum(np.concatenate(
        [np.zeros((1, 1, 36, 6)), gt_deltas], axis=1), axis=1)  # (1, T+1, 36, 6)
    gt_frames = init_frame + gt_abs[0]  # (T+1, 36, 6)

    pred_abs = np.cumsum(np.concatenate(
        [np.zeros((1, 1, 36, 6)), pred_deltas], axis=1), axis=1)
    pred_frames = init_frame + pred_abs[0]  # (T+1, 36, 6)

    T_show = min(horizon + 1, 7)  # show up to 7 steps (step 0 + T)
    step_indices = np.linspace(0, horizon, T_show).astype(int)

    fig, axes = plt.subplots(2, T_show, figsize=(3.5 * T_show, 8))

    for col, step in enumerate(step_indices):
        # GT row
        im_gt = axes[0, col].imshow(gt_frames[step], aspect='auto', cmap='inferno',
                                     vmin=0, vmax=1)
        axes[0, col].set_title(f'GT step {step}')
        axes[0, col].set_xlabel('feature')
        axes[0, col].set_ylabel('sector')

        # Pred row
        im_pr = axes[1, col].imshow(pred_frames[step], aspect='auto', cmap='inferno',
                                     vmin=0, vmax=1)
        axes[1, col].set_title(f'Pred step {step}')
        axes[1, col].set_xlabel('feature')

    axes[0, 0].set_ylabel('GT\nsector')
    axes[1, 0].set_ylabel('Pred\nsector')

    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im_gt, cax=cbar_ax, label='value')

    # Error
    err_frames = pred_frames - gt_frames
    err_mae = np.mean(np.abs(err_frames))
    fig.suptitle(f'Forward Model — Sample #{sample_idx} | '
                 f'Frame MAE: {err_mae:.4f}',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    return fig


def main():
    model = ForwardPredictor(horizon=10).to(device)
    if os.path.exists(CONFIG['model_path']):
        print(">>> Loading model...")
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    else:
        model = train_forward_model(predict_horizon=10)

    obs_samples = np.load(CONFIG['dataset_obs'])
    act_samples = np.load(CONFIG['dataset_actions'])
    tgt_samples = np.load(CONFIG['dataset_targets'])
    dyn_path = CONFIG.get('dataset_dynamics', 'dataset/dynamics_dataset_dynamics.npy')
    if os.path.exists(dyn_path):
        dyn_samples = np.load(dyn_path)
    else:
        dyn_samples = np.zeros((len(obs_samples), 2), dtype=np.float32)

    for idx in np.random.choice(len(obs_samples), size=3, replace=False):
        plot_forward_predictions(model, obs_samples, dyn_samples, act_samples, tgt_samples,
                                  horizon=10, sample_idx=idx)
    plt.show()


if __name__ == "__main__":
    main()
