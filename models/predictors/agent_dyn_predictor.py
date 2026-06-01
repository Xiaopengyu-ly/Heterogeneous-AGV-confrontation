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
    'batch_size': 256,
    'lr': 5e-4,
    'epochs': 20,
    'model_path': 'models/predictors/forward_model.pth',
    'dataset_obs': 'dataset/dynamics_dataset_obs.npy',
    'dataset_skills': 'dataset/dynamics_dataset_skills.npy',
    'dataset_targets': 'dataset/dynamics_dataset_trajectorys.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(CONFIG['device'])
print(f"Using device: {device}")

# 观测结构常量 (与 sim/obs_utils.py 对齐)
N_SECTORS = 36
SECTOR_ANGLE = 2 * np.pi / N_SECTORS
LIDAR_MAX_RANGE = 100.0
HISTORY_LEN = 3          # build_goal_obs 中 history_len
GOAL_ENCODING_DIM = 36   # goal_to_lidar_mask 输出维度


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
    """Lidar 角度维度循环梯度损失"""
    pred_grad = pred_lidar - torch.roll(pred_lidar, shifts=1, dims=-1)
    target_grad = target_lidar - torch.roll(target_lidar, shifts=1, dims=-1)
    return nn.functional.mse_loss(pred_grad, target_grad)


# ================= 2. 自回归 ForwardPredictor (改进版) =================
class ForwardPredictor(nn.Module):
    """
    自回归 Forward Model：obs_t (256-dim) + skill (5-dim) → T 帧增量

    每帧增量: [lidar_dist(36), goal_dir(2)] = 38-dim
    使用因果 Transformer decoder 自回归生成 T 帧预测。

    【改进】自回归推理时，分析性更新 history_goal(72) 和 dynamics(2)：
      - history_goal: 从预测的 goal_dir + ActionVAE(解码后的5维SAC动作) + 位置编码
        通过扇区变换计算，而非让模型隐式学习
      - dynamics: 硬件参数完全不变，保持初始值
      → Forward Model 不再需要隐式学习这 74 维 (29%) 的演化规律

    Training: teacher forcing — 一次前向完成全序列预测
    Inference: autoregressive — 逐帧预测，每步分析性更新 state 并重编码 memory
    """
    def __init__(self, obs_dim=256, skill_dim=5, out_dim=38, d_model=128, horizon=10):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.d_model = d_model

        # ---- Context Encoder (拆分版) ----
        # lidar_2d 扁平: 180-dim (5 bins × 36 sectors)，自回归中保持不变
        self.lidar_encoder = nn.Sequential(
            nn.Linear(180, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )
        # state: goal_dir(2) + history_goal(72) + dynamics(2) = 76-dim
        # 自回归每步分析性更新后重编码
        self.state_encoder = nn.Sequential(
            nn.Linear(76, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )

        # Skill encoder (不变)
        self.skill_encoder = nn.Sequential(
            nn.Linear(skill_dim, 32), nn.GELU(),
            nn.Linear(32, d_model),
        )

        # ---- Skill → ActionVAE 嵌入 预计算表 ----
        # (num_skills, 36): 每个 skill 解码后的首帧 5-dim SAC 动作 → ActionVAE 嵌入
        # 通过 register_skill_action_embeddings() 填充
        self.register_buffer(
            'skill_action_embeddings',
            torch.zeros(CONFIG['num_skills'], GOAL_ENCODING_DIM)
        )

        # ---- Delta embedding (for autoregressive feedback) ----
        self.delta_embed = nn.Linear(out_dim, d_model)

        # ---- Start token ----
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.start_token, std=0.02)

        # ---- Positional encoding ----
        self.pos_enc = PositionalEncoding(d_model, max_len=horizon + 1)

        # ---- History positional encoding buffers (for _build_history_goal) ----
        # 对齐 obs_utils.get_positional_encoding(pos, d_model=36)
        self.register_buffer('_hist_pe_0', self._compute_hist_pe(0))
        self.register_buffer('_hist_pe_1', self._compute_hist_pe(1))

        # ---- Causal Transformer Decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # ---- Output head ----
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, out_dim),
        )

        # Causal mask (reused)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(horizon, horizon) * float('-inf'), diagonal=1)
        )

    @staticmethod
    def _compute_hist_pe(pos, d_model=GOAL_ENCODING_DIM):
        """计算 history 位置编码，对齐 obs_utils.get_positional_encoding"""
        pe = torch.zeros(d_model)
        for i in range(0, d_model, 2):
            div = 10000 ** (i / d_model)
            pe[i] = (torch.sin(torch.tensor(pos / div)) + 1.0) / 2.0
            if i + 1 < d_model:
                pe[i + 1] = (torch.cos(torch.tensor(pos / div)) + 1.0) / 2.0
        return pe

    # ------------------------------------------------------------------
    #  预计算: 每个 skill → ActionVAE 嵌入
    # ------------------------------------------------------------------
    @torch.no_grad()
    def register_skill_action_embeddings(self, vq_model, action_vae):
        """
        预计算所有 16 个 skill 的 ActionVAE 嵌入 (首帧动作 → 36-dim)。

        Args:
          vq_model:   SoftVQVAE, 用于解码 skill latent → T帧动作序列
          action_vae: ActionVAE, 用于编码 5-dim 动作 → 36-dim 嵌入
        """
        num_skills = CONFIG['num_skills']
        device = self.skill_action_embeddings.device
        embeddings = []

        # 从 VQ-VAE codebook 获取每个 skill 的 latent
        codebook = vq_model.vq.embedding.weight.data  # (K, 4), 值域 [-1, 1]

        for i in range(num_skills):
            z = codebook[i:i+1]  # (1, 4)
            # VQ-VAE decoder: latent → T 帧动作
            recon = vq_model.dec(z).view(vq_model.seq_len, vq_model.action_dim)
            first_action = recon[0:1]  # (1, 5) 首帧 SAC 动作

            # ActionVAE: 5-dim 动作 → 36-dim 嵌入 (使用 mean, 无采样噪声)
            vae_embed = action_vae.get_embedding(first_action.to(device))  # (1, 36)
            embeddings.append(vae_embed.squeeze(0))

        self.skill_action_embeddings = torch.stack(embeddings).to(device)  # (16, 36)
        print(f"  ✓ ForwardPredictor: 已预计算 {num_skills} 个 skill 的 ActionVAE 嵌入 "
              f"({self.skill_action_embeddings.shape})")

    # ------------------------------------------------------------------
    #  编码 API
    # ------------------------------------------------------------------
    def encode(self, obs_flat, skill):
        """
        编码观测 + skill → context memory 和 lidar 特征 (供自回归复用)。

        Returns:
          memory:     (B, d_model) 融合后的 context
          lidar_feat: (B, d_model) lidar 特征 (自回归中不变，复用避免重算)
        """
        lidar_feat = self.lidar_encoder(obs_flat[:, :180])
        state_feat = self.state_encoder(obs_flat[:, 180:256])
        f_skill = self.skill_encoder(skill)
        memory = lidar_feat + state_feat + f_skill
        return memory, lidar_feat

    def encode_state_only(self, goal_dir, history_goal, dynamics, lidar_feat, skill):
        """
        仅重编码 state 部分 + skill，复用 lidar_feat。
        用于自回归每步更新 memory。

        Args:
          goal_dir:      (B, 2)
          history_goal:  (B, 72)
          dynamics:      (B, 2)
          lidar_feat:    (B, d_model) 复用的 lidar 特征
          skill:         (B, 5)
        Returns:
          memory: (B, d_model)
        """
        state = torch.cat([goal_dir, history_goal, dynamics], dim=1)  # (B, 76)
        state_feat = self.state_encoder(state)
        f_skill = self.skill_encoder(skill)
        return lidar_feat + state_feat + f_skill

    # ------------------------------------------------------------------
    #  分析性 history_goal 更新 (torch 版, 对齐 obs_utils)
    # ------------------------------------------------------------------
    def _goal_dir_to_lidar_mask(self, rou, etheta):
        """
        torch 版 goal_to_lidar_mask，支持批处理。
        对齐 sim/obs_utils.py::goal_to_lidar_mask

        Args:
          rou:    (B,) 距离 (米)，已反归一化
          etheta: (B,) 相对角度 (弧度) in [-π, π]
        Returns:
          mask: (B, 36) 值域 [0, 1]
        """
        B = rou.shape[0]
        device = rou.device
        sigma = SECTOR_ANGLE
        bin_size = LIDAR_MAX_RANGE / 5.0  # 20m
        n_bins = 5

        # 角度 → 浮点扇区索引
        angle = (etheta + np.pi) % (2 * np.pi)          # (B,) [0, 2π)
        center = angle / (2 * np.pi) * N_SECTORS         # (B,) [0, 36)

        # 距离编码
        bin_idx = torch.clamp((rou / bin_size).long(), 0, n_bins - 1)  # (B,)
        dist_val = torch.clamp(
            1.0 - (rou - bin_idx.float() * bin_size) / bin_size, 0.0, 1.0
        )                                                    # (B,)
        bin_bias = bin_idx.float() / float(n_bins - 1)        # (B,)

        # 扇区高斯权重 (向量化)
        sector_idx = torch.arange(N_SECTORS, device=device).float()  # (36,)
        angular_dist = torch.min(
            torch.abs(sector_idx.unsqueeze(0) - center.unsqueeze(1)),
            N_SECTORS - torch.abs(sector_idx.unsqueeze(0) - center.unsqueeze(1))
        )
        angular_dist_rad = angular_dist * SECTOR_ANGLE
        weight = torch.exp(-angular_dist_rad ** 2 / (2 * sigma ** 2))  # (B, 36)

        mask = dist_val.unsqueeze(1) * weight + 0.15 * bin_bias.unsqueeze(1)
        return torch.clamp(mask, 0.0, 1.0)

    def _build_history_goal(self, goal_history, skill_ids, cur_goal_encoding):
        """
        分析性构建 history_goal(72)。
        对齐 sim/obs_utils.py::build_goal_obs 中的 history_parts 组合逻辑。

        公式 (对每个过去帧 i):
          past_goal(36) + pos_enc(36) + ActionVAE_embed(36) + cur_goal_encoding(36)
          → clip to [0, 1]

        Args:
          goal_history:      list of (B, 36), 长度 = HISTORY_LEN-1 = 2
          skill_ids:         list of (B,) long, 长度 = HISTORY_LEN-1 = 2
                             skill 的离散 ID (0..15)，用于查表获取 ActionVAE 嵌入
          cur_goal_encoding: (B, 36) 当前帧 goal_encoding
        Returns:
          history_goal: (B, 72)
        """
        pe_buffers = [self._hist_pe_0, self._hist_pe_1]
        history_parts = []

        for i in range(len(goal_history)):
            past_goal = goal_history[i]                              # (B, 36)
            pos_enc = pe_buffers[i]                                  # (36,)

            # 按 skill ID 查表获取 ActionVAE 嵌入 (16, 36) → (B, 36)
            skill_embed = self.skill_action_embeddings[skill_ids[i]]  # (B, 36)

            combined = past_goal + pos_enc.unsqueeze(0) + skill_embed + cur_goal_encoding
            history_parts.append(torch.clamp(combined, 0.0, 1.0))

        return torch.cat(history_parts, dim=1)  # (B, 72)

    def _get_skill_id(self, skill):
        """
        从 skill 向量恢复离散 skill ID。

        skill: (B, 5), skill[:, 4] = skill_id / num_skills
        Returns: (B,) long, 值域 [0, num_skills-1]
        """
        ids = torch.round(skill[:, 4] * CONFIG['num_skills']).long()
        return torch.clamp(ids, 0, CONFIG['num_skills'] - 1)

    # ------------------------------------------------------------------
    #  主前向
    # ------------------------------------------------------------------
    def forward(self, obs_flat, skill, gt_deltas=None):
        """
        Training (teacher forcing): 传入 gt_deltas 做一次前向
        Inference (autoregressive): gt_deltas=None, 逐帧生成 + 分析性 state 更新

        Args:
            obs_flat: (B, 256)
            skill:    (B, 5)
            gt_deltas: (B, T, 38) or None
        Returns:
            deltas: (B, T, 38)
        """
        B = obs_flat.shape[0]
        T = self.horizon

        # 初始编码
        memory, lidar_feat = self.encode(obs_flat, skill)

        if self.training and gt_deltas is not None:
            # ---- Teacher forcing: 一次前向 (不变) ----
            memory = memory.unsqueeze(1)  # (B, 1, d_model)
            prev_deltas = gt_deltas[:, :-1, :]  # (B, T-1, 38)
            tgt_input = torch.cat([
                self.start_token.expand(B, 1, -1),
                self.delta_embed(prev_deltas)
            ], dim=1)  # (B, T, d_model)
            tgt_input = self.pos_enc(tgt_input)

            h = self.decoder(tgt_input, memory, tgt_mask=self.causal_mask[:T, :T])
            deltas = self.output_head(h)
            return deltas

        else:
            # ---- Autoregressive inference (改进版: 每步分析性更新 state) ----
            memory = memory.unsqueeze(1)  # (B, 1, d_model)
            tgt = self.start_token.expand(B, 1, -1)  # (B, 1, d_model)

            # 提取初始 state 分量
            cur_goal_dir = obs_flat[:, 180:182].clone()   # (B, 2) 归一化值
            cur_dynamics = obs_flat[:, 254:256].clone()   # (B, 2) 硬件参数, 始终不变

            # 当前 skill 的离散 ID (用于查 ActionVAE 嵌入表)
            cur_skill_id = self._get_skill_id(skill)       # (B,) long

            # 初始 goal_encoding (用于构建 history_goal)
            rou_m = cur_goal_dir[:, 0] * 200.0            # (B,) 反归一化 → 米
            etheta_rad = cur_goal_dir[:, 1] * np.pi       # (B,) 反归一化 → 弧度
            cur_goal_encoding = self._goal_dir_to_lidar_mask(rou_m, etheta_rad)  # (B, 36)

            # 初始化 history 缓冲区 (2 个过去帧)
            # 用初始 goal_encoding 和初始 skill_id 填充 (近似，后续步骤会被分析性值覆盖)
            goal_history = [cur_goal_encoding.clone() for _ in range(HISTORY_LEN - 1)]
            skill_id_history = [cur_skill_id.clone() for _ in range(HISTORY_LEN - 1)]

            deltas_list = []

            for i in range(T):
                tgt_pos = self.pos_enc(tgt)  # (B, i+1, d_model)
                mask = self.causal_mask[:i+1, :i+1]
                h = self.decoder(tgt_pos, memory, tgt_mask=mask)
                delta_i = self.output_head(h[:, -1:, :])   # (B, 1, 38)
                deltas_list.append(delta_i)

                # ============================================================
                # 【核心改进】分析性更新 state → 重编码 memory
                # ============================================================
                # 1. 更新 goal_dir (累积增量)
                goal_delta = delta_i[:, 0, 36:38]          # (B, 2)
                cur_goal_dir = cur_goal_dir + goal_delta

                # 2. 计算新 goal_encoding
                rou_m = cur_goal_dir[:, 0] * 200.0
                etheta_rad = cur_goal_dir[:, 1] * np.pi
                cur_goal_encoding = self._goal_dir_to_lidar_mask(rou_m, etheta_rad)

                # 3. 滑动 history 缓冲区
                goal_history.pop(0)
                goal_history.append(cur_goal_encoding)
                skill_id_history.pop(0)
                skill_id_history.append(cur_skill_id)  # 同一 skill 贯穿整个 rollout

                # 4. 分析性重建 history_goal (使用 ActionVAE 嵌入查表)
                new_history_goal = self._build_history_goal(
                    goal_history, skill_id_history, cur_goal_encoding
                )  # (B, 72)

                # 5. 重编码 state → 更新 memory (复用 lidar_feat, dynamics 不变)
                memory = self.encode_state_only(
                    cur_goal_dir, new_history_goal, cur_dynamics,
                    lidar_feat, skill
                ).unsqueeze(1)

                # 6. 嵌入 delta 供下一步 Transformer 输入
                embed_i = self.delta_embed(delta_i)  # (B, 1, d_model)
                tgt = torch.cat([tgt, embed_i], dim=1)

            return torch.cat(deltas_list, dim=1)  # (B, T, 38)


# ================= 3. 训练 =================
def train_forward_model(predict_horizen: int = 10):
    obs = np.load(CONFIG['dataset_obs'])          # (N, 256)
    skills = np.load(CONFIG['dataset_skills'])     # (N, 5)
    targets = np.load(CONFIG['dataset_targets'])   # (N, T, 38)

    print(f"Dataset shapes - Obs: {obs.shape}, Skills: {skills.shape}, Targets: {targets.shape}")

    # 裁剪到 predict_horizen
    targets = targets[:, :predict_horizen, :]

    obs_t = torch.FloatTensor(obs).to(device)
    skill_t = torch.FloatTensor(skills).to(device)
    tgt_t = torch.FloatTensor(targets).to(device)

    full_dataset = TensorDataset(obs_t, skill_t, tgt_t)
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

    ROLLOUT_WEIGHT = 0.5   # 自回归损失权重 (可调)
    GOAL_MSE_WEIGHT = 500.0

    model = ForwardPredictor(horizon=predict_horizen).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = torch.amp.GradScaler()
    '''
        ┌───────────────────┬────────────────────────────────────────────────────┐
        │       指标        │                        含义                        │
        ├───────────────────┼────────────────────────────────────────────────────┤
        │ TF_L, TF_G        │ Teacher Forcing 下 lidar/goal 每步 delta RMSE 均值 │
        ├───────────────────┼────────────────────────────────────────────────────┤
        │ AR_L, AR_G        │ 自回归下 lidar/goal 每步 delta RMSE 均值           │
        ├───────────────────┼────────────────────────────────────────────────────┤
        │ AR_L@T, AR_G@T    │ 自回归 T 步累积绝对状态 RMSE (核心指标)            │
        ├───────────────────┼────────────────────────────────────────────────────┤
        │ AR_L@1, AR_L@T 等 │ 首步/末步细节                                      │
        └───────────────────┴────────────────────────────────────────────────────┘
    '''
    print(f"{'Epoch':>5} | {'Train':>8} | {'TF_L':>8} {'TF_G':>8} | "
          f"{'AR_L':>8} {'AR_G':>8} | {'AR_L@T':>8} {'AR_G@T':>8}")

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for b_obs, b_skill, b_tgt in train_loader:
            B, T, _ = b_tgt.shape
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type):
                # ============================================================
                #  Loss 1 — Teacher Forcing (预测对齐损失)
                #  使用 gt_deltas 做一次性前向，对比每步 delta
                # ============================================================
                pred_tf = model(b_obs, b_skill, gt_deltas=b_tgt)  # (B, T, 38)

                p_lidar_tf = pred_tf[:, :, :36]
                p_goal_tf  = pred_tf[:, :, 36:38]
                t_lidar    = b_tgt[:, :, :36]
                t_goal     = b_tgt[:, :, 36:38]

                # Lidar 加权 MSE
                diff_mask = (t_lidar.abs() > 1e-4).float()
                weight_lidar = 1.0 + 5.0 * diff_mask
                mse_lidar_tf = (nn.functional.mse_loss(p_lidar_tf, t_lidar, reduction='none')
                                * weight_lidar).mean()

                # 循环梯度损失 (角度维度连续性)
                loss_grad_tf = circular_gradient_loss(p_lidar_tf, t_lidar)

                # Goal MSE (物理量，高权重)
                mse_goal_tf = nn.functional.mse_loss(p_goal_tf, t_goal) * GOAL_MSE_WEIGHT

                loss_tf = mse_lidar_tf + 1.0 * loss_grad_tf + mse_goal_tf

                # ============================================================
                #  Loss 2 — Autoregressive Rollout (自回归预测损失)
                #  不传入 gt_deltas，纯自回归滚动 T 步，对比累积绝对状态
                #  时间加权: 越远的步数权重越高 (0.5 → 1.5)
                # ============================================================
                pred_ar = model(b_obs, b_skill, gt_deltas=None)  # (B, T, 38) 自回归

                p_lidar_ar = pred_ar[:, :, :36]
                p_goal_ar  = pred_ar[:, :, 36:38]

                # 累积绝对状态 (cumsum, 非 delta)
                pred_lidar_abs = torch.cumsum(p_lidar_ar, dim=1)   # (B, T, 36)
                gt_lidar_abs   = torch.cumsum(t_lidar, dim=1)
                pred_goal_abs  = torch.cumsum(p_goal_ar, dim=1)    # (B, T, 2)
                gt_goal_abs    = torch.cumsum(t_goal, dim=1)

                # 时间加权 (后期步数权重更高)
                time_weights = torch.linspace(0.5, 1.5, steps=T, device=device)
                tw_lidar = time_weights.view(1, T, 1)
                tw_goal  = time_weights.view(1, T, 1)

                rollout_lidar_mse = (nn.functional.mse_loss(
                    pred_lidar_abs, gt_lidar_abs, reduction='none'
                ) * tw_lidar).mean()

                rollout_goal_mse = (nn.functional.mse_loss(
                    pred_goal_abs, gt_goal_abs, reduction='none'
                ) * tw_goal).mean() * GOAL_MSE_WEIGHT

                # 自回归下的循环梯度损失
                loss_grad_ar = circular_gradient_loss(p_lidar_ar, t_lidar)

                loss_ar = rollout_lidar_mse + 1.0 * loss_grad_ar + rollout_goal_mse

                # ============================================================
                #  Total Loss
                # ============================================================
                total_loss = loss_tf + ROLLOUT_WEIGHT * loss_ar

            scaler.scale(total_loss).backward()
            scaler.scale(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()

        scheduler.step()

        # ---- 验证 (同时评估 TF 和 AR) ----
        if (epoch + 1) % 4 == 0:
            model.eval()
            # Teacher forcing 指标
            tf_lidar_rmse = torch.zeros(predict_horizen, device=device)
            tf_goal_rmse  = torch.zeros(predict_horizen, device=device)
            # Autoregressive 指标 (累积绝对状态 @T)
            ar_lidar_rmse = torch.zeros(predict_horizen, device=device)
            ar_goal_rmse  = torch.zeros(predict_horizen, device=device)
            ar_lidar_abs_T = 0.0
            ar_goal_abs_T  = 0.0
            n_val = 0
            with torch.no_grad():
                for b_obs, b_skill, b_tgt in val_loader:
                    # ---- Teacher forcing ----
                    v_preds_tf = model(b_obs, b_skill, gt_deltas=b_tgt)
                    v_t_lidar = b_tgt[:, :, :36]
                    v_t_goal  = b_tgt[:, :, 36:38]

                    for s in range(predict_horizen):
                        tf_lidar_rmse[s] += torch.sqrt(
                            nn.functional.mse_loss(v_preds_tf[:, s, :36], v_t_lidar[:, s, :])
                        )
                        tf_goal_rmse[s] += torch.sqrt(
                            nn.functional.mse_loss(v_preds_tf[:, s, 36:38], v_t_goal[:, s, :])
                        )

                    # ---- Autoregressive rollout ----
                    v_preds_ar = model(b_obs, b_skill, gt_deltas=None)
                    v_p_lidar_ar = v_preds_ar[:, :, :36]
                    v_p_goal_ar  = v_preds_ar[:, :, 36:38]

                    # 每步 delta RMSE (自回归)
                    for s in range(predict_horizen):
                        ar_lidar_rmse[s] += torch.sqrt(
                            nn.functional.mse_loss(v_p_lidar_ar[:, s, :], v_t_lidar[:, s, :])
                        )
                        ar_goal_rmse[s] += torch.sqrt(
                            nn.functional.mse_loss(v_p_goal_ar[:, s, :], v_t_goal[:, s, :])
                        )

                    # 累积绝对状态 @T (自回归)
                    abs_p_lidar = torch.cumsum(v_p_lidar_ar, dim=1)
                    abs_t_lidar = torch.cumsum(v_t_lidar, dim=1)
                    abs_p_goal  = torch.cumsum(v_p_goal_ar, dim=1)
                    abs_t_goal  = torch.cumsum(v_t_goal, dim=1)

                    ar_lidar_abs_T += torch.sqrt(
                        nn.functional.mse_loss(abs_p_lidar[:, -1, :], abs_t_lidar[:, -1, :])
                    ).item()
                    ar_goal_abs_T += torch.sqrt(
                        nn.functional.mse_loss(abs_p_goal[:, -1, :], abs_t_goal[:, -1, :])
                    ).item()
                    n_val += 1

            tf_lidar_rmse /= n_val
            tf_goal_rmse  /= n_val
            ar_lidar_rmse /= n_val
            ar_goal_rmse  /= n_val
            ar_lidar_abs_T /= n_val
            ar_goal_abs_T  /= n_val

            print(f"{epoch+1:5d} | {epoch_loss/len(train_loader):8.4f} | "
                  f"{tf_lidar_rmse.mean().item():8.6f} {tf_goal_rmse.mean().item():8.6f} | "
                  f"{ar_lidar_rmse.mean().item():8.6f} {ar_goal_rmse.mean().item():8.6f} | "
                  f"{ar_lidar_abs_T:8.6f} {ar_goal_abs_T:8.6f} | "
                  f"  AR_L@1:{ar_lidar_rmse[0].item():.6f} AR_L@T:{ar_lidar_rmse[-1].item():.6f} "
                  f"AR_G@1:{ar_goal_rmse[0].item():.6f} AR_G@T:{ar_goal_rmse[-1].item():.6f}")

    torch.save(model.state_dict(), CONFIG['model_path'])
    return model


# ================= 4. 可视化 =================
def plot_forward_predictions(forward_model, vq_model, obs_samples, skill_samples, tgt_samples,
                              horizon=10, num_skills=16, sample_idx=0):
    """
    热力图对比: 预测 vs 真值
    左上: lidar 真值 (T×36)      右上: lidar 预测 (T×36)
    左下: goal 轨迹 真值 vs 预测   右下: lidar 误差图 (T×36)
    """
    forward_model.eval()
    device = next(forward_model.parameters()).device

    obs_t = torch.FloatTensor(obs_samples[sample_idx:sample_idx+1]).to(device)  # (1, 256)
    gt_skill = torch.FloatTensor(skill_samples[sample_idx:sample_idx+1]).to(device)
    gt_deltas = tgt_samples[sample_idx:sample_idx+1, :horizon, :]  # (1, T, 38)

    # Ground truth skill 预测
    with torch.no_grad():
        pred_deltas = forward_model(obs_t, gt_skill).cpu().numpy()  # (1, T, 38)

    pred_lidar_delta = pred_deltas[0, :, :36]   # (T, 36)
    pred_goal_delta = pred_deltas[0, :, 36:38]  # (T, 2)
    gt_lidar_delta = gt_deltas[0, :, :36]
    gt_goal_delta = gt_deltas[0, :, 36:38]

    # 累积得到绝对状态
    pred_lidar_abs = np.cumsum(pred_lidar_delta, axis=0)   # (T, 36)
    gt_lidar_abs = np.cumsum(gt_lidar_delta, axis=0)
    pred_goal_abs = np.cumsum(pred_goal_delta, axis=0)     # (T, 2)
    gt_goal_abs = np.cumsum(gt_goal_delta, axis=0)

    # XY 轨迹 (goal: rou/200, etheta/pi)
    pred_rou = pred_goal_abs[:, 0] * 200.0
    pred_etheta = pred_goal_abs[:, 1] * np.pi
    pred_x = pred_rou * np.cos(pred_etheta)
    pred_y = pred_rou * np.sin(pred_etheta)

    gt_rou = gt_goal_abs[:, 0] * 200.0
    gt_etheta = gt_goal_abs[:, 1] * np.pi
    gt_x = gt_rou * np.cos(gt_etheta)
    gt_y = gt_rou * np.sin(gt_etheta)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- 左上: Lidar 真值热力图 ----
    im1 = axes[0, 0].imshow(gt_lidar_abs, aspect='auto', cmap='inferno', vmin=0, vmax=1,
                             extent=[0, 35, horizon-1, 0])
    axes[0, 0].set_title('GT Lidar Distance (abs, cumsum)')
    axes[0, 0].set_xlabel('Beam Index'); axes[0, 0].set_ylabel('Step')
    plt.colorbar(im1, ax=axes[0, 0], label='Norm Distance')

    # ---- 右上: Lidar 预测热力图 ----
    im2 = axes[0, 1].imshow(pred_lidar_abs, aspect='auto', cmap='inferno', vmin=0, vmax=1,
                             extent=[0, 35, horizon-1, 0])
    axes[0, 1].set_title('Pred Lidar Distance (abs, cumsum)')
    axes[0, 1].set_xlabel('Beam Index'); axes[0, 1].set_ylabel('Step')
    plt.colorbar(im2, ax=axes[0, 1], label='Norm Distance')

    # ---- 左下: XY 轨迹对比 ----
    axes[1, 0].plot(gt_x, gt_y, 'k-o', lw=2, markersize=6, label='GT', zorder=10)
    axes[1, 0].plot(pred_x, pred_y, 'r--s', lw=1.5, markersize=5, label='Pred')
    axes[1, 0].scatter(0, 0, c='green', marker='X', s=200, zorder=12, label='Start')
    for i in range(horizon):
        axes[1, 0].annotate(f'{i+1}', (gt_x[i], gt_y[i]), fontsize=7, color='black',
                            xytext=(3, 3), textcoords='offset points')
    axes[1, 0].set_title('XY Trajectory (Goal Prediction)')
    axes[1, 0].set_xlabel('ex (m)'); axes[1, 0].set_ylabel('ey (m)')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')

    # ---- 右下: Lidar 误差图 ----
    lidar_error = pred_lidar_abs - gt_lidar_abs
    vmax = max(abs(lidar_error.min()), abs(lidar_error.max()))
    im3 = axes[1, 1].imshow(lidar_error, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             extent=[0, 35, horizon-1, 0])
    axes[1, 1].set_title('Lidar Error (Pred - GT)')
    axes[1, 1].set_xlabel('Beam Index'); axes[1, 1].set_ylabel('Step')
    plt.colorbar(im3, ax=axes[1, 1], label='Error')

    skill_id_str = f"Skill ID: {int(skill_samples[sample_idx, 4] * 16)}"
    fig.suptitle(f'Forward Model Prediction — Sample #{sample_idx} | {skill_id_str} | '
                 f'Lidar MAE: {np.mean(np.abs(lidar_error)):.4f}, '
                 f'Goal Dist Err@T: {np.hypot(gt_x[-1]-pred_x[-1], gt_y[-1]-pred_y[-1]):.1f}m',
                 fontsize=12)
    plt.tight_layout()
    return fig


def main():
    model = ForwardPredictor(horizon=10).to(device)
    if os.path.exists(CONFIG['model_path']):
        print(">>> Loading model...")
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    else:
        model = train_forward_model(predict_horizen=10)

    from models.vqvae.VQVAE_skill_generate import SoftVQVAE
    vq_model = SoftVQVAE(seq_len=10, action_dim=5, latent_dim=4, num_skills=16).to(device)
    vq_path = 'models/vqvae/vqvae_skills.pth'
    if os.path.exists(vq_path):
        vq_model.load_state_dict(torch.load(vq_path, map_location=device))
        vq_model.eval()

    obs_samples = np.load(CONFIG['dataset_obs'])
    skill_samples = np.load(CONFIG['dataset_skills'])
    tgt_samples = np.load(CONFIG['dataset_targets'])
    idx = np.random.randint(0, len(obs_samples))
    plot_forward_predictions(model, vq_model, obs_samples, skill_samples, tgt_samples,
                              horizon=10, sample_idx=idx)
    plt.show()


if __name__ == "__main__":
    main()
