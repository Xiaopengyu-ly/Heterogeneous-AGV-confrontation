import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ================= 0. 全局配置 =================
CONFIG = {
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 10,
    'embed_dim': 128,      # 最终对齐的特征维度 (体系能力潜能编码)
    'dataset_P': 'dataset/dual_mapping_P.npy',
    'dataset_T': 'dataset/dual_mapping_T.npy',
    'model_save_path': 'model_A_left_tower.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
device = torch.device(CONFIG['device'])
print(f">>> 使用计算设备: {device}")

# ================= 1. 网络结构定义 =================

# 辅助模块：位置编码 (复用你预测器中的设计)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 【左塔】装备性能编码器 (Static Parameter Encoder)
class LeftTower(nn.Module):
    def __init__(self, input_dim=20, embed_dim=128):
        super().__init__()
        # 简单的 MLP 提取静态参数特征
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, p_seq):
        # p_seq: (Batch, 20)
        return self.net(p_seq)

# 【右塔】体系能力时序编码器 (Trajectory/Capability Encoder)
class RightTower(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, embed_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=200)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, t_seq):
        # t_seq: (Batch, Seq_len=200, 2)
        x = self.input_proj(t_seq)          # (B, 200, 64)
        x = self.pos_encoder(x)             # 注入时间位置信息
        x = self.transformer(x)             # (B, 200, 64)
        
        # 时序池化 (取序列的平均特征)
        x_pooled = x.mean(dim=1)            # (B, 64)
        
        # 映射到最终的对齐空间
        return self.output_proj(x_pooled)   # (B, 128)

# ================= 2. 训练主流程 =================
def train_model_A():
    # 1. 加载数据
    if not os.path.exists(CONFIG['dataset_P']) or not os.path.exists(CONFIG['dataset_T']):
        print("错误：未找到数据文件，请先运行 sampler_and_procese.py")
        return
        
    P_data = np.load(CONFIG['dataset_P']) # (N, 20)
    T_data = np.load(CONFIG['dataset_T']) # (N, 200, 2)
    
    # 转换为 Tensor
    P_tensor = torch.FloatTensor(P_data).to(device)
    T_tensor = torch.FloatTensor(T_data).to(device)
    
    dataset = TensorDataset(P_tensor, T_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 2. 初始化双塔与优化器
    left_tower = LeftTower(input_dim=20, embed_dim=CONFIG['embed_dim']).to(device)
    right_tower = RightTower(input_dim=2, embed_dim=CONFIG['embed_dim']).to(device)
    
    # 将两者的参数合并交给一个优化器联合训练
    optimizer = optim.Adam(
        list(left_tower.parameters()) + list(right_tower.parameters()), 
        lr=CONFIG['lr']
    )
    criterion = nn.MSELoss() # 采用均方误差对齐特征空间
    
    # 3. 训练循环
    print("=== 开始联合训练双塔特征对齐模型 (Model A) ===")
    for epoch in range(1, CONFIG['epochs'] + 1):
        epoch_loss = 0.0
        
        for batch_P, batch_T in dataloader:
            optimizer.zero_grad()
            
            # 左塔提取静态装备潜力
            feat_left = left_tower(batch_P)
            
            # 右塔提取真实演化能力
            feat_right = right_tower(batch_T)
            
            # 对齐目标：左塔预测 = 右塔真实
            loss = criterion(feat_left, feat_right)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_P.size(0)
            
        epoch_loss /= len(dataset)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{CONFIG['epochs']}] | Alignment Loss: {epoch_loss:.6f}")
            
    # 4. 重点：只保存左塔 (Left Tower)
    # 因为在推理阶段，右塔(仿真过程)是不存在的，我们只需要左塔！
    torch.save(left_tower.state_dict(), CONFIG['model_save_path'])
    print(f"\n>>> 训练完成！左塔权重已保存至: {CONFIG['model_save_path']}")
    print(">>> 接下来可以通过左塔，瞬间将装备参数 P 转化为 128 维的体系潜能特征！")

if __name__ == "__main__":
    train_model_A()