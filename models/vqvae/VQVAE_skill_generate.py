import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os

# ================= 0. 配置 =================
CONFIG = {
    'T': 5,                # 必须与提取器中的 slice_len / Action Chunking Horizon 对齐
    'action_dim': 5,       # 动作维度
    'latent_dim': 4,       # 4维隐空间
    'num_skills': 16,      # 16个基础技能
    'batch_size': 1024,
    'lr': 1e-3,
    'epochs_ae': 100,      # 增加迭代次数
    'temperature': 1.0,    # Softmax 温度，控制连续性。越小越接近离散，越大越连续。
    'model_path_ae': 'vqvae_skills.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(CONFIG['device'])

# ================= 1. 核心模型：Soft-Quantizer =================
class SoftQuantizer(nn.Module):
    """
    改进点：使用 Softmax 加权代替 Argmin 采样，实现连续技能切换。
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, x, temp=1.0):
        # x: (B, latent_dim), embedding: (K, latent_dim)
        # 1. 计算负欧式距离
        distances = -torch.cdist(x, self.embedding.weight, p=2) # (B, K)
        
        # 2. Softmax 权重 (软分配)
        soft_weights = F.softmax(distances / temp, dim=1)
        
        # 3. 加权求和得到量化向量
        quantized = torch.matmul(soft_weights, self.embedding.weight) # (B, latent_dim)
        
        # 4. 辅助 Loss: 多样性 Loss (让码本向量尽量不要重复)
        w = self.embedding.weight
        norm_w = F.normalize(w, p=2, dim=1)
        cosine_sim = torch.matmul(norm_w, norm_w.t())
        ortho_loss = torch.mean(cosine_sim**2) - (1.0 / self.embedding.num_embeddings)
        
        # 找到最接近的索引仅供分析
        indices = torch.argmax(soft_weights, dim=1).unsqueeze(1)
        
        return quantized, ortho_loss, indices

class SoftVQVAE(nn.Module):
    def __init__(self, seq_len=3, action_dim=5, latent_dim=4, num_skills=16):
        super().__init__()
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.input_dim = seq_len * action_dim
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh() # 强制输出到 [-1, 1]
        )
        
        self.vq = SoftQuantizer(num_skills, latent_dim)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
            nn.Tanh() 
        )

    def forward(self, x, temp=1.0):
        batch_size = x.size(0)
        z = self.enc(x.view(batch_size, -1))
        z_q, ortho_loss, indices = self.vq(z, temp)
        
        # 【核心修改 1】动态利用 seq_len 和 action_dim 还原维度，杜绝硬编码
        recon = self.dec(z_q).view(batch_size, self.seq_len, self.action_dim)
        return recon, ortho_loss, indices

# ================= 2. 改进后的训练函数 =================
def train_soft_vqvae(raw_data):
    data_tensor = torch.FloatTensor(raw_data).to(device)
    train_size = int(0.8 * len(data_tensor))
    train_db, val_db = random_split(TensorDataset(data_tensor), [train_size, len(data_tensor)-train_size])
    
    train_loader = DataLoader(train_db, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_db, batch_size=CONFIG['batch_size'])
    
    # 【核心修改 2】实例化时严格传入 seq_len
    model = SoftVQVAE(seq_len=CONFIG['T'], action_dim=CONFIG['action_dim'], 
                      latent_dim=CONFIG['latent_dim'], num_skills=CONFIG['num_skills']).to(device)
                      
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    print(">>> 启动改进型 Soft-VQVAE 训练...")
    
    for epoch in range(CONFIG['epochs_ae']):
        model.train()
        # 随着训练进行，逐渐降低温度，让分配由“模糊”变得“清晰”
        temp = max(0.5, CONFIG['temperature'] * (0.98 ** epoch))
        total_recon, total_ortho = 0, 0
        
        for batch_x, in train_loader:
            recon_x, ortho_loss, _ = model(batch_x, temp)
            recon_loss = F.mse_loss(recon_x, batch_x)
            
            # 增加动作平滑度约束 (相邻步动作不要跳变太大)
            # 因为 T=3，所以约束的是 T[1]-T[0] 和 T[2]-T[1]
            smooth_loss = torch.mean((recon_x[:, 1:, :] - recon_x[:, :-1, :])**2)
            loss = recon_loss + 0.1 * ortho_loss + 0.05 * smooth_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_ortho += ortho_loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Recon: {total_recon/len(train_loader):.6f} | Temp: {temp:.2f}")
            
    torch.save(model.state_dict(), CONFIG['model_path_ae'])
    return model

# ================= 3. 改进后的可视化 =================
def visualize_continuous_interpolation(model):
    """
    可视化连续性：展示从 Skill A 平滑过渡到 Skill B 的过程
    """
    model.eval()
    with torch.no_grad():
        w = model.vq.embedding.weight.data # (16, 4)
        
        # 选取两个有代表性的技能 ID
        id_a, id_b = 0, 9
        vec_a, vec_b = w[id_a], w[id_b]
        
        steps = 10
        plt.figure(figsize=(18, 5)) # 拉长画布防止子图重叠
        for i in range(steps):
            # 插值
            alpha = i / (steps - 1)
            interp_vec = (1 - alpha) * vec_a + alpha * vec_b
            
            # 【核心修改 3】解码结果利用 config 动态展开
            recon_skill = model.dec(interp_vec.unsqueeze(0)).view(CONFIG['T'], CONFIG['action_dim']).cpu().numpy()
            
            plt.subplot(2, 5, i+1)
            plt.plot(recon_skill[:, 4], label='w_pred', color='red', marker='.') 
            plt.plot(recon_skill[:, 3], label='v_pred', color='green', marker='.')
            plt.plot(recon_skill[:, 2], label='etheta', color='blue', marker='.')
            plt.plot(recon_skill[:, 1], label='ey',     color='gold', marker='.')
            
            plt.title(f"Interp: {alpha:.1f}")
            plt.ylim(-1.1, 1.1)
            plt.xticks(range(CONFIG['T'])) # 明确展示横坐标步数
            if i == 0: 
                plt.legend(loc='lower left', fontsize='small')
        
        plt.suptitle(f"Continuous Skill Interpolation (Transition from Skill {id_a} to {id_b}, T={CONFIG['T']})")
        plt.tight_layout()
        plt.show()

def main():
    raw_data = np.load("dataset/action_dataset.npy")
    print(f">>> 成功加载动作数据集，形状: {raw_data.shape}")
    
    if os.path.exists(CONFIG['model_path_ae']):
        # 【核心修改 4】实例化时严格传入 seq_len
        model = SoftVQVAE(seq_len=CONFIG['T'], action_dim=CONFIG['action_dim'], 
                          latent_dim=CONFIG['latent_dim'], num_skills=CONFIG['num_skills']).to(device)
        model.load_state_dict(torch.load(CONFIG['model_path_ae']))
        print(">>> 加载已有模型")
    else:
        model = train_soft_vqvae(raw_data)
    
    # 训练完毕后，可以解开注释执行连续性验证
    visualize_continuous_interpolation(model)

if __name__ == "__main__":
    main()