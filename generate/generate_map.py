# map_generator.py
import numpy as np

class MapGenerator:
    def __init__(self, width, height, isBlank = False, scale=20, threshold=0.5, seed=None):# 50,0.3
        self.width = width
        self.height = height
        self.d_sample_hw = None
        self.scale = scale
        self.threshold = threshold 
        self.seed = seed
        self.isBlank = isBlank

        self.obs_map =  None    # 固定场景地图
        self.smoke = []  
        self.down_sampled_map= None # 降采样版网格地图，用于强化学习输入降维
        # 存放附加图层，如 {'obstacle': array, 'temp': array}
        
    # 柏林噪声生成网格地图
    def generate_map(self, d_sample_hw):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.d_sample_hw = d_sample_hw

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(t, a, b):
            return a + t * (b - a)

        def grad(hash, x, y):
            h = hash & 15
            u = np.where(h < 8, x, y)
            v = np.where(h < 4, y, x)
            return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

        def perlin_2d(x, y, p):
            xi = np.floor(x).astype(int) & 255
            yi = np.floor(y).astype(int) & 255
            xf = x - np.floor(x)
            yf = y - np.floor(y)

            u = fade(xf)
            v = fade(yf)

            aa = p[p[xi] + yi]
            ab = p[p[xi] + yi + 1]
            ba = p[p[xi + 1] + yi]
            bb = p[p[xi + 1] + yi + 1]

            x1 = lerp(u, grad(aa, xf, yf), grad(ba, xf - 1, yf))
            x2 = lerp(u, grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1))
            return lerp(v, x1, x2)

        def down_sampling(grid_map : np.ndarray, target_hw : np.ndarray):
            original_hw = grid_map.shape
            H, W = original_hw
            H_target, W_target = target_hw

            # 计算每个块的大小（向下取整，确保整除）
            block_h = H // H_target
            block_w = W // W_target

            # 裁剪到能被 block 整除的尺寸(为防止过度裁剪，最好gridmap长宽设置为2的指数)
            crop_H = block_h * H_target
            crop_W = block_w * W_target
            cropped = grid_map[:crop_H, :crop_W]

            # 初始化降采样地图
            downsampled = np.zeros((H_target, W_target), dtype=grid_map.dtype)

            # 手动分块 max pooling
            for i in range(H_target):
                for j in range(W_target):
                    h_start, h_end = i * block_h, (i + 1) * block_h
                    w_start, w_end = j * block_w, (j + 1) * block_w
                    block = cropped[h_start:h_end, w_start:w_end]
                    downsampled[i, j] = np.max(block)  # 有障碍则为1
            return downsampled
        
        
        if self.isBlank:
            grid_map = np.zeros((self.width, self.height), dtype=np.uint8)
            down_sampled_map = down_sampling(grid_map, self.d_sample_hw)
            self.obs_map = grid_map
            self.down_sampled_map = down_sampled_map
            return
        
        # 构造随机排列
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.tile(p, 2)

        # 构建坐标网格
        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        x = x / self.scale
        y = y / self.scale

        noise = perlin_2d(x, y, p)
        grid_map = (noise > self.threshold).astype(np.uint8)
        down_sampled_map = down_sampling(grid_map, self.d_sample_hw)
        self.obs_map = grid_map
        self.down_sampled_map = down_sampled_map

    def load_map(self, obs_map, down_sampled_map):
        self.obs_map = obs_map
        self.down_sampled_map = down_sampled_map

    def gridmap2axis(self, ):
        pass

    def axis2gridmap(self, ):
        pass