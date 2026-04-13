import numpy as np
import heapq
from collections import deque
import heapq

class MapProcesser:
    def chord_length_opt(self, p1, p2, cx, cy, r):
        x1, y1 = p1
        x2, y2 = p2
        r2 = r * r
        dx = x2 - x1
        dy = y2 - y1
        A = dx * dx + dy * dy
        if A == 0.0:
            # 退化为点
            return 0.0 if (x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy) > r2 else 0.0
        # 预计算 1/A（一次除法，后续全用乘法）
        invA = 1.0 / A
        # 端点相对于圆心的向量
        fx = x1 - cx
        fy = y1 - cy
        d1_sq = fx * fx + fy * fy
        gx = x2 - cx
        gy = y2 - cy
        d2_sq = gx * gx + gy * gy
        # 快速路径：两个端点都在圆内 → 返回全长
        if d1_sq <= r2 and d2_sq <= r2:
            return np.sqrt(A)
        # 计算投影参数 t_proj = -dot / A，但用乘法
        dot = fx * dx + fy * dy
        t_proj = -dot * invA  # 替代除法
        # 无分支计算圆心到线段的最近距离平方
        # clamp t_proj to [0, 1] without if
        t_clamped = t_proj
        if t_clamped < 0.0:
            t_clamped = 0.0
        elif t_clamped > 1.0:
            t_clamped = 1.0
        # 注意：这里保留两个分支，因为完全消除会更慢（见下文说明）
        # 计算 clamped 点到圆心的距离平方
        cx_closest = x1 + t_clamped * dx
        cy_closest = y1 + t_clamped * dy
        dist2 = (cx_closest - cx) * (cx_closest - cx) + (cy_closest - cy) * (cy_closest - cy)
        if dist2 > r2:
            return 0.0
        # 解二次方程
        B = 2.0 * dot
        C = d1_sq - r2
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return 0.0
        sqrt_disc = np.sqrt(disc)
        inv_2A = 0.5 * invA  # 避免再算 1/(2*A)
        t1 = (-B - sqrt_disc) * inv_2A
        t2 = (-B + sqrt_disc) * inv_2A
        # 无分支获取 [t_low, t_high] ∩ [0, 1]
        t_low = t1 if t1 < t2 else t2
        t_high = t2 if t1 < t2 else t1
        if t_low < 0.0:
            t_low = 0.0
        if t_high > 1.0:
            t_high = 1.0
        if t_low >= t_high:
            return 0.0
        return (t_high - t_low) * np.sqrt(A)

    def block_and_smoke_check(self, p_pos):
        smoke_zones = self.smoke_zones
        attenuation_coeff = self.smoke_attenuation
        """
        检查从 self.position 到 p_pos 的视线是否被障碍物遮挡，
        并计算穿过烟雾区域的衰减系数。
        参数:
            p_pos: 目标位置 (x, y)
            smoke_zones: 可选，烟雾区域列表，每个元素为 (cx, cy, radius)
            attenuation_coeff: 衰减系数 k，越大衰减越快（默认 0.5）
        返回:
            float: 探测概率衰减系数，范围 [0, 1]
                - 若被障碍物遮挡，返回 0
                - 否则返回 exp(-k * total_smoke_length)
        """
        # === 第一步：障碍物遮挡检查（Bresenham）===
        x0 = int(self.position[0] / self.grid_size)
        y0 = int(self.position[1] / self.grid_size)
        x1 = int(p_pos[0] / self.grid_size)
        y1 = int(p_pos[1] / self.grid_size)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy
        cur_x, cur_y = x0, y0
        while True:
            if self.grid_map[cur_y, cur_x] == 1:
                return 0.0  # 被障碍物完全遮挡
            if cur_x == x1 and cur_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cur_x += sx
            if e2 < dx:
                err += dx
                cur_y += sy
        
        # === 第二步：若无障碍，计算烟雾穿透总长度 ===
        if not smoke_zones:
            return 1.0  # 无烟雾，无衰减
        # total_smoke_length = 0.0
        attenuation = 1
        p1 = self.position
        p2 = p_pos
        for zones in smoke_zones:
            # 调用内部优化版弦长计算
            length_in_smoke = self.chord_length_opt(p1, p2, zones[0][0], zones[0][1], zones[2])
            # total_smoke_length += length_in_smoke
            # 使用指数衰减模型：P = exp(-k * L)
            attenuation *= np.exp(-attenuation_coeff * length_in_smoke)
        return attenuation  # 确保在 [0,1]
    
    def update_obstacles(self, n_closest=4, n_comp=2, connectivity=8):
        mode = "sector"
        if mode == "sector":
            return self.obs_sector
        elif mode == "scan":
            """
            在感知范围内扫描障碍物连通集，
            每个连通集中提取距离自身最近的一个代表点，
            并从中选出整体最近的n个。
            参数:
                n_closest (int): 返回的最近障碍物代表点数量
                connectivity (int): 连通性选择（4或8）
            """
            height, width = self.grid_map.shape
            grid_size = self.grid_size
            panel_center = np.array([width * grid_size * 0.5, height * grid_size * 0.5])
            grid_x = int((self.position[0] + panel_center[0]) / grid_size)
            grid_y = int((self.position[1] + panel_center[1]) / grid_size)
            r = int(self.sense_field / grid_size)
            # Step 1: 扫描感知圆范围内的障碍点
            '''
                感知范围没有考虑参考系旋转，控制指令需要改变
            '''
            scanned_obstacles = []
            for i in range(max(0, grid_x - r), min(width, grid_x + r + 1)):
                dx = i - grid_x
                dx2 = dx * dx
                for j in range(max(0, grid_y - r), min(height, grid_y + r + 1)):
                    dy = j - grid_y
                    if dx2 + dy * dy <= r * r and self.grid_map[j, i] == 1: # 注意：self.grid_map[y, x]
                        scanned_obstacles.append((i, j))
            # Step 2: 提取障碍物连通集
            def get_connected_components(points):
                visited = set()
                components = []
                point_set = set(points)
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else \
                    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
                for p in points:
                    if p in visited:
                        continue
                    q = deque([p])
                    comp = []
                    visited.add(p)
                    while q:
                        cur = q.popleft()
                        comp.append(cur)
                        for dx, dy in directions:
                            nb = (cur[0] + dx, cur[1] + dy)
                            if nb in point_set and nb not in visited:
                                visited.add(nb)
                                q.append(nb)
                    components.append(comp)
                return components
            components = get_connected_components(scanned_obstacles)
            # Step 3: 每个连通集选 n_comp 个代表点（离 agent 最近）
            representative_points = []
            for comp in components:
                closest_pts = heapq.nsmallest(
                    n_comp,
                    comp,
                    key=lambda pt: (pt[0] - grid_x) ** 2 + (pt[1] - grid_y) ** 2
                )
                for pt in closest_pts:
                    dist2 = (pt[0] - grid_x) ** 2 + (pt[1] - grid_y) ** 2
                    representative_points.append((dist2, pt))
            if not representative_points:
                return []
            
            # Step 4: 从代表点中选出最近的 n 个
            closest = heapq.nsmallest(n_closest, representative_points)
            # Step 5: 将 grid 坐标转换回 world 坐标
            self.local_obstacles = [
                (
                    pt[0] * grid_size - panel_center[0],
                    pt[1] * grid_size - panel_center[1]
                )
                for _, pt in closest
            ]
            return self.local_obstacles