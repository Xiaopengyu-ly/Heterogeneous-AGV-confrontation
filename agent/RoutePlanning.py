import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import distance_transform_edt

# 假设此处是通过地图全局视角为小车提供路径规划指引，并非小车自主探索路线
# 避免了处理小车陷入局部最优的情景
class AStarAPF:
    def __init__(self, grid_map, lam=5.0, gamma=2.0):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        # 8 邻域: 上下左右 + 对角
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        self.lam = lam       # 势场权重
        self.gamma = gamma   # 转弯权重

        # 预计算障碍物的距离场
        self.dist = distance_transform_edt(grid_map == 0)

    def heuristic(self, a, b):
        # 欧几里得距离
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def potential(self, x, y):
        d = self.dist[y, x]
        dist_sq = d ** 2
        mu = 0 # 均值
        sigma = 7 # 标准差
        repulsion_cost = 3000 * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((dist_sq - mu) / sigma) ** 2)
        return repulsion_cost

    def is_valid(self, x, y):
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and self.grid_map[y, x] == 0
        )

    def search(self, start_npy : np.array, goal_npy : np.array):
        start = (start_npy[0],start_npy[1])
        if not self.is_valid(goal_npy[0], goal_npy[1]):
            return None
        goal =  (goal_npy[0], goal_npy[1])
        open_set = []
        # (priority, cost, current, parent, direction)
        heapq.heappush(open_set, (0, 0, start, None, (0, 0)))
        came_from = {}
        cost_so_far = {start: 0}
        index = 1
        while open_set and index < 100000:
            index += 1
            _, cost, current, parent, prev_dir = heapq.heappop(open_set)

            if current in came_from:
                continue
            came_from[current] = parent

            if current == goal:
                break

            for dx, dy in self.directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not self.is_valid(nx, ny):
                    continue

                # 基础移动代价
                step_cost = np.hypot(dx, dy)

                # 转弯代价：如果方向不同于 prev_dir，就加罚
                turn_penalty = 0
                if prev_dir != (0, 0) and (dx, dy) != prev_dir:
                    turn_penalty = self.gamma

                new_cost = cost + step_cost + turn_penalty
                neighbor = (nx, ny)

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = (
                        new_cost
                        + self.heuristic(neighbor, goal)
                        + self.lam * self.potential(nx, ny)
                    )
                    heapq.heappush(open_set, (priority, new_cost, neighbor, current, (dx, dy)))

        if goal not in came_from:
            return None

        # 回溯路径
        path = []
        cur = goal
        while cur:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path
    
    # def extract_waypoints(self, path):
    #     """根据方向变化提取航路点"""
    #     if not path or len(path) < 2:
    #         return path

    #     prev_dx = path[1][0] - path[0][0]
    #     prev_dy = path[1][1] - path[0][1]

    #     for i in range(2, len(path)-1):
    #         dx = path[i][0] - path[i-1][0]
    #         dy = path[i][1] - path[i-1][1]
    #         if (dx, dy) != (prev_dx, prev_dy):
    #             waypoints = np.array([path[i-1][0],path[i-1][1]])
    #             return waypoints
    #         prev_dx, prev_dy = dx, dy

    #     waypoints = np.array([path[-1][0],path[-1][1]])
    #     return waypoints
    
    # 
    def extract_waypoints(self, path):
        """
        根据路径方向变化提取第一个转向点作为牵引点，
        并返回五维状态: [x, y, theta, v, w]
        """
        if not path or len(path) < 2:
            # 若路径无效，返回原点+默认状态
            return np.array([0.0, 0.0, 0.0, 0.5, 0.0])

        # 默认速度与角速度（可根据需求调整）
        DEFAULT_V = 1   # m/s
        DEFAULT_W = 0.0   # rad/s

        # 如果路径只有两个点，直接用终点
        if len(path) == 2:
            x, y = path[-1]
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            theta = np.arctan2(dy, dx)
            return np.array([float(x), float(y), theta, DEFAULT_V, DEFAULT_W])

        # 遍历路径，找第一个方向变化点
        prev_dx = path[1][0] - path[0][0]
        prev_dy = path[1][1] - path[0][1]

        for i in range(2, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            if (dx, dy) != (prev_dx, prev_dy):
                # 方向变化发生在 i-1 -> i，取 i-1 为航路点
                x, y = path[i-1]
                # 朝向取变化前的方向（即进入该点的方向）
                theta = np.arctan2(prev_dy, prev_dx)
                return np.array([float(x), float(y), theta, DEFAULT_V, DEFAULT_W])
            prev_dx, prev_dy = dx, dy

        # 若全程无转向，则取终点，方向为最后一段
        x, y = path[-1]
        theta = np.arctan2(prev_dy, prev_dx)
        return np.array([float(x), float(y), theta, DEFAULT_V, DEFAULT_W])


def main():
    from generate_config import generate_config
    generate_config(0)
    from RL_train.train_initialize import train_initialize
    env = train_initialize(0)
    grid_map = env.engine.grid_map
    height, width = grid_map.shape
    grid_size = env.engine.grid_size
    panel_center = np.array([width * grid_size / 2, height * grid_size / 2])
    start = np.array([
        int((env.engine.agents[0].position[0] + panel_center[0]) / (grid_size)),
        int((env.engine.agents[1].position[1] + panel_center[1]) / (grid_size))
    ])
    goal = np.array([
        int((env.engine.agents[0].t_pos[0] + panel_center[0]) / (grid_size)),
        int((env.engine.agents[1].t_pos[1] + panel_center[1]) / (grid_size))
    ])

    planner = AStarAPF(grid_map, lam=5.0, gamma=2.0)
    path = planner.search(start, goal)
    waypoints = planner.extract_waypoints(path)
    # 绘图
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_map, cmap="gray_r")
    if path:
        px, py = zip(*path)
        plt.plot(px, py, "r-", linewidth=1, label="Raw Path")
    if np.any(waypoints):
        wx, wy = waypoints
        plt.scatter(wx, wy, c="yellow", s=100, marker="o", label="Waypoints")
    plt.scatter(start[0], start[1], c="green", s=100, marker="o", label="Start")
    plt.scatter(goal[0], goal[1], c="blue", s=100, marker="x", label="Goal")
    plt.legend()
    plt.title("A* + APF + Turn Cost with Waypoints")
    plt.show()

    print("Raw path length:", len(path))
    print("Waypoint count:", len(waypoints))
    print("Waypoints:", waypoints)


if __name__ == "__main__":
    main()
