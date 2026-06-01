import numpy as np
import torch
from collections import deque

# Lidar has 36 sectors covering 360 degrees
N_SECTORS = 36
SECTOR_ANGLE = 2 * np.pi / N_SECTORS  # rad per sector
LIDAR_MAX_RANGE = 100.0


def goal_to_lidar_mask(rou, etheta, sigma=None):
    """
    Convert polar goal coords (rou, etheta) to a 36-dim lidar-aligned mask.

    Distance encoding is bin-normalized to match lidar_2d's 5-bin structure:
      - dist_val: [0,1] across one 20m bin (same scale as lidar_2d occupancy)
      - bin_bias:  [0,1] encodes which bin the target falls in

    rou: distance to goal (meters)
    etheta: angle to goal in [-pi, pi], relative to agent heading
    sigma: Gaussian width in radians. Defaults to one sector width (pi/18 ≈ 10°).
    """
    if sigma is None:
        sigma = SECTOR_ANGLE

    bin_size = LIDAR_MAX_RANGE / 5.0   # 20m per bin, matches lidar_2d
    n_bins = 5

    angle = (etheta + np.pi) % (2 * np.pi)
    center = angle / (2 * np.pi) * N_SECTORS  # float sector index

    mask = np.zeros(N_SECTORS, dtype=np.float32)

    # Bin-normalized distance (aligned with lidar_2d occupancy semantics)
    bin_idx = min(int(rou / bin_size), n_bins - 1)
    dist_val = np.clip(1.0 - (rou - bin_idx * bin_size) / bin_size, 0.0, 1.0)
    bin_bias = bin_idx / float(n_bins - 1)  # 0.0 .. 1.0

    for i in range(N_SECTORS):
        angular_dist = min(abs(i - center), N_SECTORS - abs(i - center))
        angular_dist_rad = angular_dist * SECTOR_ANGLE
        weight = np.exp(-angular_dist_rad ** 2 / (2 * sigma ** 2))
        mask[i] = dist_val * weight + 0.15 * bin_bias

    return np.clip(mask, 0.0, 1.0)


def get_positional_encoding(pos, d_model=36):
    """Sinusoidal positional encoding (base=10000), normalized to [0, 1].
    Used for temporal/history encoding."""
    pe = np.zeros(d_model, dtype=np.float32)
    for i in range(0, d_model, 2):
        div = 10000 ** (i / d_model)
        pe[i] = (np.sin(pos / div) + 1.0) / 2.0
        if i + 1 < d_model:
            pe[i + 1] = (np.cos(pos / div) + 1.0) / 2.0
    return pe


def get_distance_pe(dist_meters, d_model=36, base=10.0):
    """Distance positional encoding (base=10, vs base=10000 for temporal PE).
    Encodes continuous distance in meters, producing a fingerprint
    orthogonal to the temporal PE used in history_goal."""
    pe = np.zeros(d_model, dtype=np.float32)
    for i in range(0, d_model, 2):
        div = base ** (i / d_model)
        pe[i] = (np.sin(dist_meters / div) + 1.0) / 2.0
        if i + 1 < d_model:
            pe[i + 1] = (np.cos(dist_meters / div) + 1.0) / 2.0
    return pe


def build_lidar_2d(raw_lidar, n_sectors=36, n_bins=5, max_range=100.0, pe_weight=0.3):
    """
    Build 2D lidar: (n_bins, n_sectors) continuous occupancy + distance PE.

    raw_lidar: (36,) raw distances in meters per sector.
    Returns: (n_bins, n_sectors) float32, values clipped to [0, 1].

    Each of the 5 bins spans 20m. Occupancy is continuous:
      0.0 = free, 1.0 = blocked, intermediate = obstacle within bin.
    Distance PE (base=10) is added per bin to encode distance from vehicle.
    """
    bin_size = max_range / n_bins
    raw = np.asarray(raw_lidar, dtype=np.float32)
    raw = np.where(raw <= 0, max_range, raw)

    lidar_2d = np.zeros((n_bins, n_sectors), dtype=np.float32)

    # Occupancy per sector × bin
    for i in range(n_sectors):
        d = raw[i]
        for j in range(n_bins):
            b_start = j * bin_size
            b_end = (j + 1) * bin_size
            if b_end <= d:
                lidar_2d[j, i] = 0.0
            elif b_start >= d:
                lidar_2d[j, i] = 1.0
            else:
                lidar_2d[j, i] = 1.0 - (d - b_start) / bin_size

    # Distance PE: one 36-dim vector per bin
    for j in range(n_bins):
        dist_m = float(j * bin_size + bin_size * 0.5)  # bin centres: 10, 30, 50, 70, 90
        dist_pe = get_distance_pe(dist_m, d_model=n_sectors)
        lidar_2d[j] = np.clip(lidar_2d[j] + pe_weight * dist_pe, 0.0, 1.0)

    return lidar_2d


def build_goal_obs(agent, history_len, vae_model):
    """
    Build goal_dir and history_goal.

    Returns (goal_dir, history_goal):
      - goal_dir: (2,) [rou/200, etheta/pi], direct polar signal for steering + speed
      - history_goal: ((history_len-1)*36,) past masks + PE + VAE + goal_encoding

    Also updates agent.goal_history and agent.action_history buffers in-place.
    """
    dx = agent.p_pos[0] - agent.position[0]
    dy = agent.p_pos[1] - agent.position[1]
    rou = np.hypot(dx, dy)
    target_angle = np.arctan2(dy, dx)
    etheta = _angle_diff(agent.theta, target_angle)
    goal_encoding = goal_to_lidar_mask(rou, etheta)
    goal_dir = np.array([rou / 200.0, etheta / np.pi], dtype=np.float32)

    if not hasattr(agent, 'goal_history'):
        agent.goal_history = deque(
            [goal_encoding.copy() for _ in range(history_len)],
            maxlen=history_len
        )
    if not hasattr(agent, 'action_history'):
        agent.action_history = deque(
            [np.zeros(5, dtype=np.float32) for _ in range(history_len - 1)],
            maxlen=history_len - 1
        )

    history_parts = []
    for i in range(history_len - 1):
        past_goal = agent.goal_history[i]
        combined = past_goal + get_positional_encoding(i)
        past_action = agent.action_history[i]
        action_tensor = torch.from_numpy(past_action).float().unsqueeze(0)
        vae_embed = vae_model.get_embedding(action_tensor).squeeze(0).numpy()
        history_parts.append(np.clip(combined + vae_embed + goal_encoding, 0.0, 1.0))

    history_goal = np.concatenate(history_parts).astype(np.float32)

    agent.goal_history.append(goal_encoding.copy())

    return goal_dir, history_goal


def _angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi
