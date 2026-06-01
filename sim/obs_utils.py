import numpy as np
import torch
from collections import deque

# Lidar has 36 sectors covering 360 degrees
N_SECTORS = 36
SECTOR_ANGLE = 2 * np.pi / N_SECTORS  # rad per sector
LIDAR_MAX_RANGE = 100.0
HISTORY_LEN = 3  # 2 past frames + 1 current frame


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


# ============================================================
#  新统一帧格式: 每帧 = 36 sector × (5 lidar bins + 1 goal feature)
# ============================================================

def build_unified_frame(lidar_2d, goal_mask):
    """
    Build a single unified frame: combine lidar bins + goal mask per sector.

    lidar_2d: (5, 36)  — 5 bins × 36 sectors, from build_lidar_2d()
    goal_mask: (36,)   — goal_to_lidar_mask() output

    Returns: (36, 6)   — 36 sector tokens, each with [bin0..bin4 | goal]
    """
    lidar_T = lidar_2d.T  # (36, 5)
    goal = goal_mask[:, None]  # (36, 1)
    return np.concatenate([lidar_T, goal], axis=1).astype(np.float32)  # (36, 6)


def build_multi_frame_obs(goal_history, lidar_history, current_lidar_2d, current_goal_mask):
    """
    Build multi-frame observation from history buffers + current frame.

    goal_history:  list of (36,) goal masks for past frames (length = HISTORY_LEN-1)
    lidar_history: list of (5, 36) lidar_2d for past frames (length = HISTORY_LEN-1)
    current_lidar_2d: (5, 36)
    current_goal_mask: (36,)

    Returns: (HISTORY_LEN, 36, 6)  — N_frames × 36 sectors × 6 features
    """
    frames = []
    for i in range(HISTORY_LEN - 1):
        frames.append(build_unified_frame(lidar_history[i], goal_history[i]))
    frames.append(build_unified_frame(current_lidar_2d, current_goal_mask))
    return np.stack(frames, axis=0)  # (N_frames, 36, 6)


def init_agent_buffers(agent):
    """Initialize goal_history and lidar_history buffers on the agent."""
    empty_goal = np.zeros(N_SECTORS, dtype=np.float32)
    empty_lidar = np.zeros((5, N_SECTORS), dtype=np.float32)
    if not hasattr(agent, 'goal_history'):
        agent.goal_history = deque(
            [empty_goal.copy() for _ in range(HISTORY_LEN)],
            maxlen=HISTORY_LEN
        )
    if not hasattr(agent, 'lidar_history'):
        agent.lidar_history = deque(
            [empty_lidar.copy() for _ in range(HISTORY_LEN)],
            maxlen=HISTORY_LEN
        )


def update_agent_buffers(agent, goal_mask, lidar_2d):
    """Push new frame into agent buffers (sliding window)."""
    if not hasattr(agent, 'goal_history') or not hasattr(agent, 'lidar_history'):
        init_agent_buffers(agent)
    agent.goal_history.append(goal_mask.copy())
    agent.lidar_history.append(lidar_2d.copy())


def build_goal_obs(agent):
    """
    Build goal_dir and unified multi-frame observation (new format).

    Returns:
      goal_dir: (2,) [rou/200, etheta/pi]
      obs_frames: (HISTORY_LEN, 36, 6) — multi-frame unified observation
      cur_goal_mask: (36,) for buffer update
      cur_lidar_2d: (5, 36) for buffer update
    """
    dx = agent.p_pos[0] - agent.position[0]
    dy = agent.p_pos[1] - agent.position[1]
    rou = np.hypot(dx, dy)
    target_angle = np.arctan2(dy, dx)
    etheta = _angle_diff(agent.theta, target_angle)

    cur_goal_mask = goal_to_lidar_mask(rou, etheta)
    goal_dir = np.array([rou / 200.0, etheta / np.pi], dtype=np.float32)

    # Build current lidar_2d from raw sensor data
    raw_lidar = np.array(agent.obs_sector, dtype=np.float32)
    cur_lidar_2d = build_lidar_2d(raw_lidar)

    # Initialize buffers if needed
    init_agent_buffers(agent)

    # Build multi-frame observation from history
    past_goals = [agent.goal_history[i] for i in range(HISTORY_LEN - 1)]
    past_lidars = [agent.lidar_history[i] for i in range(HISTORY_LEN - 1)]
    obs_frames = build_multi_frame_obs(past_goals, past_lidars, cur_lidar_2d, cur_goal_mask)

    return goal_dir, obs_frames, cur_goal_mask, cur_lidar_2d


def build_sac_obs(agent):
    """
    Build SAC-compatible observation dict (legacy 4-key format, without ActionVAE).

    Returns:
      lidar_2d:     (5, 36)   — from build_lidar_2d()
      goal_dir:     (2,)      — [rou/200, etheta/pi]
      history_goal: (72,)     — (history_len-1) × 36, without ActionVAE
      dynamics:     (2,)      — hardware params
    """
    dx = agent.p_pos[0] - agent.position[0]
    dy = agent.p_pos[1] - agent.position[1]
    rou = np.hypot(dx, dy)
    target_angle = np.arctan2(dy, dx)
    etheta = _angle_diff(agent.theta, target_angle)

    cur_goal_mask = goal_to_lidar_mask(rou, etheta)
    goal_dir = np.array([rou / 200.0, etheta / np.pi], dtype=np.float32)

    # Build lidar_2d from raw sensor
    raw_lidar = np.array(agent.obs_sector, dtype=np.float32)
    lidar_2d = build_lidar_2d(raw_lidar)

    # Build history_goal WITHOUT ActionVAE embeddings
    init_agent_buffers(agent)

    history_parts = []
    for i in range(HISTORY_LEN - 1):
        past_goal = agent.goal_history[i]
        combined = past_goal + get_positional_encoding(i)
        # No ActionVAE embed — just past_goal + PE + cur_goal
        history_parts.append(np.clip(combined + cur_goal_mask, 0.0, 1.0))

    history_goal = np.concatenate(history_parts).astype(np.float32)

    # Update goal_history buffer
    agent.goal_history.append(cur_goal_mask.copy())

    # dynamics
    dynamics = np.array([agent.v_max / 140.0, agent.r_turn_min / 30.0], dtype=np.float32)

    return lidar_2d, goal_dir, history_goal, dynamics


def _angle_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi
