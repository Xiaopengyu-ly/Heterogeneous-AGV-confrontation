# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent swarm simulation and reinforcement learning training framework for tactical UAV scenarios with:
- Physics-based agent dynamics with smoke screens, sensing, and weapon systems
- Hierarchical RL architecture (lower-level navigation + higher-level task allocation)
- PyQt5 visualization for real-time simulation and replay

## Commands

### Environment Setup
```bash
conda activate <your-env>  # Requires conda, see .vscode/settings.json
pip install -r requirements.txt
```

### Training
```bash
python train_SAC.py           # SAC training with stable-baselines3 (12 parallel envs)
python backup/train_ray_sac.py # Ray RLlib training with curriculum learning
```

### Inference & Visualization
```bash
python start.py               # Run simulation with pre-trained SAC policy
```

### Testing
```bash
pytest test/test_env.py       # Environment reset/step smoke test
```

## Architecture

### Core Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Visualization Layer (vis/)                                  │
│  - ControlledVisWindow: Main GUI window                      │
│  - SimulationController: Real-time vs replay mode            │
│  - ReplayBuffer: Transition storage                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RL Environment Adapter (RL_train/train_sim_core_lower.py)   │
│  - RLEnvAdapter: Gymnasium wrapper around PhysicsEngine      │
│  - Action space: 15-dim (5-dim × 3-step chunking)            │
│  - Reward: Potential-based + CBF + smoothness                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Physics Engine (sim/physics_engine.py)                      │
│  - step_physics(): Controller execution                      │
│  - env_update(): Smoke, attack resolution, obstacle sensing  │
│  - reset_engine(): State restoration from snapshot           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent System (agent/)                                       │
│  - Agent: Main facade with 4 components:                     │
│    • BehaviorSystem: Move/Sense/Attack models                │
│    • DataSystem: Grid map, route planning (A*/RL)            │
│    • CheckSystem: Hit detection, env feedback                │
│    • CommSystem: Message pool broadcasting                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Environment Model (sim_env/env_model.py)                    │
│  - env_model: Base class for smoke, attack, sensing logic    │
│  - MapGenerator: Procedural obstacle generation              │
└─────────────────────────────────────────────────────────────┘
```

### Key Patterns

**Configuration Management**: `generate_config.py` generates YAML configs with:
- Map layout (256×256 grid, downsampled to 32×32)
- Agent profiles, positions, communication tensors
- Target directions and formation structures

**Agent Profiles**: Loaded via `agent_loader.py` from `agent_config/` directory with parameters for velocity, sensing, weapons, and communication.

**Action Chunking**: RL policies output 15-dim vectors (5 dims × 3 steps), but only the first 5-dim step is executed per environment step for smoother control.

**Multi-Agent Support**: `SubprocVecEnv` with 12 parallel environments for training; per-agent dict observations/actions in petting-zoo style.

### File Structure
```
├── agent/          # Agent logic (core, comm, check, get, models, loader)
├── comm/           # Message pool for inter-agent communication
├── model/          # Neural network models (VQVAE, custom)
├── RL_train/       # RL environment adapters and training init
├── sim/            # Physics engine and simulation init
├── sim_env/        # Map generator and base env model
├── test/           # Unit tests
├── vis/            # Visualization (PyQt5 windows, controllers)
├── backup/         # Archived training scripts (PPO, SAC, Ray)
├── generate_config.py  # Config/map generation
├── train_SAC.py    # Main SAC training entry point
└── start.py        # Inference/visualization entry point
```
