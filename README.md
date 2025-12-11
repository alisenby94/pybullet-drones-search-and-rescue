# PyBullet Drones - Search and Rescue

Autonomous drone navigation using reinforcement learning with vision-based obstacle avoidance.

## Quick Start

```bash
# Create venv (if not done already)
cd <project_directory>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # should be comprehensive, but untested

# OPTIONAL: Train V4 (car-like control) [MODEL INCLUDED]
python3 train_v4_system.py

# OPTIONAL: Monitor training
tensorboard --logdir ./logs

# Test trained model
python3 test_v4_system.py --model models/v4_car_control/best/best_model.zip


```

## Current System: V4 Car-Like Control

**ActionCoordinatorEnvV4** - 2DOF control (forward/backward + yaw)

**Features:**
- 🚗 Car-like movement (no lateral strafing)
- 👁️ Stereo vision depth perception (512D)
- 🎯 Sequential waypoint navigation
- 🚧 Obstacle avoidance
- 📏 Fixed altitude flight (1.0m)

**Action Space (2D):**
- `vx`: Forward/backward velocity [-2.0, +2.0] m/s
- `yaw_delta`: Turn rate ±10°/step (480°/sec)

**Observation Space (517D):**
- Depth map: 512D (32×16 downsampled stereo)
- Velocity: 2D body frame [vx, vy]
- Yaw: Current heading
- Waypoint vector: 2D relative position

**Training:**
- Algorithm: PPO (Stable-Baselines3)
- Parallel envs: 8
- Total steps: 3M
- Control freq: 48 Hz

## Documentation
Good luck

## Technologies

- **Simulation**: PyBullet, gym-pybullet-drones
- **RL**: Stable-Baselines3 (PPO)
- **Vision**: Stereo depth estimation
- **Framework**: Gymnasium

## Citation

This project uses [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones):

```bibtex
@INPROCEEDINGS{panerati2021learning,
    title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control},
    author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
    booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2021},
    pages={7512-7519},
    doi={10.1109/IROS51168.2021.9635857}
}
```