# Hierarchical RL System - Project Files

## 🎯 Core System Files

### Training Environments
- **`training_env_motor_control.py`** - Stage 2: Low-level motor control environment
  - Negative quadratic tracking reward
  - 17D observation space
  - 4D action space (RPM offsets)
  - Curriculum learning (hover → slow → fast)

- **`training_env_path_planning.py`** - Stage 1: High-level path planning environment
  - Progress-based reward with obstacle avoidance
  - 2071D observation space (includes 64x32 vision)
  - 4D action space (velocity commands)
  - Integrates frozen Stage 2 motor controller

### Training Scripts
- **`train_motor_control.py`** - Train Stage 2 motor controller
  - PPO with MLP policy
  - VecNormalize wrapper
  - Checkpoint and evaluation callbacks
  - Usage: `python train_motor_control.py --timesteps 500000 --name motor_v1`

### Vision System
- **`stereo_vision.py`** - Stereo camera system
  - Dual cameras with configurable baseline
  - 128x128 depth map capture
  - PyBullet integration

- **`depth_downsampler.py`** - Vision preprocessing
  - Min-pooling downsampler (128x128 → 64x32)
  - Preserves closest obstacles
  - 96.9% memory savings

## 🧪 Testing & Visualization
- **`test_stereo_vision.py`** - Test stereo camera system
- **`demo_stable_stereo.py`** - Demo stable stereo vision
- **`demo_realtime_stereo.py`** - Real-time stereo visualization
- **`visualize_stereo.py`** - Visualize stereo depth maps
- **`verify_ready.py`** - Verify system readiness before training

## 📚 Documentation
- **`README.md`** - Project overview
- **`ARCHITECTURE.md`** - Complete system architecture
- **`TRAINING_STRATEGY.md`** - Detailed training strategy and curriculum
- **`STEREO_VISION_README.md`** - Stereo vision system documentation
- **`PROJECT_FILES.md`** - This file (file inventory)

## 🗂️ Directory Structure

```
pybullet-drones-search-and-rescue/
├── simulation/
│   └── gym-pybullet-drones/      # PyBullet drone simulation library
├── .venv/                         # Python virtual environment
├── models/                        # Trained models (created during training)
├── logs/                          # TensorBoard logs (created during training)
└── res/                           # Resources and assets
```

## 🚀 Quick Start

### 1. Train Stage 2 (Motor Control)
```bash
python train_motor_control.py --timesteps 500000 --name motor_control_v1
```

### 2. Monitor Training
```bash
tensorboard --logdir ./logs/motor_control_v1
```

### 3. Train Stage 1 (Path Planning)
```bash
# Coming soon - after Stage 2 validation
python train_path_planning.py --timesteps 1000000 --name planning_v1 \
    --motor_controller ./models/motor_control_v1/best_model.zip
```

## 📦 Dependencies

Core dependencies (from `requirements.txt`):
- `stable-baselines3[extra]>=2.0.0` - RL algorithms
- `gymnasium>=0.29.0` - RL environment interface
- `torch>=2.0.0` - Deep learning framework
- `pybullet>=3.2.0` - Physics simulation
- `numpy>=1.20.0` - Numerical computing
- `tensorboard>=2.0.0` - Training visualization

## 🧹 Cleaned Up

The following old files from previous approaches have been removed:
- ❌ Old training environments (velocity control, vector nav, SAR, takeoff/landing)
- ❌ Old training scripts (SAR, navigation, simple hover, etc.)
- ❌ Old visualization scripts
- ❌ Old test scripts
- ❌ Old documentation (velocity control, credit assignment, etc.)
- ❌ Chat interfaces and LLM planning scripts
- ❌ MDP graph builders
- ❌ Static map generators
- ❌ All old models and logs

**This is the ONLY approach that matters now.**

## 📊 Training Progress

### Phase 1: Stage 2 Motor Control (IN PROGRESS)
- ⏳ Training with improved curriculum (hover → slow → fast)
- ⏳ Target: < 0.1 m/s tracking error
- ⏳ Duration: ~500k timesteps

### Phase 2: Stage 2 Validation (PENDING)
- ⏳ Test tracking accuracy
- ⏳ Validate stability over full episodes
- ⏳ Freeze weights for Stage 1 integration

### Phase 3: Stage 1 Path Planning (PENDING)
- ⏳ Train with frozen Stage 2
- ⏳ Target: > 80% waypoint success rate
- ⏳ Duration: ~1M timesteps

### Phase 4: End-to-End Testing (PENDING)
- ⏳ Full mission scenarios
- ⏳ Obstacle avoidance validation
- ⏳ Performance benchmarking
