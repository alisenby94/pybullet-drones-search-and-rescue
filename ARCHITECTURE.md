# Hierarchical Two-Stage RL Architecture

## Overview

This is a complete rewrite of the velocity control system into a **hierarchical two-stage RL architecture** with separated responsibilities and mission-oriented rewards.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: PATH PLANNING                          │
│                   (High-Level Decision Making)                      │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                             │
│    • 64x32 Stereo Vision Depth Map (min-pooled from 128x128)      │
│    • 3 Waypoint Sequence (relative positions)                      │
│    • Current State (pos, vel, accel, yaw rate)                     │
│    • Previous Command (for smoothness)                             │
│                                                                     │
│  MODEL: Transformer + MLP                                          │
│    • Vision Transformer for depth attention                        │
│    • 2071D observation space                                       │
│    • 10Hz planning frequency                                       │
│                                                                     │
│  OUTPUT: Desired Velocities [vx, vy, vz, ωz]                      │
│                                                                     │
│  REWARD: r₁ = δ(d_prev - d_curr) - c_v||v||₂ - c_a||a||₂ + obs    │
│    • δ = 1.0: Progress toward waypoint                             │
│    • c_v = 0.1: Velocity penalty (slower is better)                │
│    • c_a = 0.05: Acceleration penalty (smooth is better)           │
│    • obs: Exponential penalty for obstacles < 2m                   │
│    • γ = 0.99: Time discount (faster completion)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Velocity Commands
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: MOTOR CONTROL                           │
│                  (Low-Level Execution Layer)                        │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                             │
│    • Desired Velocities (from Stage 1)                             │
│    • Current Velocity & Angular Velocity                           │
│    • Orientation (RPY)                                             │
│    • Previous Motor Commands                                       │
│                                                                     │
│  MODEL: LSTM/RNN                                                   │
│    • Short-term memory for smooth control                          │
│    • 17D observation space                                         │
│    • 30Hz control frequency                                        │
│                                                                     │
│  OUTPUT: Motor RPM Commands [motor1, motor2, motor3, motor4]      │
│                                                                     │
│  REWARD: r₂ = -Σ wᵢ(vᵢ,actual - vᵢ,desired)²                      │
│    • Negative Quadratic Tracking Error                             │
│    • Pure compliance - no bonuses                                  │
│    • Heavily penalizes deviation                                   │
│    • w_vx = w_vy = w_vz = w_ωz = 1.0                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

### Stage 1: Path Planning
✅ **Stereo Vision Processing**
  - Input: 128x128 full-res depth map
  - Output: 64x32 min-pooled (preserves closest obstacles)
  - 96.9% memory savings, 32x compression
  - Wide format matches stereo aspect ratio

✅ **Waypoint Navigation**
  - Looks ahead 3 waypoints
  - Relative positioning for better generalization
  - 0.5m reach radius
  - +10 reward bonus for reaching waypoint

✅ **Obstacle Avoidance**
  - Exponential penalty: exp(-2*distance) for obstacles < 2m
  - Encourages maintaining safe clearance
  - Uses minimum distance from depth map

✅ **Smooth Flight**
  - Velocity penalty: encourages slower, controlled flight
  - Acceleration penalty: encourages smooth transitions
  - Previous command in observation for continuity

### Stage 2: Motor Control
✅ **Negative Quadratic Reward**
  - Formula: r₂ = -(error_vx² + error_vy² + error_vz² + error_ωz²)
  - Pure tracking - no complex shaping
  - Always ≤ 0 (penalty-based)

✅ **LSTM Architecture**
  - Short-term memory for temporal consistency
  - Remembers previous commands
  - Prevents oscillations

✅ **Tight Tracking**
  - Quadratic penalty heavily penalizes large errors
  - Small errors get small penalties
  - Encourages precise control

## Implementation Files

### Core Environments
- **`training_env_motor_control.py`**: Stage 2 environment (LSTM policy)
- **`training_env_path_planning.py`**: Stage 1 environment (Transformer policy)

### Vision System
- **`stereo_vision.py`**: Stereo camera system (128x128 capture)
- **`depth_downsampler.py`**: Min-pooling downsampler (→ 64x32)

### Training Scripts
- **`train_motor_control.py`**: Train Stage 2 with LSTM
- **`train_path_planning.py`**: Train Stage 1 with Transformer (TODO)

## Training Pipeline

### Phase 1: Train Motor Controller (Stage 2)
```bash
python train_motor_control.py --timesteps 500000 --name motor_control_v1
```

**Goal**: Learn tight velocity tracking
**Expected**: Near-zero tracking error after ~500k steps
**Freeze**: After convergence, save and freeze for Stage 1

### Phase 2: Train Path Planner (Stage 1)
```bash
python train_path_planning.py --timesteps 1000000 --name path_planning_v1 --motor_controller ./models/motor_control_v1/best_model.zip
```

**Goal**: Learn to navigate waypoints with obstacle avoidance
**Uses**: Frozen Stage 2 as execution layer
**Expected**: Successful waypoint navigation after ~1M steps

## Reward Design Philosophy

### Stage 1 (Mission-Oriented)
- **Episode-based**: Reward accumulated over full trajectory
- **Goal-directed**: Progress toward waypoints
- **Safety-aware**: Obstacle avoidance bonus
- **Efficiency-focused**: Time discount + velocity penalty
- **Smooth planning**: Acceleration penalty

### Stage 2 (Compliance-Oriented)
- **Step-based**: Immediate tracking accuracy
- **Command-following**: Pure compliance with input
- **No mission knowledge**: Just tracks velocities
- **Tight control**: Quadratic penalty for precision

## Key Advantages

1. **Separation of Concerns**
   - Planning doesn't worry about motor physics
   - Control doesn't worry about navigation

2. **Scalability**
   - Can improve Stage 1 without retraining Stage 2
   - Can train different planners with same controller

3. **Realism**
   - Mimics real drone systems (path planner + autopilot)
   - Allows testing control separately

4. **Efficiency**
   - Stage 1 runs at 10Hz (planning is slower)
   - Stage 2 runs at 30Hz (control is faster)
   - Min-pooling keeps vision input small

## Current Status

✅ **COMPLETED:**
- Stage 2 motor control environment
- Stage 1 path planning environment
- Stereo vision system integration
- 64x32 min-pooling downsampler
- Obstacle attenuation reward
- Training script for Stage 2

🚧 **TODO:**
- Train Stage 2 to convergence
- Create Transformer policy for Stage 1
- Integrate frozen Stage 2 into Stage 1
- Train hierarchical system end-to-end
- Add obstacles to PyBullet environment
- Tune hyperparameters

## Testing

### Test Stage 2:
```bash
python training_env_motor_control.py
```

Expected output:
- Negative rewards (tracking errors)
- Fast termination with random actions
- ~-1.27 reward for 1 m/s tracking error

### Test Stage 1:
```bash
python training_env_path_planning.py
```

Expected output:
- 2071D observation space
- 3 waypoints generated
- Depth map captured (64x32)
- Mission progress tracking

### Test Vision:
```bash
python depth_downsampler.py
```

Expected output:
- 32x compression (256x256 → 64x32)
- 0.000m obstacle preservation error
- 96.9% memory savings

## Next Steps

1. **Train Stage 2**: Run motor control training for 500k steps
2. **Validate Control**: Test tracking accuracy with trained model
3. **Build Transformer**: Implement Vision Transformer for Stage 1
4. **Hierarchical Training**: Connect stages and train path planner
5. **Add Obstacles**: Place obstacles in PyBullet for realistic testing
6. **Tune Rewards**: Adjust δ, c_v, c_a based on behavior

## Configuration

### Reward Hyperparameters
```python
# Stage 1
delta = 1.0           # Progress weight
c_v = 0.1             # Velocity penalty
c_a = 0.05            # Acceleration penalty
obstacle_threshold = 2.0  # Distance for obstacle penalty
gamma = 0.99          # Time discount

# Stage 2
w_vx = w_vy = w_vz = w_omega_z = 1.0  # Equal tracking weights
```

### Vision Configuration
```python
# Stereo cameras
baseline = 0.1m       # 10cm between cameras
fov = 90°             # Wide field of view
input_res = 128x128   # High-res capture
output_res = 64x32    # Min-pooled output
```

### Training Configuration
```python
# Stage 2
ctrl_freq = 30Hz      # Control frequency
max_steps = 1000      # ~33 seconds per episode
policy = "MlpLstm"    # LSTM for temporal consistency

# Stage 1
planning_freq = 10Hz  # Planning frequency
max_steps = 500       # ~50 seconds per episode
policy = "Transformer" # ViT for vision attention
```

## Architecture Rationale

### Why Two Stages?
- **Real drone systems** separate planning and control
- **Training efficiency**: Easier to debug/improve each layer
- **Generalization**: Same controller works with different planners
- **Computational**: Planning is expensive, control is fast

### Why Transformer for Stage 1?
- **Attention mechanism**: Focuses on relevant obstacles
- **Spatial reasoning**: Processes depth map structure
- **Long-range dependencies**: Plans multi-step paths

### Why LSTM for Stage 2?
- **Temporal consistency**: Smooth motor commands
- **Memory**: Remembers recent control history
- **Proven**: Works well for control tasks

### Why Negative Quadratic?
- **Precision**: Heavily penalizes large tracking errors
- **Simplicity**: No complex reward shaping
- **Compliance**: Pure command-following behavior
- **Gradient**: Strong signal for learning

### Why Min Pooling?
- **Safety-critical**: Closest obstacle is most important
- **Collision avoidance**: Preserves danger information
- **Compression**: 32x smaller while keeping safety data
- **Speed**: Fast downsampling operation

## Comparison to Previous System

### Old System (Velocity Control v1-v26)
- ❌ Single-stage: Tangled planning + control
- ❌ Step-based rewards: No mission awareness
- ❌ Complex shaping: Upright + velocity + yaw bonuses
- ❌ Exploit-prone: Upside-down tricks
- ❌ No vision: Blind navigation

### New System (Hierarchical)
- ✅ Two-stage: Clean separation
- ✅ Episode-based (S1): Mission-oriented
- ✅ Simple rewards: Clear objectives
- ✅ Exploit-proof: Stage 2 only cares about tracking
- ✅ Vision-based: Sees obstacles

## Future Extensions

1. **Multi-goal**: Extend to N waypoints (search patterns)
2. **Dynamic obstacles**: Moving targets/threats
3. **Energy awareness**: Battery consumption in reward
4. **Communication**: Multi-drone coordination
5. **Semantic vision**: Object detection instead of depth
6. **Meta-learning**: Adapt to new environments quickly

---

**Ready to train!** Start with Stage 2 motor control.
