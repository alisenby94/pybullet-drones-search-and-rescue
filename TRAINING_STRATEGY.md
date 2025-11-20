# Hierarchical Training Strategy

## Overview

Training a hierarchical RL system requires careful sequencing because **Stage 1 depends on Stage 2's performance**. If Stage 2 (motor control) doesn't accurately execute velocity commands, Stage 1 (path planning) will receive noisy/incorrect reward signals and fail to learn.

## The Dependency Problem

```
Stage 1 (Path Planner)
   ↓ Commands: "Go forward at 1.0 m/s"
Stage 2 (Motor Control)
   ↓ Execution: Actually goes 0.3 m/s (POOR TRACKING!)
Environment
   ↓ Outcome: Drone doesn't reach waypoint
Stage 1 Reward
   ↓ Signal: Negative reward
Stage 1 Learning
   → "Going forward is bad!" ❌ WRONG LESSON!
```

**The Problem**: Stage 1 thinks its planning is bad, when actually Stage 2's execution is bad.

**The Solution**: Train Stage 2 first until it's reliable, THEN train Stage 1.

## 📋 Three-Phase Training Pipeline

### **Phase 1: Train Stage 2 in Isolation** ⚡ CRITICAL

**Goal**: Achieve reliable velocity tracking (< 0.1 m/s error)

**Duration**: ~500k timesteps (varies by complexity)

**Command**: 
```bash
python train_motor_control.py --timesteps 500000 --name motor_control_v1
```

**Training Curriculum**:

| Episodes | Command Type | Velocity Scale | Change Interval | Difficulty |
|----------|--------------|----------------|-----------------|------------|
| 0-100k | **Static** | 30% (0.6 m/s) | Never | ⭐ Easy |
| 100k-300k | **Slow Changing** | 50% (1.0 m/s) | Every 6.7s | ⭐⭐ Medium |
| 300k+ | **Fast Changing** | 100% (2.0 m/s) | Every 1.7s | ⭐⭐⭐ Hard |

**Why Random Commands?**
- ✅ No dependency on Stage 1 (can start immediately)
- ✅ Diverse training data (all directions, all speeds)
- ✅ Prepares for any command Stage 1 might send
- ✅ Easy to validate (measure tracking error)

**Curriculum Benefits**:
1. **Static commands** (Phase 1): Learn basic control without adaptation pressure
2. **Slow changes** (Phase 2): Learn to adapt to new commands smoothly
3. **Fast changes** (Phase 3): Learn quick response for agile flight

**Monitoring Progress**:
```bash
# Watch TensorBoard
tensorboard --logdir ./logs/motor_control_v1

# Key metrics to track:
# - Mean reward (should approach 0 from negative)
# - Episode length (should stay high, no crashes)
# - Tracking error (should decrease to < 0.1 m/s)
```

**Expected Learning Curve**:
```
Episodes 0-10k:    Random flailing, frequent crashes
Episodes 10k-50k:  Basic stabilization, reducing oscillations  
Episodes 50k-150k: Decent tracking, occasional overshoot
Episodes 150k-300k: Good tracking, smooth transitions
Episodes 300k+:    Excellent tracking, quick response
```

**Acceptance Criteria** (MUST pass before Phase 2):
- [ ] Mean tracking error < 0.1 m/s over 100 episodes
- [ ] No crashes during hover commands
- [ ] Responds to command changes within 5 steps (0.17s)
- [ ] Smooth motor commands (no oscillations)
- [ ] Stable across full velocity range

### **Phase 2: Validate Stage 2** ✅

**Goal**: Ensure Stage 2 is production-ready before integration

**Duration**: 1-2 hours of testing

**Tests to Run**:

**Test 1: Static Command Tracking**
```bash
python test_motor_control.py --model motor_control_v1 --test static
# Expected: Error < 0.05 m/s on hover, forward, backward, strafe
```

**Test 2: Dynamic Command Response**
```bash
python test_motor_control.py --model motor_control_v1 --test dynamic
# Expected: Smooth transitions, no overshoot, < 5 step settle time
```

**Test 3: Stress Test**
```bash
python test_motor_control.py --model motor_control_v1 --test stress
# Expected: No crashes during rapid command changes
```

**Test 4: Visualization**
```bash
python visualize_motor_control.py --model motor_control_v1
# Expected: Smooth flight, tight tracking visible
```

**Validation Checklist**:
- [ ] Hover test: ≤ 0.05 m/s drift
- [ ] Forward test: ≤ 0.08 m/s tracking error
- [ ] Strafe test: ≤ 0.10 m/s tracking error
- [ ] Ascend/descend: ≤ 0.08 m/s tracking error
- [ ] Yaw rotation: ≤ 0.05 rad/s tracking error
- [ ] Command transition: Smooth (no spikes)
- [ ] 100 consecutive episodes: 0 crashes

**If validation fails**: 
- Retrain Stage 2 with adjusted hyperparameters
- Check for bugs in reward computation
- Verify observation normalization
- DO NOT proceed to Phase 3

### **Phase 3: Train Stage 1 with Frozen Stage 2** 🎯

**Goal**: Learn path planning with reliable motor control

**Duration**: ~1M timesteps (longer than Stage 2)

**Prerequisites**:
- ✅ Stage 2 trained and validated
- ✅ Stage 2 weights frozen
- ✅ Integration code tested

**Setup**:
```python
# Load and freeze Stage 2
motor_controller = PPO.load("./models/motor_control_v1/best_model.zip")
motor_controller.policy.eval()
for param in motor_controller.policy.parameters():
    param.requires_grad = False

# Create Stage 1 environment with frozen Stage 2
env = PathPlanningAviary(motor_controller=motor_controller)
```

**Command**:
```bash
python train_path_planning.py \
    --timesteps 1000000 \
    --name path_planning_v1 \
    --motor_controller ./models/motor_control_v1/best_model.zip
```

**Training Curriculum**:

| Episodes | Waypoints | Distance | Obstacles | Difficulty |
|----------|-----------|----------|-----------|------------|
| 0-50k | 2-3 | 2-3m apart | None | ⭐ Easy |
| 50k-150k | 3-5 | 5-7m apart | Static (walls) | ⭐⭐ Medium |
| 150k-300k | 5-8 | 7-10m apart | Static (boxes) | ⭐⭐⭐ Hard |
| 300k+ | 8-12 | Random | Dynamic (moving) | ⭐⭐⭐⭐ Expert |

**Why This Order?**
1. **Short paths first**: Learn basic waypoint navigation
2. **Add distance**: Learn longer-term planning
3. **Add obstacles**: Learn avoidance behavior
4. **Add dynamics**: Learn adaptive planning

**Monitoring Progress**:
```bash
tensorboard --logdir ./logs/path_planning_v1

# Key metrics:
# - Waypoints reached per episode (should increase)
# - Episode length (should optimize - not too slow)
# - Min obstacle distance (should stay > 1m)
# - Mean reward (should increase)
```

**Expected Behaviors**:

**Early (0-50k episodes)**:
- Random exploration
- Occasional waypoint reaches (luck)
- Frequent collisions
- Poor velocity planning

**Middle (50k-200k episodes)**:
- Consistent waypoint reaching
- Better obstacle avoidance
- Smoother velocity profiles
- Occasional stuck behavior

**Late (200k+ episodes)**:
- Efficient paths
- Proactive obstacle avoidance
- Optimized velocity use
- Rare failures

## 🔄 Integration Architecture

```python
# Stage 1 step (10Hz planning)
def stage1_step(obs):
    # Stage 1 observes environment
    vision = get_depth_map()  # 64x32
    waypoints = get_waypoints()  # 3 targets
    state = get_state()  # pos, vel, accel
    
    # Stage 1 decides velocities
    desired_vel = stage1_policy(vision, waypoints, state)
    # → [vx, vy, vz, ωz]
    
    # Pass to Stage 2
    return desired_vel

# Stage 2 step (30Hz control)
def stage2_step(desired_vel):
    # Stage 2 observes state
    actual_vel = get_velocity()
    orientation = get_orientation()
    prev_action = get_previous_action()
    
    # Stage 2 computes motor commands
    motor_rpms = stage2_policy(desired_vel, actual_vel, orientation, prev_action)
    # → [rpm1, rpm2, rpm3, rpm4]
    
    # Execute motors
    execute(motor_rpms)
    
    # Compute Stage 2 reward (tracking error)
    stage2_reward = -sum((actual_vel - desired_vel)^2)
    return stage2_reward

# Full hierarchical loop (every 0.1s)
for planning_step in range(500):  # Stage 1 episode
    desired_vel = stage1_step(obs)
    
    # Stage 2 executes for 3 control steps (10Hz→30Hz)
    for control_step in range(3):
        stage2_reward = stage2_step(desired_vel)
        # Note: Stage 2 reward NOT used for Stage 1 training
    
    # Stage 1 gets environment outcome
    new_pos = get_position()
    stage1_reward = compute_stage1_reward(new_pos, waypoints, obstacles)
    
    # Only Stage 1 learns (Stage 2 frozen)
    stage1_policy.update(stage1_reward)
```

## 🛡️ Preventing Common Pitfalls

### Pitfall 1: Training Stage 1 Too Early
**Symptom**: Stage 1 reward never improves, random behavior
**Cause**: Stage 2 not reliable yet
**Fix**: Validate Stage 2 thoroughly before Phase 3

### Pitfall 2: Forgetting to Freeze Stage 2
**Symptom**: Both stages learning simultaneously, unstable training
**Cause**: Stage 2 weights not frozen
**Fix**: Add `requires_grad=False` to all Stage 2 parameters

### Pitfall 3: Mismatched Frequencies
**Symptom**: Jerky motion, Stage 1 doesn't see smooth execution
**Cause**: Stage 1 and Stage 2 running at same frequency
**Fix**: Stage 1 @ 10Hz, Stage 2 @ 30Hz (3:1 ratio)

### Pitfall 4: Reward Leakage
**Symptom**: Stage 1 optimizes for Stage 2's reward instead of mission
**Cause**: Using Stage 2's tracking reward for Stage 1 updates
**Fix**: Keep rewards completely separate

### Pitfall 5: Insufficient Stage 2 Training
**Symptom**: Stage 1 learns, but performance plateaus early
**Cause**: Stage 2 still has 0.3-0.5 m/s tracking error
**Fix**: Retrain Stage 2 until < 0.1 m/s error

## 📊 Performance Metrics

### Stage 2 (Motor Control)
| Metric | Target | Good | Acceptable | Poor |
|--------|--------|------|------------|------|
| Hover error | < 0.05 m/s | < 0.10 m/s | < 0.20 m/s | > 0.20 m/s |
| Tracking error | < 0.08 m/s | < 0.15 m/s | < 0.30 m/s | > 0.30 m/s |
| Response time | < 3 steps | < 5 steps | < 10 steps | > 10 steps |
| Crash rate | 0% | < 1% | < 5% | > 5% |

### Stage 1 (Path Planning)
| Metric | Target | Good | Acceptable | Poor |
|--------|--------|------|------------|------|
| Waypoint success | > 95% | > 80% | > 60% | < 60% |
| Min obstacle dist | > 1.5m | > 1.0m | > 0.5m | < 0.5m |
| Path efficiency | > 90% | > 75% | > 60% | < 60% |
| Episode completion | > 90% | > 75% | > 50% | < 50% |

## 🧪 Testing Protocol

### After Phase 1 (Stage 2 Complete)
```bash
# Run full validation suite
./scripts/validate_stage2.sh motor_control_v1

# Manual inspection
python visualize_motor_control.py --model motor_control_v1 \
    --commands hover,forward,circle,square,random

# Performance benchmark  
python benchmark_motor_control.py --model motor_control_v1 \
    --episodes 1000 --report tracking_errors.csv
```

### After Phase 3 (Stage 1 Complete)
```bash
# Test waypoint navigation
python test_path_planning.py --model path_planning_v1 \
    --waypoints 5 --obstacles static

# Test obstacle avoidance
python test_path_planning.py --model path_planning_v1 \
    --waypoints 3 --obstacles dense

# Full mission test
python test_full_mission.py \
    --stage1 path_planning_v1 \
    --stage2 motor_control_v1 \
    --scenario search_and_rescue
```

## 🎯 Success Criteria

### Minimum Viable System
- [x] Stage 2 trains without crashes
- [x] Stage 2 tracks static commands (< 0.15 m/s error)
- [ ] Stage 2 passes validation tests
- [ ] Stage 1 trains with frozen Stage 2
- [ ] Stage 1 reaches 80%+ waypoints
- [ ] Hierarchical system completes simple missions

### Production Ready System
- [ ] Stage 2 error < 0.08 m/s on all commands
- [ ] Stage 2 responds in < 3 control steps
- [ ] Stage 1 reaches 95%+ waypoints
- [ ] Stage 1 maintains > 1.0m obstacle clearance
- [ ] System completes complex missions reliably
- [ ] Real-time performance (< 100ms per decision)

## 📚 Key Takeaways

1. **Sequential Training is Essential**: Stage 2 → Validation → Stage 1
2. **Random Commands Work**: No need for Stage 1 during Stage 2 training
3. **Curriculum Learning Helps**: Start easy, gradually increase difficulty
4. **Validation is Critical**: Don't skip testing between phases
5. **Freezing Prevents Instability**: Lock Stage 2 during Stage 1 training
6. **Separate Rewards**: Each stage optimizes its own objective
7. **Different Frequencies**: 10Hz planning, 30Hz control
8. **Patience Pays Off**: Stage 2 may need 500k+ steps to converge

## 🚀 Quick Start

```bash
# Phase 1: Train motor control (8-12 hours)
python train_motor_control.py --timesteps 500000 --name motor_v1

# Phase 2: Validate (1 hour)
python test_motor_control.py --model motor_v1 --full-suite

# Phase 3: Train path planning (24-48 hours)
python train_path_planning.py --timesteps 1000000 \
    --name planning_v1 --motor_controller ./models/motor_v1/best_model.zip

# Deploy!
python run_mission.py --planning planning_v1 --motor motor_v1
```

---

**Ready to train!** Start with Phase 1: `python train_motor_control.py`
