# Reward Structure Documentation

## Philosophy

The reward system balances multiple objectives to produce safe, efficient, and precise drone navigation:

1. **Sparse rewards** for high-level goals (waypoint completion)
2. **Shaped rewards** for trajectory quality (alignment, stability)
3. **Distance-scaled rewards** that adapt based on proximity to target
4. **Multi-objective balance** where no single component dominates

## Core Reward Components

### 1. Survival Bonus
**Value:** +1.0 per timestep  
**Purpose:** Encourage staying alive and continuing the mission

**Behavior:**
- Constant reward for each step the drone remains operational
- Encourages learning behaviors that avoid crashes
- Provides baseline positive reinforcement

**Example States:**
- Flying safely: +1.0 per step
- About to crash: Still +1.0, but termination imminent
- Hovering: +1.0 per step (survival matters even when stationary)

---

### 2. Progress Reward
**Value:** 10.0 × (distance_decreased)  
**Purpose:** Reward moving closer to the current waypoint

**Behavior:**
- Positive when moving toward waypoint
- Negative when moving away
- Magnitude scales with how much closer/farther per step

**Example States:**
```
Distance 10m → 9.9m (moved 0.1m closer):
  reward = 10.0 × 0.1 = +1.0

Distance 5m → 4.8m (moved 0.2m closer):
  reward = 10.0 × 0.2 = +2.0

Distance 3m → 3.1m (moved 0.1m farther):
  reward = 10.0 × (-0.1) = -1.0
```

---

### 3. Waypoint Completion
**Value:** +200.0  
**Purpose:** Major reward for reaching waypoint (within 1.0m radius)

**Behavior:**
- Large sparse reward that dominates other components
- Signals successful navigation milestone
- Given once per waypoint when entering radius
- Sized to dominate crash penalties (ensures positive learning signal)

**Example States:**
```
Distance 1.2m → 0.9m (entered waypoint):
  reward = +200.0 (plus other components)

Distance 0.5m (already in waypoint):
  reward = 0.0 (only given once on entry)
```

---

### 4. Roll Stability
**Formula:** `(1 / cosh(4 × roll)) - 1`  
**Weight:** 10.0  
**Purpose:** CRITICAL - Must learn stable flight before navigation

**Behavior:**
- Maximum reward (0.0) when perfectly level
- Strong penalty that increases with roll angle
- Dominates other shaping rewards to prioritize stability
- Hyperbolic secant provides smooth gradient for learning

**Example States:**
```
Roll = 0° (perfectly level):
  reward = (1 / cosh(0)) - 1 × 10.0 = 0.0
  → Perfect stability, baseline for learning

Roll = ±10° (slight bank):
  reward = (1 / cosh(0.7)) - 1 × 10.0 ≈ -2.0
  → Notable penalty, discourages tilting

Roll = ±20° (moderate bank):
  reward = (1 / cosh(1.4)) - 1 × 10.0 ≈ -5.3
  → Strong penalty, must correct

Roll = ±30° (steep bank):
  reward = (1 / cosh(2.1)) - 1 × 10.0 ≈ -7.6
  → Very strong penalty, approaching crash
```

---

### 5. Pitch Stability
**Formula:** `((tanh(12π × pitch + 5) - 3) / 2 + 1/cosh(2 × pitch))`  
**Weight:** 10.0  
**Purpose:** CRITICAL - Asymmetric penalty emphasizing backward tilt is dangerous

**Behavior:**
- Allows/encourages slight forward tilt for forward flight
- Heavily penalizes backward tilt (dangerous for control)
- Dominates other shaping rewards to prioritize stability
- Combines hyperbolic tangent (asymmetry) with secant (smooth penalty)

**Example States:**
```
Pitch = 0° (level):
  reward ≈ 0.0
  → Perfect stability, baseline

Pitch = -10° (nose up, slight forward tilt):
  reward ≈ -0.6 (minor penalty, acceptable for forward flight)

Pitch = +10° (nose down, backward tilt):
  reward ≈ -0.6 (stronger penalty, discouraged)

Pitch = +20° (severe backward tilt):
  reward ≈ -2.0 (strong penalty, must correct)

Pitch = +30° (extreme backward tilt):
  reward ≈ -3.8 (very strong penalty, approaching crash)
```

---

### 6. Distance-Scaled Altitude Control
**Formula:** `-1.5 × e^(-dist/2.5) × |altitude_error|`  
**Purpose:** Altitude precision matters MORE as drone approaches waypoint

**Behavior:**
- Far from target: minimal altitude penalty (freedom to explore)
- Close to target: strong altitude penalty (precision landing)
- Smooth exponential scaling provides wide effective range

**Example States (0.5m altitude error):**
```
Distance = 5.0m (far):
  scale = 1.0 × e^(-2.0) = 0.135
  reward = -1.5 × 0.135 × 0.5 = -0.10
  → Minor penalty, altitude flexibility

Distance = 3.0m (approaching):
  scale = 1.0 × e^(-1.2) = 0.301
  reward = -1.5 × 0.301 × 0.5 = -0.23
  → Starting to care about altitude

Distance = 1.0m (close):
  scale = 1.0 × e^(-0.4) = 0.670
  reward = -1.5 × 0.670 × 0.5 = -0.50
  → Strong incentive for altitude matching

Distance = 0.5m (very close):
  scale = 1.0 × e^(-0.2) = 0.819
  reward = -1.5 × 0.819 × 0.5 = -0.61
  → Critical precision required
```

---

### 7. Forward Velocity Reward
**Formula:** `1.0 × 2.0 × tanh(vel_forward / 2.0) × tanh(dist / 2.0)`  
**Purpose:** Encourage forward movement when far, neutral when close

**Behavior:**
- Far from target: reward speed (get there quickly)
- Close to target: reward approaches zero (hovering is better)
- Saturating function prevents rewarding excessive speed

**Example States:**
```
Distance = 5.0m, Speed = 2.0 m/s forward:
  distance_factor = tanh(2.5) = 0.99
  speed_factor = tanh(1.0) = 0.76
  reward = 1.0 × 2.0 × 0.76 × 0.99 = +1.50
  → Strong reward for moving forward

Distance = 2.0m, Speed = 2.0 m/s forward:
  distance_factor = tanh(1.0) = 0.76
  reward = 1.0 × 2.0 × 0.76 × 0.76 = +1.16
  → Moderate reward

Distance = 1.0m, Speed = 2.0 m/s forward:
  distance_factor = tanh(0.5) = 0.46
  reward = 1.0 × 2.0 × 0.76 × 0.46 = +0.70
  → Low reward, prefer slowing down

Distance = 0.5m, Speed = 0.5 m/s forward:
  distance_factor = tanh(0.25) = 0.24
  speed_factor = tanh(0.25) = 0.24
  reward = 1.0 × 2.0 × 0.24 × 0.24 = +0.12
  → Minimal reward, hovering encouraged
```

---

### 8. Velocity Alignment Reward
**Formula:** `1.5 × (0.5 + 2.5 × e^(-dist/3.0)) × alignment`  
**Purpose:** MASSIVELY reward moving directly toward target, especially when close

**Behavior:**
- Starts mattering at 4-5m out (wide effective range)
- Grows exponentially as drone approaches
- Positive for aimed trajectories, negative for moving away
- Offsets speed penalties when well-aligned

**Key Feature:** Alignment = dot product of velocity direction with direction to waypoint
- +1.0 = moving directly toward target
- 0.0 = moving perpendicular
- -1.0 = moving directly away

**Example States (Speed = 2.0 m/s):**
```
Distance = 5.0m, Perfect alignment (+1.0):
  scale = 0.5 + 2.5 × e^(-1.67) = 0.98
  reward = 1.5 × 0.98 × 1.0 = +1.46
  → Good early signal for trajectory

Distance = 3.0m, Perfect alignment (+1.0):
  scale = 0.5 + 2.5 × e^(-1.0) = 1.42
  reward = 1.5 × 1.42 × 1.0 = +2.13
  → Strong reward for aimed approach

Distance = 1.0m, Perfect alignment (+1.0):
  scale = 0.5 + 2.5 × e^(-0.33) = 2.29
  reward = 1.5 × 2.29 × 1.0 = +3.44
  → Very strong precision incentive

Distance = 1.0m, Perpendicular (0.0):
  reward = 1.5 × 2.29 × 0.0 = 0.0
  → Neutral, not encouraged

Distance = 1.0m, Moving away (-1.0):
  reward = 1.5 × 2.29 × (-1.0) = -3.44
  → Strong penalty for wrong direction
```

---

### 9. Misalignment Speed Penalty
**Formula:** `-1.0 × e^(-dist/2.5) × misalignment_factor × (speed² / safe_speed²)`  
**Purpose:** Penalize high speeds ONLY when misaligned and close

**Behavior:**
- Well-aimed + fast = minimal penalty (efficient)
- Poorly-aimed + fast = strong penalty (dangerous)
- Far from target = minimal penalty (exploration freedom)
- Misalignment factor = (1 - alignment) / 2

**Example States (Speed = 2.0 m/s):**
```
Distance = 5.0m, Perfect alignment (+1.0):
  proximity = e^(-2.0) = 0.135
  misalignment = (1 - 1.0) / 2 = 0.0
  penalty = -1.0 × 0.135 × 0.0 × 1.0 = 0.0
  → No penalty, fast approach OK

Distance = 2.0m, Neutral alignment (0.0):
  proximity = e^(-0.8) = 0.449
  misalignment = (1 - 0.0) / 2 = 0.5
  penalty = -1.0 × 0.449 × 0.5 × 1.0 = -0.22
  → Moderate penalty, slow down or aim better

Distance = 1.0m, Moving away (-1.0):
  proximity = e^(-0.4) = 0.670
  misalignment = (1 - (-1.0)) / 2 = 1.0
  penalty = -1.0 × 0.670 × 1.0 × 1.0 = -0.67
  → Strong penalty, wrong direction!

Distance = 0.5m, Neutral alignment (0.0):
  proximity = e^(-0.2) = 0.819
  misalignment = 0.5
  penalty = -1.0 × 0.819 × 0.5 × 1.0 = -0.41
  → Strong penalty even when close, precision critical
```

---

### 10. Action Smoothness Penalty
**Formula:** `-0.5 × ||action_t - action_{t-1}||`  
**Purpose:** Discourage jerky control, encourage smooth trajectories

**Behavior:**
- Penalizes large action changes between timesteps
- Promotes stable, efficient flight
- Prevents oscillations and energy waste

**Example States:**
```
Action change = [0.0, 0.0, 0.0, 0.0] (no change):
  magnitude = 0.0
  penalty = -0.5 × 0.0 = 0.0
  → Smooth control, no penalty

Action change = [0.1, -0.1, 0.05, -0.05] (small adjustment):
  magnitude = 0.158
  penalty = -0.5 × 0.158 = -0.08
  → Minor penalty for minor change

Action change = [0.5, -0.5, 0.3, -0.3] (large adjustment):
  magnitude = 0.83
  penalty = -0.5 × 0.83 = -0.42
  → Significant penalty for jerky control
```

---

### 11. Hover Reward (Final Waypoint Only)
**Formula:** `50.0 × e^(-dist² / (2 × radius²))`  
**Purpose:** Massive sustained reward for hovering at final waypoint

**Behavior:**
- Only active on the last waypoint
- Exponential falloff encourages staying very close
- Sustained reward for remaining in position
- Intermediate waypoints: no hover reward (pass through quickly)

**Example States (Final waypoint only):**
```
Distance = 0.1m (very close):
  reward = 50.0 × e^(-0.01 / 2.0) = +49.75
  → Massive reward, stay here!

Distance = 0.5m (at edge):
  reward = 50.0 × e^(-0.25 / 2.0) = +44.06
  → Strong reward, get closer

Distance = 1.0m (waypoint radius):
  reward = 50.0 × e^(-1.0 / 2.0) = +30.33
  → Moderate reward, approach

Distance = 2.0m (outside radius):
  reward = 50.0 × e^(-4.0 / 2.0) = +6.77
  → Small reward, move in

Intermediate waypoint (any distance):
  reward = 0.0
  → No hover reward, keep moving
```

---

### 12. Crash Penalty
**Value:** -50.0 (standard), -75.0 (obstacle collision)  
**Purpose:** Negative signal for termination, but sized to preserve positive learning signal

**Behavior:**
- Applied only on episode termination
- Extra penalty for obstacle collisions
- Deliberately smaller than waypoint reward (200.0) to ensure net positive learning
- Even at 50% success rate, net reward is +150.0 per waypoint

**Example States:**
```
Ground crash (altitude < 0.1m):
  penalty = -50.0
  → Learn to maintain altitude

Obstacle collision (PyBullet contact detected):
  penalty = -75.0
  → Learn to avoid obstacles

Normal flight:
  penalty = 0.0
  → No penalty when flying safely
```

**Learning Signal Analysis:**
```
Success rate 50% (1 crash per waypoint):
  Net = +200 + 1×(-50) = +150.0 → Strong positive signal

Success rate 33% (2 crashes per waypoint):
  Net = +200 + 2×(-50) = +100.0 → Good positive signal

Success rate 25% (3 crashes per waypoint):
  Net = +200 + 3×(-50) = +50.0 → Still learning

Must crash >4× per waypoint for negative signal
```

---

## Reward Magnitude Comparison

### Typical Per-Step Rewards (Mid-Flight)
```
Component                    Value Range
─────────────────────────────────────────
Roll/pitch stability         -15.0 to 0.0  ← DOMINANT
Survival                     +1.0
Progress (moving 0.1m)       +1.0
Forward velocity             +0.5 to +1.5
Velocity alignment           +1.5 to +4.0
Altitude penalty             -0.6 to -0.1
Speed penalty                -0.7 to 0.0
Smoothness penalty           -0.4 to 0.0
─────────────────────────────────────────
Typical total (level)        +1.0 to +5.0
Typical total (tilted 20°)   -6.0 to -2.0
Typical total (tilted 30°)   -10.0 to -6.0
```

### Milestone Rewards
```
Event                        Value
────────────────────────────────────
Waypoint reached             +200.0
Hover at final (sustained)   +50.0/step
Crash                        -50.0
Obstacle collision           -75.0
```

---

## Scenario Examples

### Scenario 1: Long-Range Approach
**State:** 8m from waypoint, moving at 2.5 m/s, perfectly aligned, level flight

**Reward Breakdown:**
```
Roll stability:         0.0  (level - CRITICAL baseline)
Pitch stability:        0.0  (level - CRITICAL baseline)
Survival:              +1.0
Progress (0.25m):      +2.5
Altitude:              -0.04 (distant, minimal penalty)
Forward velocity:      +1.5  (fast forward, far from target)
Alignment:             +1.2  (good aim, far from target)
Speed penalty:          0.0  (well-aligned, no penalty)
Smoothness:            -0.1  (small control change)
────────────────────────────
Total:                 +6.1  (efficient approach)
```

### Scenario 2: Final Approach
**State:** 1.5m from waypoint, moving at 1.0 m/s, well-aligned (0.8), slightly off altitude (0.3m)

**Reward Breakdown:**
```
Survival:              +1.0
Progress (0.1m):       +1.0
Roll stability:        -0.02 (slight tilt)
Pitch stability:       -0.01 (minor pitch)
Altitude:              -0.18 (close, altitude matters more)
Forward velocity:      +0.5  (slower near target)
Alignment:             +2.9  (good aim, close range)
Speed penalty:         -0.03 (slight misalignment penalty)
Smoothness:            -0.05 (smooth control)
────────────────────────────
Total:                 +5.1  (good final approach)
```

### Scenario 3: Waypoint Capture
**State:** 0.9m from waypoint (just entered), moving at 0.5 m/s, aligned

**Reward Breakdown:**
```
Survival:              +1.0
Progress (0.1m):       +1.0
Waypoint reached:      +200.0  ← Dominates!
Roll stability:         0.0
Pitch stability:        0.0
Altitude:              -0.2
Forward velocity:      +0.2
Alignment:             +3.5
Speed penalty:          0.0
Smoothness:            -0.03
────────────────────────────
Total:                 +205.5 (success!)
```

### Scenario 4: Hovering at Final Waypoint
**State:** 0.3m from final waypoint, hovering (0.1 m/s), aligned

**Reward Breakdown:**
```
Survival:              +1.0
Progress (0.0m):        0.0  (hovering)
Roll stability:         0.0
Pitch stability:        0.0
Altitude:              -0.1  (precise altitude)
Forward velocity:      +0.05 (minimal movement OK)
Alignment:             +0.0  (hovering, no direction)
Speed penalty:          0.0  (low speed)
Smoothness:            -0.02
Hover reward:          +47.5 ← Dominates at final WP!
────────────────────────────
Total:                 +48.4 (sustained hovering)
```

### Scenario 5: Poor Trajectory (Overshooting)
**State:** 0.8m from waypoint, moving at 2.5 m/s, moving away (-0.5 alignment)

**Reward Breakdown:**
```
Survival:              +1.0
Progress (-0.25m):     -2.5  (moving away!)
Roll stability:        -0.05 (banking)
Pitch stability:       -0.03
Altitude:              -0.44 (close + altitude error = bad)
Forward velocity:      +0.4  (speed less valuable close)
Alignment:             -1.8  (wrong direction!)
Speed penalty:         -1.2  (fast + misaligned + close!)
Smoothness:            -0.2  (aggressive maneuvering)
────────────────────────────
Total:                 -4.8  (learn to slow down!)
```

### Scenario 6: Obstacle Avoidance
**State:** 5m from waypoint, 0.4m from obstacle, moving at 1.5 m/s

**Reward Breakdown:**
```
Survival:              +1.0
Progress (0.15m):      +1.5
Roll stability:        -0.02
Pitch stability:       -0.01
Altitude:              -0.10 (far, minimal altitude penalty)
Forward velocity:      +1.3
Alignment:             +1.5
Speed penalty:         -0.05 (far + slight misalignment)
Smoothness:            -0.1
Obstacle proximity:    -2.5  ← Warning signal!
────────────────────────────
Total:                 +2.5  (learn to steer clear)
```

---

## Design Principles

### 1. Stability-First Hierarchy
**Roll/pitch stability dominates all shaping rewards.** The drone must learn to fly level before it can learn navigation. Stability rewards are 10× larger than other components, creating a clear learning priority:
1. **First:** Stay level (avoid -2 to -15 penalties)
2. **Second:** Move efficiently (gain +1 to +4 rewards)
3. **Third:** Reach waypoints (gain +200 milestones)

This hierarchy ensures the agent doesn't learn "reach waypoints by any means" but rather "fly stably AND reach waypoints."

### 2. Multi-Objective Balance (Within Navigation)
Within navigation rewards (excluding stability), no single component should dominate others. Navigation rewards are scaled to similar magnitudes (±0.5 to ±4.0 range).

### 2. Distance-Aware Scaling
Rewards adapt based on distance to target:
- **Far (>3m):** Freedom to explore, speed encouraged, altitude flexible
- **Medium (1-3m):** Trajectory matters, altitude guidance begins
- **Close (<1m):** Precision required, alignment critical, altitude exact

### 3. Complementary Signals
Rewards work together to shape behavior:
- Alignment rewards → offset speed penalties when aimed correctly
- Altitude scaling → complements approach trajectory
- Progress + waypoint → sparse goal, dense shaping

### 4. Smooth Gradients
All reward functions use smooth, differentiable functions (tanh, exp, cosh) to provide clear gradients for learning. No discontinuities or cliffs.

### 5. Interpretability
Each reward component has a clear physical meaning and expected behavior. Engineers can tune individual weights without breaking the system.

---

## Learning Progression

The reward structure is designed to enforce a natural learning progression:

**Phase 1: Stability (Steps 0-100k)**
- Agent learns to minimize roll/pitch penalties (-15 to 0)
- Flying level becomes the dominant behavior
- Crashes decrease as stability improves

**Phase 2: Movement (Steps 100k-500k)**
- Once stable, agent learns to move forward (+1.5 velocity reward)
- Progress rewards (+1.0 per 0.1m) reinforce approaching waypoints
- Alignment rewards (+1.5 to +4.0) shape trajectory

**Phase 3: Precision (Steps 500k-1M+)**
- Altitude control kicks in near waypoints (-0.6 penalty for errors)
- Speed modulation prevents overshooting (-0.7 penalty)
- Smooth control emerges (-0.4 penalty for jerky movements)

**Phase 4: Mastery (Steps 1M+)**
- Consistent waypoint reaching (+200 per waypoint)
- Curriculum learning: waypoint perturbations after 1M steps
- Hovering at final waypoint (+50 sustained)

---

## Tuning Guidelines

### If the drone is...

**Too cautious / slow:**
- Increase `forward_weight` (currently 1.0 → try 1.5)
- Decrease `speed_penalty_weight` (currently 1.0 → try 0.7)

**Overshooting waypoints:**
- Increase `alignment_weight` (currently 1.5 → try 2.0)
- Increase `speed_penalty_weight` (currently 1.0 → try 1.5)

**Poor altitude control:**
- Increase `altitude_weight` (currently 1.5 → try 2.0)
- Adjust altitude scaling range (currently 2.5 → try 2.0 for earlier engagement)

**Unstable / oscillating:**
- Increase `smoothness_weight` (currently 0.5 → try 1.0)
- Check roll/pitch weights (currently 1.0 each)

**Not hovering at final waypoint:**
- Increase `hover_weight` (currently 50.0 → try 75.0)
- Adjust hover radius falloff (currently uses waypoint_radius² = 1.0)

**Hitting obstacles:**
- Check obstacle proximity penalty implementation
- Increase obstacle avoidance reward weight
- Verify voxel grid collision detection

---

## Implementation Notes

All reward components are computed in `_computeReward()` method in `action_coordinator_env.py`. Key implementation details:

1. **State vector:** Position, orientation, velocities extracted from PyBullet
2. **Distance calculation:** Euclidean distance to current waypoint
3. **Alignment calculation:** Dot product of velocity unit vector with direction to waypoint
4. **Proximity factors:** Exponential scaling ensures smooth, wide-range influence
5. **Reward accumulation:** Components added sequentially, logged in info dict for debugging

### Logged Metrics (Available in TensorBoard)
- All individual reward components
- Alignment, speed, distance factors
- Altitude error, roll, pitch
- Progress, waypoint distance
- Obstacle proximity (if enabled)
