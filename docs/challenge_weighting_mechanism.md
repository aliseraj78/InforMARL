# Challenge Weighting Mechanism

## Overview

Challenge weighting is a reward amplification technique that dynamically adjusts the importance of training experiences based on their difficulty. More challenging situations receive higher reward weights, causing the agent to learn more from difficult experiences.

## How It Works

### 1. Challenge Assessment (Every Timestep)

At each step during training, the system evaluates multiple challenge metrics:

- **Distance to Goal**: How far the agent is from its target
- **Obstacle Density**: How close the agent is to obstacles
- **Collision Risk**: Proximity to other agents (collision potential)
- **Boundary Violation**: How close the agent is to environment boundaries
- **Multi-Agent Coordination**: Complexity of coordinating with other agents

### 2. Challenge Score Calculation

Each metric is normalized to [0, 1] and combined using weighted sum:

```
total_challenge = Σ(weight_i × metric_i)
```

Where each challenge type has a configurable weight:

- Distance Weight (default: 1.5)
- Obstacle Weight (default: 2.0)
- Collision Weight (default: 1.8)
- Coordination Weight (default: 1.5)

### 3. Reward Amplification

The total challenge score is normalized and converted to a reward multiplier:

```
normalized_challenge = (total_challenge - min_threshold) / (max_threshold - min_threshold)
normalized_challenge = clip(normalized_challenge, 0, 1)

reward_weight = 1.0 + (amplification_factor - 1.0) × normalized_challenge
```

**Default Parameters:**

- `min_threshold`: 0.1 (minimum challenge to start weighting)
- `max_threshold`: 8.0 (maximum challenge for normalization)
- `amplification_factor`: 2.0 (maximum reward multiplier)

### 4. Final Reward

```
weighted_reward = base_reward × reward_weight
```

## Example Scenarios

| Situation                               | Challenge Level  | Weight | Effect            |
| --------------------------------------- | ---------------- | ------ | ----------------- |
| Agent near goal, no obstacles           | Low (~0.2)       | ~1.0x  | Normal learning   |
| Agent far from goal, few obstacles      | Medium (~2.0)    | ~1.3x  | Moderate emphasis |
| Agent surrounded by obstacles           | High (~5.0)      | ~1.7x  | High emphasis     |
| Agent near boundary with collision risk | Very High (~7.0) | ~2.0x  | Maximum emphasis  |

## Key Benefits

1. **Focuses Learning**: Agents learn more from difficult situations
2. **Sample Efficiency**: Better utilization of challenging experiences
3. **Robust Behavior**: Improves performance in complex scenarios
4. **Adaptive**: Automatically identifies and emphasizes hard cases

## Configuration Parameters

When training with challenge weighting, use these flags:

```bash
--enable_challenge_weighting True
--challenge_amplification_factor 2.0
--challenge_min_threshold 0.1
--challenge_max_threshold 8.0
--challenge_distance_weight 1.5
--challenge_obstacle_weight 2.0
--challenge_collision_weight 1.8
--challenge_coordination_weight 1.5
```

## Implementation Flow

```
Training Step
    ↓
Environment Step (action → next state)
    ↓
Calculate Base Reward
    ↓
Assess Challenge Metrics
    ↓
Calculate Challenge Weight (1.0 to amplification_factor)
    ↓
Apply Weight: weighted_reward = base_reward × weight
    ↓
Store Experience in Buffer
    ↓
Policy Update (uses weighted rewards)
```

## Technical Details

**Location in Code:**

- Challenge assessment: `utils/challenge_assessment.py`
- Scenario integration: `multiagent/custom_scenarios/navigation_challenge_weighted.py`
- Training script: `onpolicy/scripts/train_mpe_challenge_weighted.py`

**When It Takes Effect:**

- Every timestep during training
- Applied at reward calculation time (before storing in buffer)
- Affects policy gradient updates through weighted rewards

## Tuning Guidelines

- **Increase `amplification_factor`** (e.g., 3.0): More emphasis on challenges
- **Decrease `min_threshold`** (e.g., 0.05): Start weighting earlier
- **Increase obstacle/collision weights**: Focus on navigation safety
- **Increase coordination weight**: Focus on multi-agent cooperation
