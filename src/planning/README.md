# Planning with Options: SMDP Value Iteration

This module implements planning with temporally extended actions (options) using **Synchronous Value Iteration (SVI)** with proper SMDP (Semi-Markov Decision Process) backups.

## 🎯 Overview

This module demonstrates that planning with options allows an agent to reason and learn at a higher temporal level, leading to:
- **Faster convergence** (fewer sweeps to reach optimal policy)
- **More efficient planning** (better value propagation per computational step)
- **Quicker learning** (useful behavior discovered faster)

## 🧩 Core Components

### Environment: Four-Rooms Gridworld

- **13×13 grid** divided into four connected rooms
- **Hallway positions** linking rooms:
  - (3, 6) - Right hallway of Room 1 → Room 2
  - (6, 2) - Bottom hallway of Room 1 → Room 3
  - (7, 9) - Bottom hallway of Room 2 → Room 4
  - (10, 6) - Right hallway of Room 3 → Room 4
- **Stochastic transitions**:
  - 2/3 probability: move in intended direction
  - 1/3 probability: move in one of the other 3 directions (1/9 each)
- **Goal-directed**: Reward of +1 for reaching goal state, 0 otherwise
- **Discount factor**: $\gamma = 0.99$

### Hallway Options

Eight pre-defined multi-step options, one per room-hallway pair:

| Option | Initiation Set | Target | Policy |
|--------|----------------|--------|--------|
| Room1→Right | Room 1 cells | (3, 6) | Navigate to right hallway |
| Room1→Bottom | Room 1 cells | (6, 2) | Navigate to bottom hallway |
| Room2→Left | Room 2 cells | (3, 6) | Navigate to left hallway |
| Room2→Bottom | Room 2 cells | (7, 9) | Navigate to bottom hallway |
| Room3→Top | Room 3 cells | (6, 2) | Navigate to top hallway |
| Room3→Right | Room 3 cells | (10, 6) | Navigate to right hallway |
| Room4→Top | Room 4 cells | (7, 9) | Navigate to top hallway |
| Room4→Left | Room 4 cells | (10, 6) | Navigate to left hallway |

Each option provides a **shortest-path policy** to its target hallway within its room.

## 🔬 SMDP Value Iteration

### Bellman Backup for Options

The planner uses the **fundamental matrix approach** for SMDP backups:

$$V(s) \leftarrow \max_{o \in \mathcal{O}(s)} \left[ r_o(s) + \sum_{s'} p_{ss'}^o V(s') \right]$$

Where:
- $r_o(s)$: Expected discounted reward = $\mathbb{E}\left[\sum_{t=0}^{\tau-1} \gamma^t r_t \mid s, o\right]$
- $p_{ss'}^o$: Expected discounted termination = $\mathbb{E}\left[\gamma^\tau \mathbb{1}(s_\tau = s') \mid s, o\right]$
- $\tau$: Random option duration (time until termination)

### Key Implementation Details

1. **Fundamental Matrix**: $N = (I - \gamma P_{II})^{-1}$
   - Computes expected discounted occupancy: $\mathbb{E}\left[\sum_{t=0}^{\tau-1} \gamma^t \mathbb{1}(s_t=j) \mid s_0=i\right]$
   - Used to calculate $r_o(s) = N \cdot r_I$ (matrix-vector product)

2. **Discounted Termination**: $p_{ss'}^o = \gamma \cdot N \cdot P_{IB}$
   - Properly couples discount $\gamma^\tau$ with stochastic termination time $\tau$
   - Handles stochastic transitions that may exit the option early

3. **Mixed Planning**: Can plan with primitives, options, or both
   - Primitives use standard 1-step MDP backup
   - Options use multi-step SMDP backup
   - Both compete in $\max_o$ operator (with tie-breaking for options)

## 📋 Files

- **`hallway_options.py`** - Defines the 8 hallway options with policies
- **`svi_planner.py`** - Implements SVI with SMDP backups for options

## 🚀 Usage Example

```python
from src.env.env import FourRooms
from src.planning.hallway_options import get_hallway_options, get_primitive_actions
from src.planning.svi_planner import SVIPlanner

env = FourRooms()
goal_state = (3, 6)  # Hallway goal

# Plan with primitives
primitives = get_primitive_actions()
planner_prim = SVIPlanner(env, primitives, gamma=0.99, theta=1e-6)
iters_prim = planner_prim.plan(goal_state)

# Plan with options
options = get_hallway_options()
planner_opt = SVIPlanner(env, options, gamma=0.99, theta=1e-6)
iters_opt = planner_opt.plan(goal_state)

print(f"Primitives: {iters_prim} iterations")
print(f"Options: {iters_opt} iterations")
print(f"Speedup: {iters_prim/iters_opt:.2f}x faster")
```

## 📚 References

- **Sutton, R. S., Precup, D., & Singh, S. (1999).** Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1-2), 181-211.