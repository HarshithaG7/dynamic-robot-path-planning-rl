# Dynamic Robot Path Planning Using Reinforcement Learning
## Member 1 — Environment & Simulation Module

Welcome to the **Environment & Simulation** module for our 3rd-year AI/ML engineering project: **Dynamic Robot Path Planning Using Reinforcement Learning**.

This module establishes the core discrete 2D simulation environment (`RobotEnvironment`), providing a standard, Gym-like Markov Decision Process (MDP) interface that supports both tabular **Q-Learning** and **SARSA** (implemented by Member 2), and integrates with our interactive **Streamlit dashboard** (implemented by Member 3).

---

## 1. Project Directory Structure

```text
dynamic-robot-path-planning/
│
├── environment.py          # Complete RobotEnvironment class & simulation dynamics
├── test_environment.py     # Automated unit test suite (16 comprehensive unit tests)
├── demo_environment.py     # Runnable simulation demo & action trace generator
├── requirements.txt        # External dependencies (numpy, matplotlib)
├── demo_simulation.png     # Rendered visualization produced by demo
└── README.md               # Architecture documentation, API guide, & integration manual
```

---

## 2. Environment Architecture & Key Concepts

### A. 10×10 Grid World
- **Grid Coordinates**: Discrete cells `(row, col)` indexed from `(0, 0)` at the top-left to `(9, 9)` at the bottom-right.
- **Start Position**: Default `(0, 0)`.
- **Goal Destination**: Default `(9, 9)`.
- **Fixed Walls**: Configurable static barriers (default: 10 wall blocks) that prevent robot penetration. Neither start, goal, nor moving obstacle waypoints overlap with walls.
- **Dynamic Obstacles**: 3 independent moving obstacles with synchronized cyclic patrol paths (cycle length $K = 4$).

### B. Action Space
The environment defines 4 discrete actions:
| Action ID | Action Name | Row Delta ($\Delta r$) | Col Delta ($\Delta c$) |
| :---: | :---: | :---: | :---: |
| `0` | `UP` | `-1` | `0` |
| `1` | `DOWN` | `+1` | `0` |
| `2` | `LEFT` | `0` | `-1` |
| `3` | `RIGHT` | `0` | `+1` |

### C. State Representation for Tabular RL
The environment provides a compact, hashable 3-tuple state:
$$\text{state} = (r_{\text{robot}}, c_{\text{robot}}, \text{phase})$$
where $\text{phase} = \text{step\_count} \pmod K$, and $K$ is the least common multiple (LCM) of the obstacle path lengths (default $K = 4$).

#### Why this state representation is optimal for Tabular Q-Learning and SARSA:
1. **Solves the Curse of Dimensionality**:
   If obstacle positions were tracked as raw coordinates $(r, c, o_{1r}, o_{1c}, o_{2r}, o_{2c}, o_{3r}, o_{3c})$, the state space would be $10^8 = 100,000,000$ states—rendering tabular RL impossible to converge.
   By leveraging deterministic cyclic paths, all 3 obstacle coordinates are a pure mathematical function of `phase`. Thus, the state space is simply:
   $$10 \times 10 \times 4 = 400 \text{ states}$$
2. **Instant Convergence**:
   A Q-table with $400 \text{ states} \times 4 \text{ actions} = 1,600 \text{ values}$ fits in a lightweight NumPy array `Q[10, 10, 4, 4]` or Python `dict`, converging to an optimal policy in just a few hundred episodes.
3. **Strict Markov Property**:
   Because obstacle positions and subsequent states depend only on the current state and action, the environment satisfies the Markov Decision Process (MDP) property.

### D. Physics & Collision Mechanics
During `step(action)`:
1. **Intended Move Calculation**: Intended cell is calculated.
2. **Wall & Grid Bounds Enforcement**: If the intended cell is out of bounds or hits a fixed wall, the robot does not move (stays in place) and incurs a wall penalty.
3. **Obstacle Progression**: Environment step count increments and obstacles advance along their cyclic patrol paths.
4. **Collision Detection**:
   - **Co-location Collision**: Robot and an obstacle occupy the exact same cell.
   - **Swap / Cross-over Collision**: Robot and an obstacle simultaneously swap adjacent cells (e.g. robot moved $A \to B$ while obstacle moved $B \to A$).
   - Upon collision, `done = True`, `collision = True`, and penalty of `-100` is returned.
5. **Goal Detection**: If robot reaches `(9, 9)` without collision, `done = True`, `success = True`, and reward of `+100` is returned.

### E. Configurable Reward Structure
| Event | Reward Value | Termination | Description |
| :--- | :---: | :---: | :--- |
| **Normal Step** | `-1.0` | No | Encourages the robot to reach the goal via the shortest path |
| **Wall / Boundary Hit** | `-5.0` (`-1` step + `-4` wall) | No | Penalizes bumping into walls/edges while keeping robot in place |
| **Goal Reached** | `+100.0` | Yes (`success=True`) | Terminal reward for completing the navigation task |
| **Obstacle Collision** | `-100.0` | Yes (`collision=True`) | Terminal penalty for colliding with dynamic obstacles |
| **Timeout (Max Steps)** | `-1.0` | Yes (`max_steps_reached=True`) | Truncates endless wandering (default 50 or 100 steps) |

---

## 3. Installation & Setup (Windows)

### Prerequisites
- Python 3.9+ (Python 3.11–3.13 supported)
- Windows PowerShell or Command Prompt

### Step 1: Navigate to Project Directory
```powershell
cd c:\Users\priya\OneDrive\Documents\ML-2\dynamic-robot-path-planning
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 4. Execution Commands

### Run Unit Tests
Run the automated test suite covering all 16 specifications:
```powershell
python test_environment.py
```
*Expected result*: `Ran 16 tests in 0.001s ... OK`

### Run Environment Demo
Run the visual and terminal demonstration script:
```powershell
python demo_environment.py
```
This prints the ASCII grid transitions, executes a sample action sequence, demonstrates wall bumps and obstacle movements, and saves `demo_simulation.png`.

---

## 5. Environment API Reference (`RobotEnvironment`)

### Class Initialization
```python
from environment import RobotEnvironment

env = RobotEnvironment(
    grid_size=10,                      # 10x10 grid
    start_pos=(0, 0),                  # Robot start
    goal_pos=(9, 9),                   # Target destination
    walls=None,                        # None uses default 10 walls
    obstacle_paths=None,               # None uses default 3 cyclic paths
    max_steps=100,                     # Episode step limit
    reward_step=-1.0,                  # Normal step reward
    reward_wall=-4.0,                  # Wall bump penalty
    reward_goal=100.0,                 # Goal reward
    reward_collision=-100.0            # Collision penalty
)
```

### Core Methods
- `state = env.reset()`
  Resets the robot to start, resets step count and status flags, and returns the initial state tuple `(0, 0, 0)`.
- `next_state, reward, done, info = env.step(action)`
  Executes discrete action `action` ($0, 1, 2, 3$). Returns:
  - `next_state`: `(row, col, phase)`
  - `reward`: `float`
  - `done`: `bool`
  - `info`: dictionary containing `robot_pos`, `goal_pos`, `obstacle_positions`, `collision`, `success`, `step_count`, `phase`, etc.
- `state = env.get_state()`
  Returns the current state tuple `(row, col, phase)`.
- `obstacles = env.get_obstacles()`
  Returns the list of current `(row, col)` positions of all 3 obstacles.
- `env.render(mode="text")`
  Prints clean ASCII grid with symbols (`R` robot, `G` goal, `#` wall, `O` obstacle, `!` collision).
- `ax = env.plot(title=None, save_path=None, show=True, ax=None)`
  Draws or updates a 2D Matplotlib plot with a custom legend, grid coordinates, and color-coded elements.
- `snapshot = env.get_snapshot()`
  Returns a serializable dictionary containing full environment state for UI integration.

---

## 6. Integration Guide for Member 2 (Reinforcement Learning)

Member 2 will implement **Q-Learning** (off-policy TD control) and **SARSA** (on-policy TD control).

### Recommended State-Action Structure
Because our state is `(row, col, phase)` where $row \in [0, 9]$, $col \in [0, 9]$, and $phase \in [0, 3]$, Member 2 can use either:
1. **NumPy 4D Array**: `Q = np.zeros((10, 10, 4, 4))` where `Q[r, c, phase, action]` stores the Q-value.
2. **Python `defaultdict`**: `Q = defaultdict(lambda: np.zeros(4))` where keys are state tuples `(r, c, phase)`.

### Sample Training Loop Template for Member 2
```python
import numpy as np
from environment import RobotEnvironment

# Initialize environment
env = RobotEnvironment(max_steps=100)

# Initialize Q-table: (10, 10, 4, 4)
num_episodes = 1000
alpha = 0.1       # Learning rate
gamma = 0.99      # Discount factor
epsilon = 1.0     # Initial exploration rate
min_epsilon = 0.05
decay_rate = 0.995

# --- Q-LEARNING UPDATE EXAMPLE ---
for episode in range(num_episodes):
    state = env.reset()  # (r, c, phase)
    done = False
    
    while not done:
        r, c, p = state
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        else:
            action = int(np.argmax(Q[r, c, p]))
            
        next_state, reward, done, info = env.step(action)
        nr, nc, np_phase = next_state
        
        # Q-Learning update (Off-policy: max over next actions)
        best_next_q = np.max(Q[nr, nc, np_phase])
        td_target = reward + (0.0 if done else gamma * best_next_q)
        Q[r, c, p, action] += alpha * (td_target - Q[r, c, p, action])
        
        state = next_state
        
    epsilon = max(min_epsilon, epsilon * decay_rate)
```

---

## 7. Integration Guide for Member 3 (Streamlit Dashboard)

Member 3 will build the interactive web dashboard. `RobotEnvironment` provides two direct hooks for Streamlit:

### Approach A: Render with Matplotlib in Streamlit
```python
import streamlit as st
import matplotlib.pyplot as plt
from environment import RobotEnvironment

env = RobotEnvironment()
state = env.reset()

st.title("Robot Path Planning Simulation")
fig, ax = plt.subplots(figsize=(6, 6))
env.plot(ax=ax, show=False)
st.pyplot(fig)
plt.close(fig)
```

### Approach B: Use `env.get_snapshot()` for Custom Dynamic UI
```python
snapshot = env.get_snapshot()
st.write(f"Robot Position: {snapshot['robot_pos']}")
st.write(f"Obstacle Positions: {snapshot['obstacle_positions']}")
st.write(f"Current Phase: {snapshot['phase']}")
st.write(f"Status: Done={snapshot['done']}, Success={snapshot['success']}")
```

---

## 8. Summary of Completed Deliverables
- [x] Complete, tested `RobotEnvironment` class in `environment.py`
- [x] 16 passing automated unit tests in `test_environment.py`
- [x] Runnable simulation script in `demo_environment.py`
- [x] Clear dependency specifications in `requirements.txt`
- [x] Zero external reinforcement learning dependencies embedded in environment
- [x] Fully ready for seamless integration with Member 2 and Member 3
