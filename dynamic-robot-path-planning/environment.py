"""
environment.py
=============================================================================
Module: Member 1 — Environment & Simulation
Project: Dynamic Robot Path Planning Using Reinforcement Learning
Author: Member 1 (Environment & Simulation Engineer)

Description:
    This module implements a discrete 2D grid world simulation (`RobotEnvironment`)
    for training and evaluating Reinforcement Learning (RL) agents (Q-Learning and
    SARSA). The environment features:
      - 10x10 discrete grid world (100 cells).
      - Configurable static walls and boundaries.
      - 3 dynamic (moving) obstacles following deterministic cyclic patrol paths.
      - Compact Markovian state representation: (robot_row, robot_col, obstacle_phase).
      - Configurable reward structure and robust collision detection.
      - ASCII text rendering and Matplotlib visualization tools.
      - Standard Gym-like interface: reset() and step(action).

No learning algorithms (Q-learning or SARSA) are implemented here. This module
serves as the foundational simulation environment for Member 2 (Algorithms)
and Member 3 (Streamlit Dashboard & Evaluation).
=============================================================================
"""

import math
from typing import List, Tuple, Set, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class RobotEnvironment:
    """
    2D Grid World Environment for Autonomous Robot Path Planning.

    State Representation:
        The environment returns a hashable tuple:
            state = (robot_row, robot_col, obstacle_phase)

        Why this representation is ideal for Tabular Q-Learning and SARSA:
          1. Compactness: On a 10x10 grid with obstacles synchronized on a
             cycle of length K (default K = 4), the total number of states is
             10 * 10 * 4 = 400 states.
          2. Tabular Feasibility: A state-action table of size 400 x 4 = 1,600
             entries easily converges in a few hundred episodes.
          3. Markov Property: Because obstacle movement is deterministic and
             cyclical, the obstacle positions are uniquely determined by
             `obstacle_phase`. The state is therefore fully Markovian.
    """

    # Discrete Action Definitions
    ACTION_UP: int = 0
    ACTION_DOWN: int = 1
    ACTION_LEFT: int = 2
    ACTION_RIGHT: int = 3

    ACTION_NAMES: Dict[int, str] = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    }

    ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # Up: decrement row
        1: (1, 0),   # Down: increment row
        2: (0, -1),  # Left: decrement col
        3: (0, 1)    # Right: increment col
    }

    # Default Static Walls (row, col)
    # Selected to create realistic corridors without blocking all feasible paths
    DEFAULT_WALLS: List[Tuple[int, int]] = [
        (1, 1), (1, 2),
        (3, 7), (3, 8),
        (4, 2), (5, 2),
        (5, 4), (5, 5),
        (8, 6), (8, 7)
    ]

    # Default Moving Obstacle Repeating Paths (3 obstacles)
    # Each obstacle has a 4-step cyclic patrol path (Length = 4 -> cycle_length = 4)
    # None of these paths overlap with start (0, 0), goal (9, 9), or default walls.
    DEFAULT_OBSTACLE_PATHS: List[List[Tuple[int, int]]] = [
        # Obstacle 0: Upper horizontal patrol across row 2, columns 3 -> 5
        [(2, 3), (2, 4), (2, 5), (2, 4)],
        # Obstacle 1: Right vertical patrol across column 7, rows 4 -> 6
        [(4, 7), (5, 7), (6, 7), (5, 7)],
        # Obstacle 2: Lower horizontal patrol across row 7, columns 2 -> 4
        [(7, 2), (7, 3), (7, 4), (7, 3)]
    ]

    def __init__(
        self,
        grid_size: int = 10,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (9, 9),
        walls: Optional[List[Tuple[int, int]]] = None,
        obstacle_paths: Optional[List[List[Tuple[int, int]]]] = None,
        max_steps: int = 100,
        reward_step: float = -1.0,
        reward_wall: float = -4.0,
        reward_goal: float = 100.0,
        reward_collision: float = -100.0
    ):
        """
        Initialize the 2D Robot Environment.

        Args:
            grid_size: Dimension of the square grid (default: 10 -> 10x10).
            start_pos: (row, col) coordinates for robot start position.
            goal_pos: (row, col) coordinates for goal destination.
            walls: List of (row, col) tuples representing fixed walls.
            obstacle_paths: List of repeating coordinate sequences for 3 obstacles.
            max_steps: Maximum allowed steps per episode before truncation.
            reward_step: Base reward penalty incurred on every movement step.
            reward_wall: Additional penalty for bumping into a wall or grid boundary.
            reward_goal: Terminal reward granted upon reaching the goal.
            reward_collision: Terminal penalty incurred upon colliding with an obstacle.
        """
        # Validate grid size
        if grid_size < 3:
            raise ValueError(f"grid_size must be at least 3, got {grid_size}")
        self.grid_size: int = grid_size

        # Validate start and goal
        self._validate_in_bounds(start_pos, "start_pos")
        self._validate_in_bounds(goal_pos, "goal_pos")
        if start_pos == goal_pos:
            raise ValueError(f"start_pos {start_pos} cannot be the same as goal_pos {goal_pos}")

        self.start_pos: Tuple[int, int] = start_pos
        self.goal_pos: Tuple[int, int] = goal_pos

        # Initialize and validate fixed walls
        wall_list = walls if walls is not None else self.DEFAULT_WALLS
        self.walls: Set[Tuple[int, int]] = set()
        for w in wall_list:
            self._validate_in_bounds(w, "wall coordinate")
            if w == self.start_pos:
                raise ValueError(f"Wall at {w} overlaps with start_pos {self.start_pos}")
            if w == self.goal_pos:
                raise ValueError(f"Wall at {w} overlaps with goal_pos {self.goal_pos}")
            self.walls.add(w)

        # Initialize and validate moving obstacles
        obs_paths = obstacle_paths if obstacle_paths is not None else self.DEFAULT_OBSTACLE_PATHS
        if not obs_paths or len(obs_paths) == 0:
            raise ValueError("At least one moving obstacle path must be provided.")
        self.obstacle_paths: List[List[Tuple[int, int]]] = []
        for i, path in enumerate(obs_paths):
            if not path or len(path) == 0:
                raise ValueError(f"Obstacle path {i} cannot be empty.")
            sanitized_path = []
            for coord in path:
                self._validate_in_bounds(coord, f"obstacle {i} waypoint")
                if coord in self.walls:
                    raise ValueError(f"Obstacle {i} waypoint {coord} overlaps with a wall.")
                if coord == self.start_pos:
                    raise ValueError(f"Obstacle {i} waypoint {coord} overlaps with start_pos.")
                if coord == self.goal_pos:
                    raise ValueError(f"Obstacle {i} waypoint {coord} overlaps with goal_pos.")
                sanitized_path.append(coord)
            self.obstacle_paths.append(sanitized_path)

        # Compute cycle length via Least Common Multiple (LCM) of all path lengths
        path_lengths = [len(p) for p in self.obstacle_paths]
        self.cycle_length: int = math.lcm(*path_lengths)

        # Verify no two obstacles occupy the same cell at the same phase
        for phase_idx in range(self.cycle_length):
            positions_at_phase = [
                path[phase_idx % len(path)] for path in self.obstacle_paths
            ]
            if len(positions_at_phase) != len(set(positions_at_phase)):
                raise ValueError(
                    f"Obstacle overlap detected at phase {phase_idx}: positions {positions_at_phase}"
                )

        # Episode limits & reward configuration
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.max_steps: int = max_steps
        self.reward_step: float = float(reward_step)
        self.reward_wall: float = float(reward_wall)
        self.reward_goal: float = float(reward_goal)
        self.reward_collision: float = float(reward_collision)

        # Dynamic state variables (initialized in reset)
        self.robot_pos: Tuple[int, int] = self.start_pos
        self.step_count: int = 0
        self.done: bool = False
        self.collision: bool = False
        self.success: bool = False

    # -------------------------------------------------------------------------
    # Helper Validation Methods
    # -------------------------------------------------------------------------
    def _validate_in_bounds(self, pos: Tuple[int, int], name: str = "coordinate") -> None:
        """Verify that a given (row, col) is within grid dimensions."""
        if not (isinstance(pos, (tuple, list)) and len(pos) == 2):
            raise TypeError(f"{name} must be a 2-element tuple/list of ints, got {pos}")
        r, c = pos
        if not (isinstance(r, int) and isinstance(c, int)):
            raise TypeError(f"{name} indices must be integers, got ({type(r)}, {type(c)})")
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            raise ValueError(
                f"{name} {pos} is out of bounds for a {self.grid_size}x{self.grid_size} grid."
            )

    @property
    def phase(self) -> int:
        """Current cyclic movement phase of dynamic obstacles."""
        return self.step_count % self.cycle_length

    # -------------------------------------------------------------------------
    # Core Environment API
    # -------------------------------------------------------------------------
    def reset(self) -> Tuple[int, int, int]:
        """
        Reset the environment to its initial state.

        Returns:
            initial_state: Hashable tuple (robot_row, robot_col, obstacle_phase).
        """
        self.robot_pos = self.start_pos
        self.step_count = 0
        self.done = False
        self.collision = False
        self.success = False
        return self.get_state()

    def get_state(self) -> Tuple[int, int, int]:
        """
        Return the current RL state.

        Returns:
            state: (robot_row, robot_col, obstacle_phase)
        """
        return (self.robot_pos[0], self.robot_pos[1], self.phase)

    def get_obstacles(self) -> List[Tuple[int, int]]:
        """
        Return the current positions of all moving obstacles.

        Returns:
            positions: List of (row, col) coordinates for each obstacle.
        """
        current_phase = self.phase
        return [path[current_phase % len(path)] for path in self.obstacle_paths]

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Update Order & Physics:
          1. Validation: Validate action is in {0, 1, 2, 3} and episode is not done.
          2. Robot Intended Move: Calculate prospective position based on action.
          3. Wall & Boundary Check: If the move targets a wall or is out of bounds,
             the robot remains in its current cell and incurs a wall penalty.
             Otherwise, the robot advances to the target cell.
          4. Obstacle Progression: Advance the environment step counter and update
             all moving obstacles to their new positions for phase (t + 1).
          5. Collision Detection:
             - Co-location Collision: Robot's new position matches an obstacle's new position.
             - Swap / Cross-over Collision: Robot and an obstacle cross paths directly
               (robot moved from cell A to B, while obstacle moved from cell B to A).
             If a collision occurs, episode terminates with `reward_collision`.
          6. Goal Detection: If robot reaches `goal_pos` without collision,
             episode terminates with `reward_goal`.
          7. Max Steps Check: If `step_count >= max_steps`, episode terminates.

        Args:
            action: Integer action (0=Up, 1=Down, 2=Left, 3=Right).

        Returns:
            next_state: (robot_row, robot_col, obstacle_phase)
            reward: Float reward signal.
            done: Boolean indicating if the episode has terminated.
            info: Dictionary containing diagnostic and visualization metadata.
        """
        # Step after episode ended guard
        if self.done:
            raise RuntimeError(
                "Cannot call step() on an episode that has already ended. "
                "Call reset() to start a new episode."
            )

        # Action validation
        if action not in self.ACTION_DELTAS:
            raise ValueError(
                f"Invalid action: {action}. Must be one of "
                f"0 (Up), 1 (Down), 2 (Left), 3 (Right)."
            )

        # Previous state positions
        prev_robot_pos = self.robot_pos
        prev_obstacles = self.get_obstacles()

        # Calculate intended robot position
        delta = self.ACTION_DELTAS[action]
        intended_pos = (prev_robot_pos[0] + delta[0], prev_robot_pos[1] + delta[1])

        # Evaluate grid boundaries and walls
        is_out_of_bounds = not (0 <= intended_pos[0] < self.grid_size and 0 <= intended_pos[1] < self.grid_size)
        is_wall = intended_pos in self.walls
        hit_wall_or_boundary = is_out_of_bounds or is_wall

        if hit_wall_or_boundary:
            # Robot cannot penetrate wall or boundary; stays in place
            next_robot_pos = prev_robot_pos
        else:
            next_robot_pos = intended_pos

        # Update environment time and advance moving obstacles
        self.step_count += 1
        new_obstacles = self.get_obstacles()

        # Collision detection (Co-location or Swap)
        collision_detected = False

        # Scenario A: Co-location (robot and an obstacle occupy the same cell)
        if next_robot_pos in new_obstacles:
            collision_detected = True

        # Scenario B: Crossover/Swap (robot and obstacle exchanged adjacent cells)
        if not collision_detected and not hit_wall_or_boundary:
            for obs_idx in range(len(self.obstacle_paths)):
                if (prev_robot_pos == new_obstacles[obs_idx] and
                        next_robot_pos == prev_obstacles[obs_idx]):
                    collision_detected = True
                    break

        # Process terminal states and rewards
        self.robot_pos = next_robot_pos

        if collision_detected:
            self.collision = True
            self.done = True
            self.success = False
            reward = self.reward_collision
            max_steps_reached = False
        elif self.robot_pos == self.goal_pos:
            self.collision = False
            self.done = True
            self.success = True
            reward = self.reward_goal
            max_steps_reached = False
        else:
            self.collision = False
            self.success = False
            # Base step penalty plus wall penalty if blocked
            reward = self.reward_step + (self.reward_wall if hit_wall_or_boundary else 0.0)

            if self.step_count >= self.max_steps:
                self.done = True
                max_steps_reached = True
            else:
                self.done = False
                max_steps_reached = False

        # Diagnostic info dictionary for logging and Member 3 dashboard
        info: Dict[str, Any] = {
            "robot_pos": self.robot_pos,
            "goal_pos": self.goal_pos,
            "obstacle_positions": new_obstacles,
            "action_taken": action,
            "action_name": self.ACTION_NAMES[action],
            "hit_wall_or_boundary": hit_wall_or_boundary,
            "collision": self.collision,
            "success": self.success,
            "step_count": self.step_count,
            "phase": self.phase,
            "max_steps_reached": max_steps_reached
        }

        return self.get_state(), reward, self.done, info

    # -------------------------------------------------------------------------
    # Visualization & Diagnostic Utilities
    # -------------------------------------------------------------------------
    def render(self, mode: str = "text") -> Optional[str]:
        """
        Render the current environment state as a clean ASCII grid in terminal.

        Symbols:
            'R' : Robot
            'G' : Goal
            '#' : Fixed Wall
            'O' : Moving Obstacle
            '!' : Robot Collided with Obstacle
            '.' : Empty Cell
        """
        obstacles = set(self.get_obstacles())
        lines = []

        header = (
            f"--- Grid ({self.grid_size}x{self.grid_size}) | "
            f"Step: {self.step_count}/{self.max_steps} | Phase: {self.phase} ---"
        )
        lines.append(header)
        lines.append("+" + "---" * self.grid_size + "+")

        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.robot_pos and pos in obstacles:
                    row_chars.append(" ! ")  # Collision symbol
                elif pos == self.robot_pos:
                    row_chars.append(" R ")
                elif pos == self.goal_pos:
                    row_chars.append(" G ")
                elif pos in self.walls:
                    row_chars.append(" # ")
                elif pos in obstacles:
                    row_chars.append(" O ")
                else:
                    row_chars.append(" . ")
            lines.append("|" + "".join(row_chars) + "|")

        lines.append("+" + "---" * self.grid_size + "+")
        output = "\n".join(lines)

        if mode == "text":
            print(output)
            return None
        return output

    def plot(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Render a clean, modern Matplotlib 2D visualization of the environment.

        Args:
            title: Custom plot title (optional).
            save_path: If specified, save figure to this file path.
            show: If True, call plt.show() (set to False for headless testing).
            ax: Existing Matplotlib Axes to draw on (useful for Streamlit).

        Returns:
            ax: The Matplotlib Axes containing the plot.
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            created_fig = True

        ax.clear()
        obstacles = self.get_obstacles()

        # Set grid boundaries and orientation
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)  # Inverted Y so (0,0) is top-left
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_xticklabels(range(self.grid_size))
        ax.set_yticklabels(range(self.grid_size))
        ax.grid(True, color="#CBD5E1", linestyle="-", linewidth=1.0)
        ax.set_aspect("equal")

        # Color background cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if pos in self.walls:
                    rect_color = "#334155"  # Slate dark for walls
                elif pos == self.goal_pos:
                    rect_color = "#DCFCE7"  # Soft mint green for goal cell
                elif pos == self.start_pos:
                    rect_color = "#E0E7FF"  # Soft lavender for start cell
                else:
                    rect_color = "#F8FAFC"  # Clean off-white for empty cells

                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=rect_color,
                    edgecolor="none"
                )
                ax.add_patch(rect)

        # Plot moving obstacles (Crimson with 'O' marker)
        for idx, (obs_r, obs_c) in enumerate(obstacles):
            circle = patches.Circle(
                (obs_c, obs_r), 0.35,
                facecolor="#EF4444",
                edgecolor="#991B1B",
                linewidth=1.5,
                label="Moving Obstacle (O)" if idx == 0 else ""
            )
            ax.add_patch(circle)
            ax.text(
                obs_c, obs_r, f"O{idx + 1}",
                color="white",
                fontsize=8,
                weight="bold",
                ha="center",
                va="center"
            )

        # Plot Goal (Emerald star/patch)
        goal_r, goal_c = self.goal_pos
        ax.plot(
            goal_c, goal_r,
            marker="*", markersize=20,
            color="#059669", markeredgecolor="#047857",
            label="Goal (G)", linestyle="None"
        )

        # Plot Robot
        rob_r, rob_c = self.robot_pos
        if self.collision:
            # Highlight collision with warning burst
            ax.plot(
                rob_c, rob_r,
                marker="X", markersize=18,
                color="#DC2626", markeredgecolor="#7F1D1D",
                label="Robot Collided (!)", linestyle="None"
            )
        else:
            circle = patches.Circle(
                (rob_c, rob_r), 0.36,
                facecolor="#2563EB",
                edgecolor="#1D4ED8",
                linewidth=1.8,
                label="Robot (R)"
            )
            ax.add_patch(circle)
            ax.text(
                rob_c, rob_r, "R",
                color="white", fontsize=9, weight="bold",
                ha="center", va="center"
            )

        # Wall legend dummy patch
        wall_dummy = patches.Patch(facecolor="#334155", label="Fixed Wall (#)")

        # Configure title and status labels
        status_str = "RUNNING"
        if self.success:
            status_str = "SUCCESS (GOAL REACHED)"
        elif self.collision:
            status_str = "TERMINATED (COLLISION)"
        elif self.done:
            status_str = "TERMINATED (MAX STEPS)"

        plot_title = title or (
            f"Robot Environment | Step: {self.step_count}/{self.max_steps} "
            f"| Phase: {self.phase} | Status: {status_str}"
        )
        ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=12)

        # Assemble clean legend outside or above plot
        handles, labels = ax.get_legend_handles_labels()
        handles.append(wall_dummy)
        labels.append("Fixed Wall (#)")
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            fontsize=8,
            frameon=True
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        if show and created_fig:
            plt.show()

        return ax

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Return a comprehensive state dictionary snapshot.
        Specifically provided for Member 3's Streamlit dashboard integration.
        """
        return {
            "grid_size": self.grid_size,
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "robot_pos": self.robot_pos,
            "walls": list(self.walls),
            "obstacle_positions": self.get_obstacles(),
            "cycle_length": self.cycle_length,
            "phase": self.phase,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "collision": self.collision,
            "success": self.success,
            "state_tuple": self.get_state()
        }

    def get_grid_matrix(self) -> np.ndarray:
        """
        Return a numeric 2D NumPy array representation of the grid for fast inspection.

        Cell Values:
            0: Empty cell
            1: Wall
            2: Moving obstacle
            3: Robot
            4: Goal
            5: Collision
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for r, c in self.walls:
            grid[r, c] = 1
        for r, c in self.get_obstacles():
            grid[r, c] = 2
        gr, gc = self.goal_pos
        grid[gr, gc] = 4
        rr, rc = self.robot_pos
        if self.collision:
            grid[rr, rc] = 5
        else:
            grid[rr, rc] = 3
        return grid
