"""
test_environment.py
=============================================================================
Automated Unit Tests for Member 1 — Environment & Simulation
Project: Dynamic Robot Path Planning Using Reinforcement Learning
Framework: Python built-in `unittest`

Tests cover all 15 required criteria:
  1. Initial grid, robot, and goal positions
  2. Correct reset behavior
  3. All four movement directions (Up, Down, Left, Right)
  4. Grid-boundary handling
  5. Fixed-wall collision prevention
  6. Moving-obstacle position updates
  7. Moving-obstacle collision detection (Co-location and Swap)
  8. Goal-reaching behavior and rewards
  9. Maximum-step termination
  10. Invalid action handling (ValueError)
  11. Stepping after episode termination (RuntimeError)
  12. State consistency and hashability for Q-table indexing
  13. Repeated episode resets
  14. Deterministic obstacle movement over multiple cycles
  15. Verification that obstacles never occupy prohibited cells
=============================================================================
"""

import unittest
from environment import RobotEnvironment


class TestRobotEnvironment(unittest.TestCase):
    """Comprehensive test suite for RobotEnvironment."""

    def setUp(self):
        """Initialize a standard test environment before each test."""
        self.env = RobotEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            goal_pos=(9, 9),
            max_steps=50,
            reward_step=-1.0,
            reward_wall=-4.0,
            reward_goal=100.0,
            reward_collision=-100.0
        )

    # 1. Initial grid, robot, and goal positions
    def test_initial_configuration(self):
        state = self.env.get_state()
        self.assertEqual(self.env.grid_size, 10)
        self.assertEqual(self.env.start_pos, (0, 0))
        self.assertEqual(self.env.goal_pos, (9, 9))
        self.assertEqual(self.env.robot_pos, (0, 0))
        self.assertEqual(state, (0, 0, 0))
        self.assertEqual(self.env.step_count, 0)
        self.assertFalse(self.env.done)
        self.assertFalse(self.env.collision)
        self.assertFalse(self.env.success)

    # 2. Correct reset behavior
    def test_reset_behavior(self):
        # Take some actions
        self.env.step(RobotEnvironment.ACTION_RIGHT)
        self.env.step(RobotEnvironment.ACTION_DOWN)
        self.assertGreater(self.env.step_count, 0)

        # Reset environment
        init_state = self.env.reset()
        self.assertEqual(init_state, (0, 0, 0))
        self.assertEqual(self.env.robot_pos, (0, 0))
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.phase, 0)
        self.assertFalse(self.env.done)
        self.assertFalse(self.env.collision)
        self.assertFalse(self.env.success)

    # 3. All four movement directions
    def test_movement_directions(self):
        # Start at (2, 2) in a clear subgrid with no obstacles
        custom_env = RobotEnvironment(
            grid_size=10,
            start_pos=(2, 2),
            goal_pos=(9, 9),
            walls=[],
            obstacle_paths=[[(5, 5)]]
        )
        custom_env.reset()

        # Action 3: Right -> (2, 3)
        state, reward, done, info = custom_env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertEqual(custom_env.robot_pos, (2, 3))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)

        # Action 1: Down -> (3, 3)
        state, reward, done, info = custom_env.step(RobotEnvironment.ACTION_DOWN)
        self.assertEqual(custom_env.robot_pos, (3, 3))
        self.assertEqual(reward, -1.0)

        # Action 2: Left -> (3, 2)
        state, reward, done, info = custom_env.step(RobotEnvironment.ACTION_LEFT)
        self.assertEqual(custom_env.robot_pos, (3, 2))
        self.assertEqual(reward, -1.0)

        # Action 0: Up -> (2, 2)
        state, reward, done, info = custom_env.step(RobotEnvironment.ACTION_UP)
        self.assertEqual(custom_env.robot_pos, (2, 2))
        self.assertEqual(reward, -1.0)

    # 4. Grid-boundary handling
    def test_grid_boundary_handling(self):
        self.env.reset()
        # At (0, 0), moving UP is out of bounds
        state, reward, done, info = self.env.step(RobotEnvironment.ACTION_UP)
        self.assertEqual(self.env.robot_pos, (0, 0))  # Must remain in place
        self.assertEqual(reward, -5.0)  # -1 step + -4 wall penalty
        self.assertTrue(info["hit_wall_or_boundary"])
        self.assertFalse(done)

        # At (0, 0), moving LEFT is out of bounds
        state, reward, done, info = self.env.step(RobotEnvironment.ACTION_LEFT)
        self.assertEqual(self.env.robot_pos, (0, 0))
        self.assertEqual(reward, -5.0)
        self.assertTrue(info["hit_wall_or_boundary"])

    # 5. Fixed-wall collision prevention
    def test_fixed_wall_prevention(self):
        # Custom environment with a wall directly right of start
        wall_env = RobotEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            goal_pos=(9, 9),
            walls=[(0, 1)],
            obstacle_paths=[[(5, 5)]]
        )
        wall_env.reset()

        # Attempt to step RIGHT into wall (0, 1)
        state, reward, done, info = wall_env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertEqual(wall_env.robot_pos, (0, 0))  # Must NOT penetrate wall
        self.assertEqual(reward, -5.0)  # -1 + -4
        self.assertTrue(info["hit_wall_or_boundary"])
        self.assertFalse(done)

    # 6. Moving-obstacle position updates
    def test_moving_obstacle_updates(self):
        self.env.reset()
        initial_obs = self.env.get_obstacles()
        self.assertEqual(initial_obs, [(2, 3), (4, 7), (7, 2)])
        self.assertEqual(self.env.phase, 0)

        # Step 1
        self.env.step(RobotEnvironment.ACTION_DOWN)
        step1_obs = self.env.get_obstacles()
        self.assertEqual(step1_obs, [(2, 4), (5, 7), (7, 3)])
        self.assertEqual(self.env.phase, 1)

        # Step 2
        self.env.step(RobotEnvironment.ACTION_DOWN)
        step2_obs = self.env.get_obstacles()
        self.assertEqual(step2_obs, [(2, 5), (6, 7), (7, 4)])
        self.assertEqual(self.env.phase, 2)

        # Step 3
        self.env.step(RobotEnvironment.ACTION_DOWN)
        step3_obs = self.env.get_obstacles()
        self.assertEqual(step3_obs, [(2, 4), (5, 7), (7, 3)])
        self.assertEqual(self.env.phase, 3)

        # Step 4 (wraps back to phase 0)
        self.env.step(RobotEnvironment.ACTION_DOWN)
        step4_obs = self.env.get_obstacles()
        self.assertEqual(step4_obs, initial_obs)
        self.assertEqual(self.env.phase, 0)

    # 7. Moving-obstacle collision detection (Co-location and Swap)
    def test_obstacle_colocation_collision(self):
        # Set up an obstacle that moves into (1, 0) at step 1
        col_env = RobotEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            goal_pos=(9, 9),
            walls=[],
            obstacle_paths=[[(2, 0), (1, 0)]]  # at step 1, obstacle enters (1, 0)
        )
        col_env.reset()

        # Robot moves DOWN to (1, 0) at step 1, co-locating with obstacle
        state, reward, done, info = col_env.step(RobotEnvironment.ACTION_DOWN)
        self.assertTrue(done)
        self.assertTrue(col_env.collision)
        self.assertTrue(info["collision"])
        self.assertFalse(info["success"])
        self.assertEqual(reward, -100.0)

    def test_obstacle_swap_collision(self):
        # Swap scenario:
        # Start at (0, 0). Obstacle path does NOT touch start (0, 0).
        # At step 1, robot moves to (0, 1), obstacle moves to (0, 2).
        # At step 2, robot attempts to move to (0, 2), while obstacle moves to (0, 1).
        # They swap adjacent cells (0, 1) <-> (0, 2) -> swap collision!
        swap_env = RobotEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            goal_pos=(9, 9),
            walls=[],
            obstacle_paths=[[(0, 3), (0, 2), (0, 1)]]
        )
        swap_env.reset()

        # Step 1: Robot moves RIGHT to (0, 1), obstacle moves to (0, 2)
        swap_env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertEqual(swap_env.robot_pos, (0, 1))
        self.assertEqual(swap_env.get_obstacles(), [(0, 2)])
        self.assertFalse(swap_env.done)

        # Step 2: Robot moves RIGHT to (0, 2), while obstacle moves to (0, 1)
        state, reward, done, info = swap_env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertTrue(done)
        self.assertTrue(swap_env.collision)
        self.assertTrue(info["collision"])
        self.assertEqual(reward, -100.0)

    # 8. Goal-reaching behavior
    def test_goal_reaching_success(self):
        # Place robot adjacent to goal
        goal_env = RobotEnvironment(
            grid_size=10,
            start_pos=(9, 8),
            goal_pos=(9, 9),
            walls=[],
            obstacle_paths=[[(0, 0), (0, 1)]]
        )
        goal_env.reset()

        # Move RIGHT into goal
        state, reward, done, info = goal_env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertTrue(done)
        self.assertTrue(goal_env.success)
        self.assertFalse(goal_env.collision)
        self.assertTrue(info["success"])
        self.assertEqual(reward, 100.0)
        self.assertEqual(goal_env.robot_pos, (9, 9))

    # 9. Maximum-step termination
    def test_max_steps_termination(self):
        short_env = RobotEnvironment(
            grid_size=10,
            start_pos=(0, 0),
            goal_pos=(9, 9),
            walls=[],
            obstacle_paths=[[(5, 5)]],
            max_steps=3
        )
        short_env.reset()

        # Step 1
        _, _, done, info = short_env.step(RobotEnvironment.ACTION_DOWN)
        self.assertFalse(done)
        # Step 2
        _, _, done, info = short_env.step(RobotEnvironment.ACTION_DOWN)
        self.assertFalse(done)
        # Step 3 (reaches max_steps)
        _, _, done, info = short_env.step(RobotEnvironment.ACTION_DOWN)
        self.assertTrue(done)
        self.assertTrue(info["max_steps_reached"])
        self.assertFalse(info["success"])
        self.assertFalse(info["collision"])

    # 10. Invalid action handling
    def test_invalid_action(self):
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(4)  # 4 is not in 0..3
        with self.assertRaises(ValueError):
            self.env.step(-1)

    # 11. Stepping after episode termination
    def test_stepping_after_termination(self):
        goal_env = RobotEnvironment(
            grid_size=10,
            start_pos=(9, 8),
            goal_pos=(9, 9),
            walls=[],
            obstacle_paths=[[(0, 0)]]
        )
        goal_env.reset()
        _, _, done, _ = goal_env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertTrue(done)

        # Next step without reset must raise RuntimeError
        with self.assertRaises(RuntimeError):
            goal_env.step(RobotEnvironment.ACTION_DOWN)

    # 12. State consistency and hashability
    def test_state_consistency_and_hashability(self):
        self.env.reset()
        state = self.env.get_state()
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 3)

        # Verify hashability by inserting state into a Python set and dictionary
        q_table = {}
        q_table[state] = [0.0, 0.0, 0.0, 0.0]
        self.assertIn(state, q_table)

        # Step and verify updated state matches robot position and phase
        next_state, _, _, _ = self.env.step(RobotEnvironment.ACTION_RIGHT)
        self.assertEqual(next_state, (self.env.robot_pos[0], self.env.robot_pos[1], self.env.phase))
        q_table[next_state] = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(len(q_table), 2)

    # 13. Repeated episode resets
    def test_repeated_resets(self):
        for _ in range(5):
            state = self.env.reset()
            self.assertEqual(state, (0, 0, 0))
            self.assertEqual(self.env.robot_pos, (0, 0))
            self.assertEqual(self.env.step_count, 0)
            self.assertEqual(self.env.phase, 0)
            self.assertFalse(self.env.done)
            self.env.step(RobotEnvironment.ACTION_DOWN)
            self.assertEqual(self.env.step_count, 1)

    # 14. Deterministic obstacle movement over multiple cycles
    def test_deterministic_obstacle_movement(self):
        self.env.reset()
        obs_trace_1 = []
        for _ in range(12):  # 3 full cycles of length 4
            self.env.step(RobotEnvironment.ACTION_DOWN)
            obs_trace_1.append(self.env.get_obstacles())

        # Reset and repeat
        self.env.reset()
        obs_trace_2 = []
        for _ in range(12):
            self.env.step(RobotEnvironment.ACTION_DOWN)
            obs_trace_2.append(self.env.get_obstacles())

        self.assertEqual(obs_trace_1, obs_trace_2)

    # 15. Prohibited cells verification
    def test_obstacles_never_occupy_prohibited_cells(self):
        # Verify default obstacle paths never hit walls, start, or goal
        for idx, path in enumerate(self.env.obstacle_paths):
            for coord in path:
                self.assertNotIn(
                    coord, self.env.walls,
                    f"Obstacle {idx} occupies a fixed wall at {coord}"
                )
                self.assertNotEqual(
                    coord, self.env.start_pos,
                    f"Obstacle {idx} occupies start position at {coord}"
                )
                self.assertNotEqual(
                    coord, self.env.goal_pos,
                    f"Obstacle {idx} occupies goal position at {coord}"
                )


if __name__ == "__main__":
    unittest.main()
