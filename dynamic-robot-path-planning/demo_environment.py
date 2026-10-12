"""
demo_environment.py
=============================================================================
Interactive Simulation Demo for Member 1 — Environment & Simulation
Project: Dynamic Robot Path Planning Using Reinforcement Learning

Description:
    This script demonstrates the functionality of `RobotEnvironment`:
      1. Initializes the 10x10 environment with fixed walls and moving obstacles.
      2. Renders the initial ASCII grid.
      3. Executes a transparent scripted sequence of sample actions to show:
         - Valid movement (Right, Down)
         - Wall bump prevention and penalty
         - Dynamic obstacle updates at each step
         - State representation and reward feedback
      4. Saves a Matplotlib visualization image (`demo_simulation.png`).
      5. Demonstrates proper episode termination and reset handling.

NOTE: This script executes a fixed demonstration sequence to verify environment
mechanics and visualization. It does NOT represent a trained RL policy
(policy training is implemented by Member 2).
=============================================================================
"""

import sys
from environment import RobotEnvironment


def run_demo():
    print("=" * 70)
    print("   DYNAMIC ROBOT PATH PLANNING — ENVIRONMENT & SIMULATION DEMO   ")
    print("=" * 70)
    print("Initializing 10x10 RobotEnvironment...")

    # Create the standard environment instance
    env = RobotEnvironment(
        grid_size=10,
        start_pos=(0, 0),
        goal_pos=(9, 9),
        max_steps=50,
        reward_step=-1.0,
        reward_wall=-4.0,
        reward_goal=100.0,
        reward_collision=-100.0
    )

    # Reset environment to get initial state
    initial_state = env.reset()
    print(f"\n[Environment Reset]")
    print(f"Start Position   : {env.start_pos}")
    print(f"Goal Position    : {env.goal_pos}")
    print(f"Number of Walls  : {len(env.walls)}")
    print(f"Moving Obstacles : {len(env.obstacle_paths)}")
    print(f"Cycle Length (K) : {env.cycle_length}")
    print(f"Initial State    : {initial_state}  (row={initial_state[0]}, col={initial_state[1]}, phase={initial_state[2]})")
    print(f"Initial Obstacles: {env.get_obstacles()}")

    print("\nInitial Grid Visualization (ASCII):")
    env.render()

    # Define a transparent scripted sequence of actions for demonstration
    # 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
    demo_actions = [
        (RobotEnvironment.ACTION_RIGHT, "RIGHT (Valid move towards col 1)"),
        (RobotEnvironment.ACTION_DOWN,  "DOWN  (Attempt to bump into wall at (1, 1))"),
        (RobotEnvironment.ACTION_RIGHT, "RIGHT (Valid move to col 2)"),
        (RobotEnvironment.ACTION_DOWN,  "DOWN  (Attempt to bump into wall at (1, 2))"),
        (RobotEnvironment.ACTION_RIGHT, "RIGHT (Valid move to col 3)"),
        (RobotEnvironment.ACTION_DOWN,  "DOWN  (Valid move into row 1, col 3)"),
        (RobotEnvironment.ACTION_DOWN,  "DOWN  (Valid move into row 2, col 3 — near obstacle 0)"),
    ]

    print("\n" + "-" * 70)
    print("Executing Demonstration Action Sequence:")
    print("-" * 70)

    cumulative_reward = 0.0

    for step_num, (action, explanation) in enumerate(demo_actions, start=1):
        prev_robot = env.robot_pos
        prev_obs = env.get_obstacles()

        next_state, reward, done, info = env.step(action)
        cumulative_reward += reward

        print(f"\n[Step {step_num}] Action: {info['action_name']} ({action}) -> {explanation}")
        print(f"  Robot Position  : {prev_robot} -> {info['robot_pos']}")
        print(f"  Obstacle Pos    : {info['obstacle_positions']} (Phase: {info['phase']})")
        print(f"  Wall/Bnd Bump   : {info['hit_wall_or_boundary']}")
        print(f"  Step Reward     : {reward:+.1f} | Cumulative: {cumulative_reward:+.1f}")
        print(f"  Returned State  : {next_state}")
        print(f"  Episode Status  : Done={done} | Collision={info['collision']} | Success={info['success']}")

        # Render ASCII grid to observe changes
        env.render()

        if done:
            print("\n>>> Episode Terminated! <<<")
            if info["success"]:
                print("Result: Goal successfully reached!")
            elif info["collision"]:
                print("Result: Collision with moving obstacle detected!")
            elif info["max_steps_reached"]:
                print("Result: Maximum episode steps reached.")
            break

    # Save a visual plot to file for inspection
    save_filename = "demo_simulation.png"
    print(f"\nSaving graphical plot to '{save_filename}'...")
    env.plot(
        title=f"Demo Simulation | Step {env.step_count} | Phase {env.phase}",
        save_path=save_filename,
        show=False  # Headless friendly
    )
    print(f"Visualization saved successfully: {save_filename}")

    # Demonstrate safe reset
    print("\nDemonstrating Environment Reset after demo:")
    reset_state = env.reset()
    print(f"Environment reset confirmed. Current state: {reset_state}")
    print("=" * 70)
    print("Demo completed successfully. Ready for Member 2 & Member 3 integration!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
