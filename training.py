"""
training.py
Member 2 - Training and testing for Q-learning and SARSA

Main functions (these are what Member 3 can call from the dashboard):
    create_agent(name, env, alpha, gamma, epsilon, ...)
    train_agent(env, agent, episodes)       -> training history
    test_agent(env, agent, episodes)        -> test results
    get_learned_path(env, agent)            -> robot path (list of (row, col))
    train_and_test(name, episodes, ...)     -> agent, train_results, test_results

Every episode result has this format (same as evaluation.py expects):
    {"success": bool, "steps": int, "total_reward": float, "collision": bool}
"""

from agents import QLearningAgent, SARSAAgent
from environment import RobotEnvironment


# -------------------------------------------------------------
# Create an agent by name
# -------------------------------------------------------------
def create_agent(name, env, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, seed=42):
    """name must be 'Q-learning' or 'SARSA'."""
    settings = dict(
        grid_size=env.grid_size,
        cycle_length=env.cycle_length,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    if name == "Q-learning":
        return QLearningAgent(**settings)
    if name == "SARSA":
        return SARSAAgent(**settings)
    raise ValueError("name must be 'Q-learning' or 'SARSA'")


# -------------------------------------------------------------
# Run ONE episode (learning on or off)
# -------------------------------------------------------------
def run_episode(env, agent, learn=True):
    state = env.reset()
    action = agent.choose_action(state, greedy=not learn)
    total_reward = 0.0
    steps = 0

    while True:
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        next_action = agent.choose_action(next_state, greedy=not learn)

        if learn:
            # Reaching the time limit is not a "real" end of the task,
            # so only goal / collision count as terminal.
            terminal = done and not info["max_steps_reached"]
            agent.update(state, action, reward, next_state,
                         next_action, terminal)

        state, action = next_state, next_action

        if done:
            break

    return {
        "success": bool(info["success"]),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "collision": bool(info["collision"]),
    }


# -------------------------------------------------------------
# Train
# -------------------------------------------------------------
def train_agent(env, agent, episodes=1000):
    """Train the agent. Returns one result per training episode."""
    history = []
    for _ in range(episodes):
        history.append(run_episode(env, agent, learn=True))
        agent.decay_epsilon()
    return history


# -------------------------------------------------------------
# Test (no learning, no random moves)
# -------------------------------------------------------------
def test_agent(env, agent, episodes=100):
    return [run_episode(env, agent, learn=False) for _ in range(episodes)]


# -------------------------------------------------------------
# Path the robot follows after learning (for the dashboard)
# -------------------------------------------------------------
def get_learned_path(env, agent):
    state = env.reset()
    path = [env.robot_pos]
    while True:
        action = agent.choose_action(state, greedy=True)
        state, _, done, _ = env.step(action)
        path.append(env.robot_pos)
        if done:
            break
    return path


# -------------------------------------------------------------
# Train + test + save, all in one call
# -------------------------------------------------------------
def train_and_test(name, episodes=1000, alpha=0.1, gamma=0.95,
                   epsilon=1.0, test_episodes=100, seed=42, save=True):
    env = RobotEnvironment()
    agent = create_agent(name, env, alpha, gamma, epsilon, seed=seed)
    train_results = train_agent(env, agent, episodes)
    test_results = test_agent(env, agent, test_episodes)

    if save:
        file_name = name.lower().replace("-", "_") + "_policy.npy"
        agent.save(file_name)

    return agent, train_results, test_results


# -------------------------------------------------------------
# Run this file directly to train both agents
# -------------------------------------------------------------
if __name__ == "__main__":
    for name in ["Q-learning", "SARSA"]:
        agent, train_res, test_res = train_and_test(name, episodes=1000)

        n = len(test_res)
        wins = sum(r["success"] for r in test_res)
        crashes = sum(r["collision"] for r in test_res)
        avg_steps = sum(r["steps"] for r in test_res) / n
        avg_reward = sum(r["total_reward"] for r in test_res) / n

        print(f"\n{name}")
        print(f"  Success rate : {wins / n * 100:.1f}%")
        print(f"  Collision rate: {crashes / n * 100:.1f}%")
        print(f"  Average steps : {avg_steps:.1f}")
        print(f"  Average reward: {avg_reward:.1f}")
        print(f"  Policy saved  : {name.lower().replace('-', '_')}_policy.npy")