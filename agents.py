"""
agents.py
Member 2 - Reinforcement Learning Algorithms

Contains two tabular RL agents that work with RobotEnvironment:
    - QLearningAgent
    - SARSAAgent

Both agents share the same interface, so the training loop
and the dashboard can treat them in the same way.

State from the environment: (row, col, phase)
Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
"""

import numpy as np


class BaseAgent:
    """Shared code: Q-table, epsilon-greedy action choice, save/load."""

    def __init__(
        self,
        grid_size=10,
        cycle_length=4,
        n_actions=4,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,

    ):
        self.grid_size = grid_size
        self.cycle_length = cycle_length
        self.n_actions = n_actions

        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # exploration rate (current)
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        # Q-table: one value for every (row, col, phase, action)
        self.q_table = np.zeros(
            (grid_size, grid_size, cycle_length, n_actions)
        )

    # ---------------------------------------------------------
    def choose_action(self, state, greedy=False):
        """
        Pick an action using epsilon-greedy.
        greedy=True means no exploration (used for testing).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.best_action(state)

    def best_action(self, state):
        """Action with the highest Q-value (ties broken randomly)."""
        q_values = self.q_table[state]

        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def decay_epsilon(self):
        """Call once at the end of every episode."""
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

    # ---------------------------------------------------------
    def save(self, path):
        """Save the learned Q-table (the learned policy)."""
        np.save(path, self.q_table)

    def load(self, path):
        """Load a saved Q-table."""
        self.q_table = np.load(path)

    def update(self, state, action, reward, next_state, next_action, done):
        raise NotImplementedError


class QLearningAgent(BaseAgent):
    """
    Q-learning (off-policy).
    Learns from the BEST next action, whatever it actually does next.

    Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))
    """

    name = "Q-learning"


    def update(self, state, action, reward, next_state,
               next_action=None, done=False):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        old_value = self.q_table[state][action]
        self.q_table[state][action] = (
            old_value + self.alpha * (target - old_value)
        )


class SARSAAgent(BaseAgent):
    """
    SARSA (on-policy).
    Learns from the next action it ACTUALLY takes.

    Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
    """

    name = "SARSA"

    def update(self, state, action, reward, next_state,
               next_action=None, done=False):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]

        old_value = self.q_table[state][action]
        self.q_table[state][action] = (

            old_value + self.alpha * (target - old_value)
        )