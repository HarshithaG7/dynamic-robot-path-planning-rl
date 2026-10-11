
"""
evaluation.py

Evaluation utilities for Dynamic Robot Path Planning
Using Reinforcement Learning.

This module:
1. Evaluates episode results.
2. Calculates success, steps, reward, and collision metrics.
3. Compares Q-learning and SARSA results.
4. Returns pandas DataFrames suitable for Streamlit.

Expected episode-result format:
{
    "success": True,
    "steps": 18,
    "total_reward": 75.0,
    "collision": False
}

IMPORTANT:
These functions evaluate results produced by a real
environment or agent. They do not train agents themselves.
"""

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


REQUIRED_FIELDS = {
    "success",
    "steps",
    "total_reward",
    "collision",
}


def validate_episode_results(
    episode_results: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validate and normalize a collection of episode results.

    Each result must contain:
        success: bool
        steps: non-negative integer
        total_reward: numeric
        collision: bool

    Returns a list of validated episode dictionaries.
    """
    results = list(episode_results)

    if not results:
        raise ValueError(
            "No episode results were provided. "
            "Run evaluation episodes first."
        )

    validated = []

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise TypeError(
                f"Episode {index + 1} must be a dictionary."
            )

        missing = REQUIRED_FIELDS - set(result.keys())

        if missing:
            raise ValueError(
                f"Episode {index + 1} is missing fields: "
                f"{sorted(missing)}"
            )

        if not isinstance(result["success"], bool):
            raise TypeError(
                f"Episode {index + 1}: success must be True or False."
            )

        if not isinstance(result["collision"], bool):
            raise TypeError(
                f"Episode {index + 1}: collision must be True or False."
            )

        steps = result["steps"]

        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError(
                f"Episode {index + 1}: steps must be an integer."
            )

        if steps < 0:
            raise ValueError(
                f"Episode {index + 1}: steps cannot be negative."
            )

        reward = result["total_reward"]

        if isinstance(reward, bool) or not isinstance(
            reward, (int, float)
        ):
            raise TypeError(
                f"Episode {index + 1}: total_reward must be numeric."
            )

        validated.append({
            "success": result["success"],
            "steps": steps,
            "total_reward": float(reward),
            "collision": result["collision"],
        })

    return validated


def evaluate_agent(
    episode_results: Iterable[Dict[str, Any]],
    algorithm_name: str = "Agent",
) -> Dict[str, Any]:
    """
    Calculate performance metrics for one algorithm.

    Parameters
    ----------
    episode_results:
        Iterable of episode dictionaries.

    algorithm_name:
        Name to show in comparison tables.

    Returns
    -------
    Dictionary containing:
        algorithm
        episodes
        successes
        success_rate
        average_steps
        average_reward
        collision_episodes
        collision_rate
    """
    results = validate_episode_results(episode_results)

    total_episodes = len(results)

    successes = sum(
        result["success"] for result in results
    )

    collision_episodes = sum(
        result["collision"] for result in results
    )

    success_rate = (
        successes / total_episodes
    ) * 100

    collision_rate = (
        collision_episodes / total_episodes
    ) * 100

    average_steps = sum(
        result["steps"] for result in results
    ) / total_episodes

    average_reward = sum(
        result["total_reward"] for result in results
    ) / total_episodes

    return {
        "algorithm": algorithm_name,
        "episodes": total_episodes,
        "successes": successes,
        "success_rate": round(success_rate, 2),
        "average_steps": round(average_steps, 2),
        "average_reward": round(average_reward, 2),
        "collision_episodes": collision_episodes,
        "collision_rate": round(collision_rate, 2),
    }


def compare_algorithms(
    q_learning_results: Iterable[Dict[str, Any]],
    sarsa_results: Iterable[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compare Q-learning and SARSA using episode results.

    Both inputs should contain results from real evaluation
    episodes. For a fair comparison, use the same test
    scenarios and evaluation settings.
    """
    q_metrics = evaluate_agent(
        q_learning_results,
        algorithm_name="Q-learning",
    )

    sarsa_metrics = evaluate_agent(
        sarsa_results,
        algorithm_name="SARSA",
    )

    columns = [
        "algorithm",
        "episodes",
        "success_rate",
        "average_steps",
        "average_reward",
        "collision_rate",
    ]

    return pd.DataFrame(
        [q_metrics, sarsa_metrics]
    )[columns].rename(columns={
        "algorithm": "Algorithm",
        "episodes": "Episodes",
        "success_rate": "Success Rate (%)",
        "average_steps": "Average Steps",
        "average_reward": "Average Reward",
        "collision_rate": "Collision Rate (%)",
    })


def create_episode_dataframe(
    episode_results: Iterable[Dict[str, Any]],
    algorithm_name: str = "Agent",
) -> pd.DataFrame:
    """
    Create a row-per-episode DataFrame for plotting
    learning curves or inspecting evaluation results.
    """
    results = validate_episode_results(episode_results)

    dataframe = pd.DataFrame(results)

    dataframe.insert(
        0,
        "episode",
        range(1, len(dataframe) + 1),
    )

    dataframe.insert(
        1,
        "algorithm",
        algorithm_name,
    )

    dataframe["success_rate_so_far"] = (
        dataframe["success"].astype(float).expanding().mean() * 100
    )

    dataframe["average_reward_so_far"] = (
        dataframe["total_reward"].expanding().mean()
    )

    dataframe["average_steps_so_far"] = (
        dataframe["steps"].expanding().mean()
    )

    dataframe["collision_rate_so_far"] = (
        dataframe["collision"].astype(float).expanding().mean() * 100
    )

    return dataframe


def print_evaluation_report(
    metrics: Dict[str, Any],
) -> None:
    """Print a readable summary of evaluation metrics."""
    print("\nEvaluation Report")
    print("=" * 35)
    print(f"Algorithm: {metrics['algorithm']}")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Successful episodes: {metrics['successes']}")
    print(f"Success rate: {metrics['success_rate']:.2f}%")
    print(f"Average steps: {metrics['average_steps']:.2f}")
    print(f"Average reward: {metrics['average_reward']:.2f}")
    print(
        "Episodes with collisions: "
        f"{metrics['collision_episodes']}"
    )
    print(f"Collision rate: {metrics['collision_rate']:.2f}%")
    print("=" * 35)


if __name__ == "__main__":
    # Small synthetic example for testing the evaluator.
    # These are fabricated test cases, NOT experimental results.
    example_episodes = [
        {
            "success": True,
            "steps": 18,
            "total_reward": 75.0,
            "collision": False,
        },
        {
            "success": True,
            "steps": 20,
            "total_reward": 70.0,
            "collision": False,
        },
        {
            "success": False,
            "steps": 30,
            "total_reward": -20.0,
            "collision": True,
        },
    ]

    example_metrics = evaluate_agent(
        example_episodes,
        algorithm_name="Example agent",
    )

    print_evaluation_report(example_metrics)

    print("\nPer-episode data:")
    print(create_episode_dataframe(example_episodes))

    print(
        "\nThe values above are synthetic test cases. "
        "Replace them with actual environment results."
    )
