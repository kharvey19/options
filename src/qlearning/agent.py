import gymnasium as gym
import numpy as np
from tqdm import trange


def train_qlearning(
    env: gym.Env[tuple[int, int], int], n_episodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trains a Q-learning agent on the given environment.

    Args:
        env (gym.Env): The environment to train the agent on.
        n_episodes (int): The number of episodes to train for.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array containing the rewards obtained in each episode.
            - np.ndarray: An array containing the number of actions taken in each episode.
    """
    Q = np.zeros(
        (
            env.observation_space.nvec[0],
            env.observation_space.nvec[1],
            env.action_space.n,
        )
    )

    rewards = []
    steps = []
    pbar = trange(n_episodes)
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        reward = 0
        step_count = 0

        while not done:
            if np.random.random() < 0.1:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[obs])

            next_obs, reward, done, _, _ = env.step(action)
            step_count += 1

            if done:
                Q[obs][action] += 0.01 * (reward - Q[obs][action])
            else:
                Q[obs][action] += 0.01 * (
                    reward + 0.99 * np.max(Q[next_obs]) - Q[obs][action]
                )
            obs = next_obs
        rewards.append(reward)
        steps.append(step_count)

        if episode % 1000 == 0:
            pbar.set_description(
                f"Episode {episode} | Reward {np.mean(rewards[-100:])} | Steps {np.mean(steps[-100:]):.1f}"
            )
    return np.array(rewards), np.array(steps)
