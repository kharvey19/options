import numpy as np
from tqdm import trange

from src.env.env import FourRooms


def train_qlearning():
    env = FourRooms()
    Q = np.zeros(
        (
            env.observation_space.nvec[0],
            env.observation_space.nvec[1],
            env.action_space.n,
        )
    )

    rewards = []
    pbar = trange(50000)
    for episode in pbar:
        obs, _ = env.reset()
        done = False

        while not done:
            if np.random.random() < 0.1:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[obs])

            next_obs, reward, done, _, _ = env.step(action)

            if done:
                Q[obs][action] += 0.01 * (reward - Q[obs][action])
            else:
                Q[obs][action] += 0.01 * (
                    reward + 0.99 * np.max(Q[next_obs]) - Q[obs][action]
                )
            obs = next_obs
        rewards.append(reward)

        if episode % 1000 == 0:
            pbar.set_description(
                f"Episode {episode} | Reward {np.mean(rewards[-100:])}"
            )
    return np.array(rewards)
