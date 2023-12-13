import matplotlib.pyplot as plt
import numpy as np

from env import FourRooms
from options import train_options
from qlearning import train_qlearning

if __name__ == "__main__":
    env = FourRooms()

    options_rewards = []
    qlearning_rewards = []

    for _ in range(30):
        options_rewards.append(train_options(env=env, n_episodes=50000))
        qlearning_rewards.append(train_qlearning(env=env, n_episodes=50000))

    options_rewards = np.array(options_rewards).mean(axis=0)
    qlearning_rewards = np.array(qlearning_rewards).mean(axis=0)

    window = 100
    options_smooth = np.convolve(
        options_rewards, np.ones(window) / window, mode="valid"
    )
    qlearning_smooth = np.convolve(
        qlearning_rewards, np.ones(window) / window, mode="valid"
    )

    plt.plot(options_smooth, label="Options")
    plt.plot(qlearning_smooth, label="Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Options vs. Q-Learning in FourRooms Environment")
    plt.legend()
    plt.savefig("results.png")
    plt.show()
