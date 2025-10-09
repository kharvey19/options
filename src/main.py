import matplotlib.pyplot as plt
import numpy as np

from env import FourRooms
from options import train_options
from qlearning import train_qlearning

if __name__ == "__main__":
    env = FourRooms()

    options_rewards = []
    options_steps = []
    qlearning_rewards = []
    qlearning_steps = []

    for _ in range(30):
        opt_rew, opt_steps = train_options(env=env, n_episodes=30000)
        options_rewards.append(opt_rew)
        options_steps.append(opt_steps)
        
        qlearn_rew, qlearn_steps = train_qlearning(env=env, n_episodes=30000)
        qlearning_rewards.append(qlearn_rew)
        qlearning_steps.append(qlearn_steps)

    options_rewards = np.array(options_rewards).mean(axis=0)
    options_steps = np.array(options_steps).mean(axis=0)
    qlearning_rewards = np.array(qlearning_rewards).mean(axis=0)
    qlearning_steps = np.array(qlearning_steps).mean(axis=0)

    window = 100
    
    # Smooth rewards
    options_rewards_smooth = np.convolve(
        options_rewards, np.ones(window) / window, mode="valid"
    )
    qlearning_rewards_smooth = np.convolve(
        qlearning_rewards, np.ones(window) / window, mode="valid"
    )
    
    # Smooth steps
    options_steps_smooth = np.convolve(
        options_steps, np.ones(window) / window, mode="valid"
    )
    qlearning_steps_smooth = np.convolve(
        qlearning_steps, np.ones(window) / window, mode="valid"
    )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    ax1.plot(options_rewards_smooth, label="Options")
    ax1.plot(qlearning_rewards_smooth, label="Q-Learning")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot number of actions
    ax2.plot(options_steps_smooth, label="Options")
    ax2.plot(qlearning_steps_smooth, label="Q-Learning")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Number of Actions")
    ax2.set_title("Number of Actions per Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
