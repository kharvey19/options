from typing import Tuple

import gymnasium as gym
import numpy as np
from tqdm import trange

from .options import (
    Option,
    PrimitiveOption,
    create_room_four_option_one,
    create_room_four_option_two,
    create_room_one_option_one,
    create_room_one_option_two,
    create_room_three_option_one,
    create_room_three_option_two,
    create_room_two_option_one,
    create_room_two_option_two,
)


def train_options(
    env: gym.Env[np.ndarray[np.float64], int],
    n_episodes: int,
) -> np.ndarray[np.float64]:
    """Train options for the FourRooms environment."""
    options = _load_options(env)
    Q = np.zeros(
        (
            env.observation_space.nvec[0],
            env.observation_space.nvec[1],
            len(options),
        )
    )

    rewards = []
    pbar = trange(n_episodes)
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            option, option_idx = _get_option(obs, options, Q)
            start_obs = obs
            final_obs, done, option_return, option_steps = _execute_option(
                obs, option, env, done
            )

            if done:
                Q[start_obs][option_idx] += 0.01 * (
                    option_return - Q[start_obs][option_idx]
                )
            else:
                Q[start_obs][option_idx] += 0.01 * (
                    option_return
                    + 0.99**option_steps * np.max(Q[final_obs])
                    - Q[start_obs][option_idx]
                )
            obs = final_obs
            episode_return += option_return
        rewards.append(episode_return)

        if episode % 100 == 0:
            pbar.set_description(
                f"Episode {episode} | Reward {np.mean(rewards[-100:])}"
            )
    return np.array(rewards)


def _load_options(
    env: gym.Env[np.ndarray[np.float64], int]
) -> Tuple[Option | PrimitiveOption, ...]:
    """Load options for the FourRooms environment."""
    return (
        create_room_one_option_one(),
        create_room_one_option_two(),
        create_room_two_option_one(),
        create_room_two_option_two(),
        create_room_three_option_one(),
        create_room_three_option_two(),
        create_room_four_option_one(),
        create_room_four_option_two(),
        PrimitiveOption(env.RIGHT),
        PrimitiveOption(env.LEFT),
        PrimitiveOption(env.UP),
        PrimitiveOption(env.DOWN),
    )


def _get_option(
    obs: tuple[int, int],
    options: tuple[Option | PrimitiveOption, ...],
    Q: np.ndarray[np.float64],
) -> tuple[Option | PrimitiveOption, int]:
    """Get an option for the given observation."""
    option, option_idx = options[0], 0
    option_valid = False

    while not option_valid:
        if np.random.random() < 0.1:
            option_idx = np.random.randint(len(options))
        else:
            option_idx = np.argmax(Q[obs])
        option = options[option_idx]

        option_valid = option.is_available(obs)
    return option, option_idx


def _execute_option(
    obs: tuple[int, int],
    option: Option | PrimitiveOption,
    env: gym.Env[np.ndarray[np.float64], int],
    done: bool,
) -> tuple[tuple[int, int], bool, float, int]:
    """Execute the given option for the given observation."""
    step_count = 0
    option_return = 0
    one_action_taken = False
    while not one_action_taken or (not option.is_terminated(obs) and not done):
        action = option.select_action(obs)
        obs_next, reward, done, _, _ = env.step(action)

        option_return += 0.99**step_count * reward
        step_count += 1
        obs = obs_next
        one_action_taken = True
    return obs, done, option_return, step_count
