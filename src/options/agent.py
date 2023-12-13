import numpy as np
from env.env import FourRooms
from tqdm import trange

from options import (
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

if __name__ == "__main__":
    env = FourRooms()
    options = [
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
    ]
    Q = np.zeros(
        (
            env.observation_space.nvec[0],
            env.observation_space.nvec[1],
            len(options),
        )
    )

    rewards = []
    pbar = trange(10000)
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            option_valid = False

            while not option_valid:
                if np.random.random() < 0.1:
                    option_idx = np.random.randint(len(options))
                else:
                    option_idx = np.argmax(Q[obs])
                option = options[option_idx]

                option_valid = option.is_available(obs)

            start_obs = obs
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

            if done:
                Q[start_obs][option_idx] += 0.01 * (
                    option_return - Q[start_obs][option_idx]
                )
            else:
                Q[start_obs][option_idx] += 0.01 * (
                    option_return
                    + 0.99**step_count * np.max(Q[obs])
                    - Q[start_obs][option_idx]
                )
            episode_return += option_return
        rewards.append(episode_return)

        pbar.set_description(f"Episode {episode} | Reward {np.mean(rewards[-100:])}")
