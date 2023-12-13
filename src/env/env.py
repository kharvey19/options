import gymnasium as gym
import numpy as np
from termcolor import colored


class FourRooms(gym.Env):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.MultiDiscrete((13, 13))
        self.action_space = gym.spaces.Discrete(4)
        self.max_steps = 1000

    def reset(self, seed=None):
        self.agent_state = (1, 1)
        self.goal_state = (9, 9)
        self.step_count = 0

        self._init_map()
        return self.agent_state, {}

    def step(self, action):
        self.step_count += 1

        if action == self.RIGHT:
            next_state = (self.agent_state[0], self.agent_state[1] + 1)
        elif action == self.LEFT:
            next_state = (self.agent_state[0], self.agent_state[1] - 1)
        elif action == self.UP:
            next_state = (self.agent_state[0] - 1, self.agent_state[1])
        elif action == self.DOWN:
            next_state = (self.agent_state[0] + 1, self.agent_state[1])
        else:
            raise ValueError("Invalid action")

        if self.map[next_state] == 0:
            self.agent_state = next_state
        reward = self.agent_state == self.goal_state
        truncated = self.step_count >= self.max_steps
        terminated = reward == 1 or truncated
        info = {}
        return self.agent_state, reward, terminated, truncated, info

    def render(self):
        # print the map with walls in red in terminal
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if (i, j) == self.agent_state:
                    print(colored("A", "blue"), end=" ")
                elif (i, j) == self.goal_state:
                    print(colored("G", "red"), end=" ")
                elif self.map[i, j] == 1:
                    print(colored("x", "green"), end=" ")
                else:
                    print(".", end=" ")
            print()

    def _init_map(self):
        self.map = np.zeros((13, 13), dtype=np.int32)

        # Outer walls
        self.map[0, :] = 1
        self.map[12, :] = 1
        self.map[:, 0] = 1
        self.map[:, 12] = 1

        # Inner walls
        self.map[:, 6] = 1
        self.map[6, :6] = 1
        self.map[7, 6:] = 1

        # Hallways
        self.map[3, 6] = 0
        self.map[6, 2] = 0
        self.map[7, 9] = 0
        self.map[10, 6] = 0
