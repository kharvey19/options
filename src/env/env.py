import gymnasium as gym
import numpy as np
from termcolor import colored


class FourRooms(gym.Env[tuple[int, int], np.int32]):
    """
    FourRooms environment class.

    This class represents a grid-world environment with four rooms.
    The agent's goal is to navigate from the starting
    position to the goal position while avoiding obstacles.

    Attributes:
        RIGHT (int): Action constant representing moving right.
        LEFT (int): Action constant representing moving left.
        UP (int): Action constant representing moving up.
        DOWN (int): Action constant representing moving down.

    Methods:
        __init__(): Initializes the FourRooms environment.
        reset(seed=None): Resets the environment to its initial state.
        step(action): Takes a step in the environment based on the given action.
        render(): Renders the current state of the environment.
        _init_map(): Initializes the map of the environment.

    """

    RIGHT: int = 0
    LEFT: int = 1
    UP: int = 2
    DOWN: int = 3

    def __init__(self, **kwargs: dict[any]) -> None:
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.MultiDiscrete((13, 13))
        self.action_space = gym.spaces.Discrete(4)
        self.max_steps = 10000
        self.agent_state = (0, 0)
        self.goal_state = (0, 0)
        self.step_count = 0
        self.map = np.zeros((13, 13), dtype=np.int32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, any] | None = None,
    ) -> tuple[tuple[int, int], dict[any, any]]:
        self.agent_state = (7, 9)
        self.goal_state = (9, 9)
        self.step_count = 0

        self._init_map()
        return self.agent_state, {}

    def step(
        self, action: int
    ) -> tuple[tuple[int, int], int, bool, bool, dict[any, any]]:
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
        reward = 1 if self.agent_state == self.goal_state else 0
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
