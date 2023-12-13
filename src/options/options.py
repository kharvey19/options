from itertools import product

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3


class Option:
    """Option class represents an option in a reinforcement learning setting.

    An option is a temporally extended action that can be taken by an agent.
    It consists of an initiation set, a policy, and a termination set.
    The initiation set specifies the states in which the option can be initiated.
    The policy maps states to actions, determining the action to take in each state.
    The termination set specifies the states in which the option can be terminated.

    Attributes:
        initiation_set (list[tuple[int, int]]): The set of states in
            which the option can be initiated.
        policy (dict[tuple[int, int], int]): The policy mapping
            states to actions.
        termination_set (list[tuple[int, int]]): The set of states in
            which the option can be terminated.
    """

    def __init__(
        self,
        initiation_set: list[tuple[int, int]],
        policy: dict[tuple[int, int], int],
        termination_set: list[tuple[int, int]],
    ) -> None:
        self.initiation_set = initiation_set
        self.policy = policy
        self.termination_set = termination_set

    def is_available(self, obs: tuple[int, int]) -> bool:
        """Return if the option is availble to be taken in the given state."""
        return obs in self.initiation_set

    def is_terminated(self, obs: tuple[int, int]) -> bool:
        """Return if the option is terminated in the given state.

        In this case, the option is terminated if the state is in the termination set.
        In general, the option will terminate stochastically conditioned on the state. Thus,
        a conditional distribution should be sampled to determine if the option terminates.
        """
        return obs in self.termination_set

    def select_action(self, obs: tuple[int, int]) -> int:
        """Return the action to take in the given state."""
        return self.policy[obs]


class PrimitiveOption:
    """
    Represents a primitive option in an options framework.

    Args:
        action (int): The action associated with the option.

    Attributes:
        action (int): The action associated with the option.
    """

    def __init__(self, action: int):
        self.action = action

    def is_available(self, obs: tuple[int, int]) -> bool:
        """Checks if the option is available in the given observation."""
        return True

    def is_terminated(self, obs: tuple[int, int]) -> bool:
        """Checks if the option is terminated in the given observation."""
        return True

    def select_action(self, obs: tuple[int, int]) -> int:
        """Selects the action for the option based on the given observation."""
        return self.action


def create_room_one_option_one() -> Option:
    """
    Create option one for room one.

    This function creates an option for navigating through room one. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
        (6, 2) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (1, 1): RIGHT, (1, 2): RIGHT, (1, 3): RIGHT, (1, 4): RIGHT, (1, 5): DOWN,
        (2, 1): RIGHT, (2, 2): RIGHT, (2, 3): RIGHT, (2, 4): RIGHT, (2, 5): DOWN,
        (3, 1): RIGHT, (3, 2): RIGHT, (3, 3): RIGHT, (3, 4): RIGHT, (3, 5): RIGHT,
        (4, 1): RIGHT, (4, 2): RIGHT, (4, 3): RIGHT, (4, 4): RIGHT, (4, 5): UP,
        (5, 1): RIGHT, (5, 2): RIGHT, (5, 3): RIGHT, (5, 4): RIGHT, (5, 5): UP,
        (6, 2): UP
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_one_option_two() -> Option:
    """
    Create option two for room one.

    This function creates an option for navigating through room one. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
        (3, 6) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (1, 1): DOWN,  (1, 2): DOWN, (1, 3): LEFT, (1, 4): LEFT, (1, 5): LEFT,
        (2, 1): DOWN,  (2, 2): DOWN, (2, 3): LEFT, (2, 4): LEFT, (2, 5): LEFT,
        (3, 1): DOWN,  (3, 2): DOWN, (3, 3): LEFT, (3, 4): LEFT, (3, 5): LEFT,
        (4, 1): DOWN,  (4, 2): DOWN, (4, 3): LEFT, (4, 4): LEFT, (4, 5): LEFT,
        (5, 1): RIGHT, (5, 2): DOWN, (5, 3): LEFT, (5, 4): LEFT, (5, 5): LEFT,
        (3, 6): LEFT # This is the other hallway
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_two_option_one() -> Option:
    """
    Create option one for room two.

    This function creates an option for navigating through room two. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (1, 7), (1, 8), (1, 9), (1, 10), (1, 11),
        (2, 7), (2, 8), (2, 9), (2, 10), (2, 11),
        (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
        (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
        (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
        (3, 6) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (1, 7): DOWN,  (1, 8): DOWN, (1, 9): DOWN, (1, 10): DOWN, (1, 11): DOWN,
        (2, 7): DOWN,  (2, 8): DOWN, (2, 9): DOWN, (2, 10): DOWN, (2, 11): DOWN,
        (3, 7): DOWN,  (3, 8): DOWN, (3, 9): DOWN, (3, 10): DOWN, (3, 11): DOWN,
        (4, 7): DOWN,  (4, 8): DOWN, (4, 9): DOWN, (4, 10): DOWN, (4, 11): DOWN,
        (5, 7): DOWN,  (5, 8): DOWN, (5, 9): DOWN, (5, 10): DOWN, (5, 11): DOWN,
        (6, 7): RIGHT, (6, 8): RIGHT, (6, 9): DOWN, (6, 10): LEFT, (6, 11): LEFT,
        (3, 6): RIGHT # This is the other hallway
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_two_option_two() -> Option:
    """
    Create option two for room two.

    This function creates an option for navigating through room two. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (1, 7), (1, 8), (1, 9), (1, 10), (1, 11),
        (2, 7), (2, 8), (2, 9), (2, 10), (2, 11),
        (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
        (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
        (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
        (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
        (7, 9) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (1, 7): DOWN, (1, 8): LEFT, (1, 9): LEFT, (1, 10): LEFT, (1, 11): LEFT,
        (2, 7): DOWN, (2, 8): LEFT, (2, 9): LEFT, (2, 10): LEFT, (2, 11): LEFT,
        (3, 7): LEFT, (3, 8): LEFT, (3, 9): LEFT, (3, 10): LEFT, (3, 11): LEFT,
        (4, 7): UP,   (4, 8): LEFT, (4, 9): LEFT, (4, 10): LEFT, (4, 11): LEFT,
        (5, 7): UP,   (5, 8): LEFT, (5, 9): LEFT, (5, 10): LEFT, (5, 11): LEFT,
        (6, 7): UP,   (6, 8): LEFT, (6, 9): LEFT, (6, 10): LEFT, (6, 11): LEFT,
        (7, 9): UP # This is the other hallway
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_three_option_one() -> Option:
    """
    Create option one for room three.

    This function creates an option for navigating through room three. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
        (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
        (9, 1), (9, 2), (9, 3), (9, 4), (9, 5),
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
        (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
        (6, 2) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (7, 1): RIGHT,  (7, 2):  RIGHT, (7, 3):  RIGHT, (7, 4):  RIGHT, (7, 5):  DOWN,
        (8, 1): RIGHT,  (8, 2):  RIGHT, (8, 3):  RIGHT, (8, 4):  RIGHT, (8, 5):  DOWN,
        (9, 1): RIGHT,  (9, 2):  RIGHT, (9, 3):  RIGHT, (9, 4):  RIGHT, (9, 5):  DOWN,
        (10, 1): RIGHT, (10, 2): RIGHT, (10, 3): RIGHT, (10, 4): RIGHT, (10, 5): RIGHT,
        (11, 1): RIGHT, (11, 2): RIGHT, (11, 3): RIGHT, (11, 4): LEFT, (11, 5): UP,
        (6, 2): DOWN
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_three_option_two() -> Option:
    """
    Create option two for room three.

    This function creates an option for navigating through room three. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
        (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
        (9, 1), (9, 2), (9, 3), (9, 4), (9, 5),
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
        (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
        (10, 6) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (7, 1): RIGHT, (7, 2):  UP, (7, 3):  LEFT, (7, 4):  LEFT, (7, 5):  LEFT,
        (8, 1): UP,    (8, 2):  UP, (8, 3):  UP,   (8, 4):  UP,   (8, 5):  UP,
        (9, 1): UP,    (9, 2):  UP, (9, 3):  UP,   (9, 4):  UP,   (9, 5):  UP,
        (10, 1): UP,   (10, 2): UP, (10, 3): UP,   (10, 4): UP,   (10, 5): UP,
        (11, 1): UP,   (11, 2): UP, (11, 3): UP,   (11, 4): UP,   (11, 5): UP,
        (10, 6): LEFT
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_four_option_one() -> Option:
    """
    Create option one for room four.

    This function creates an option for navigating through room four. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (8, 7), (8, 8), (8, 9), (8, 10), (8, 11),
        (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
        (10, 7), (10, 8), (10, 9), (10, 10), (10, 11),
        (11, 7), (11, 8), (11, 9), (11, 10), (11, 11),
        (10, 6) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (8, 7): RIGHT, (8, 8):  RIGHT, (8, 9):  UP, (8, 10):  LEFT, (8, 11): LEFT,
        (9, 7): UP,    (9, 8):  UP,    (9, 9):  UP, (9, 10):  UP,   (9, 11):  UP,
        (10, 7): UP,   (10, 8): UP,    (10, 9): UP, (10, 10): UP,   (10, 11): UP,
        (11, 7): UP,   (11, 8): UP,    (11, 9): UP, (11, 10): UP,   (11, 11): UP,
        (10, 6): RIGHT
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)


def create_room_four_option_two() -> Option:
    """
    Create option two for room four.

    This function creates an option for navigating through room four. It defines the initiation set,
    termination set, and policy for the option.

    Returns:
        Option: The created option object.
    """
    # fmt: off
    initiation_set =[
        (8, 7), (8, 8), (8, 9), (8, 10), (8, 11),
        (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
        (10, 7), (10, 8), (10, 9), (10, 10), (10, 11),
        (11, 7), (11, 8), (11, 9), (11, 10), (11, 11),
        (7, 9) # This is the other hallway
    ]
    # fmt: on

    termination_set = []
    for row, col in product(range(13), repeat=2):
        if (row, col) not in initiation_set:
            termination_set.append((row, col))

    # fmt: off
    policy = {
        (8, 7): DOWN,  (8, 8):  LEFT, (8, 9):  LEFT, (8, 10):  LEFT, (8, 11): LEFT,
        (9, 7): DOWN,  (9, 8):  LEFT, (9, 9):  LEFT, (9, 10):  LEFT, (9, 11):  LEFT,
        (10, 7): LEFT, (10, 8): LEFT, (10, 9): LEFT, (10, 10): LEFT, (10, 11): LEFT,
        (11, 7): UP,   (11, 8): LEFT, (11, 9): LEFT, (11, 10): LEFT, (11, 11): LEFT,
        (7, 9): DOWN
    }
    # fmt: on
    return Option(initiation_set, policy, termination_set)
