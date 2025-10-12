"""Hallway options for the Four Rooms environment.

Each hallway option takes the agent to one of the four hallway cells.
The hallway cells are at positions: (3, 6), (6, 2), (7, 9), (10, 6)
"""

from typing import Dict, List, Tuple

# Action constants
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3


class HallwayOption:
    """Represents a hallway option that navigates to a specific hallway cell."""
    
    def __init__(
        self,
        name: str,
        target_hallway: Tuple[int, int],
        room_cells: List[Tuple[int, int]],
        policy: Dict[Tuple[int, int], int],
    ):
        self.name = name
        self.target_hallway = target_hallway
        self.room_cells = set(room_cells)  # States where this option is valid
        self.policy = policy
        self.termination_set = {target_hallway}
    
    def can_initiate(self, state: Tuple[int, int]) -> bool:
        """Check if option can be initiated from this state.
        
        Can initiate from inside the room OR at the target hallway (for value propagation).
        """
        return (state in self.room_cells) or (state == self.target_hallway)
    
    def get_action(self, state: Tuple[int, int]) -> int:
        """Get the action to take in this state.
        
        Raises KeyError if state is not in the option's policy.
        """
        if state not in self.policy:
            raise KeyError(f"State {state} not in policy for option {self.name}")
        return self.policy[state]
    
    def terminates_at(self, state: Tuple[int, int]) -> bool:
        """Check if option terminates at this state.
        
        Terminates when reached the target hallway.
        """
        return state in self.termination_set


def get_hallway_options() -> List[HallwayOption]:
    """Get all 8 hallway options for the Four Rooms environment.
    
    Each room has 2 options corresponding to the 2 hallways accessible from that room.
    Returns 8 options total (4 rooms × 2 hallways each).
    """
    options = []
    
    # Room 1 (top-left): rows 1-5, cols 1-5
    # Can reach hallways at (3, 6) and (6, 2)
    
    # Room 1 → Hallway (3, 6) [right hallway]
    room1_cells = [(r, c) for r in range(1, 6) for c in range(1, 6)]
    room1_to_right = HallwayOption(
        name="Room1→Right",
        target_hallway=(3, 6),
        room_cells=room1_cells,
        policy={
            (1, 1): RIGHT, (1, 2): RIGHT, (1, 3): RIGHT, (1, 4): RIGHT, (1, 5): DOWN,
            (2, 1): RIGHT, (2, 2): RIGHT, (2, 3): RIGHT, (2, 4): RIGHT, (2, 5): DOWN,
            (3, 1): RIGHT, (3, 2): RIGHT, (3, 3): RIGHT, (3, 4): RIGHT, (3, 5): RIGHT,
            (4, 1): RIGHT, (4, 2): RIGHT, (4, 3): RIGHT, (4, 4): RIGHT, (4, 5): UP,
            (5, 1): RIGHT, (5, 2): RIGHT, (5, 3): RIGHT, (5, 4): RIGHT, (5, 5): UP,
        }
    )
    options.append(room1_to_right)
    
    # Room 1 → Hallway (6, 2) [bottom hallway]
    room1_to_bottom = HallwayOption(
        name="Room1→Bottom",
        target_hallway=(6, 2),
        room_cells=room1_cells,
        policy={
            (1, 1): DOWN, (1, 2): DOWN, (1, 3): LEFT, (1, 4): LEFT, (1, 5): LEFT,
            (2, 1): DOWN, (2, 2): DOWN, (2, 3): LEFT, (2, 4): LEFT, (2, 5): LEFT,
            (3, 1): DOWN, (3, 2): DOWN, (3, 3): LEFT, (3, 4): LEFT, (3, 5): LEFT,
            (4, 1): DOWN, (4, 2): DOWN, (4, 3): LEFT, (4, 4): LEFT, (4, 5): LEFT,
            (5, 1): RIGHT, (5, 2): DOWN, (5, 3): LEFT, (5, 4): LEFT, (5, 5): LEFT,
        }
    )
    options.append(room1_to_bottom)
    
    # Room 2 (top-right): rows 1-6, cols 7-11
    # Can reach hallways at (3, 6) and (7, 9)
    
    # Room 2 → Hallway (3, 6) [left hallway]
    room2_cells = [(r, c) for r in range(1, 7) for c in range(7, 12)]
    room2_to_left = HallwayOption(
        name="Room2→Left",
        target_hallway=(3, 6),
        room_cells=room2_cells,
        policy={
            (1, 7): LEFT, (1, 8): LEFT, (1, 9): DOWN, (1, 10): LEFT, (1, 11): LEFT,
            (2, 7): LEFT, (2, 8): LEFT, (2, 9): DOWN, (2, 10): LEFT, (2, 11): LEFT,
            (3, 7): LEFT, (3, 8): LEFT, (3, 9): LEFT, (3, 10): LEFT, (3, 11): LEFT,
            (4, 7): LEFT, (4, 8): LEFT, (4, 9): UP, (4, 10): LEFT, (4, 11): LEFT,
            (5, 7): LEFT, (5, 8): LEFT, (5, 9): UP, (5, 10): LEFT, (5, 11): LEFT,
            (6, 7): LEFT, (6, 8): LEFT, (6, 9): UP, (6, 10): LEFT, (6, 11): LEFT,
        }
    )
    options.append(room2_to_left)
    
    # Room 2 → Hallway (7, 9) [bottom hallway]
    room2_to_bottom = HallwayOption(
        name="Room2→Bottom",
        target_hallway=(7, 9),
        room_cells=room2_cells,
        policy={
            (1, 7): DOWN, (1, 8): DOWN, (1, 9): DOWN, (1, 10): LEFT, (1, 11): LEFT,
            (2, 7): DOWN, (2, 8): DOWN, (2, 9): DOWN, (2, 10): LEFT, (2, 11): LEFT,
            (3, 7): DOWN, (3, 8): DOWN, (3, 9): DOWN, (3, 10): LEFT, (3, 11): LEFT,
            (4, 7): DOWN, (4, 8): DOWN, (4, 9): DOWN, (4, 10): LEFT, (4, 11): LEFT,
            (5, 7): DOWN, (5, 8): DOWN, (5, 9): DOWN, (5, 10): LEFT, (5, 11): LEFT,
            (6, 7): DOWN, (6, 8): DOWN, (6, 9): DOWN, (6, 10): LEFT, (6, 11): LEFT,
        }
    )
    options.append(room2_to_bottom)
    
    # Room 3 (bottom-left): rows 7-11, cols 1-5
    # Can reach hallways at (6, 2) and (10, 6)
    
    # Room 3 → Hallway (6, 2) [top hallway]
    room3_cells = [(r, c) for r in range(7, 12) for c in range(1, 6)]
    room3_to_top = HallwayOption(
        name="Room3→Top",
        target_hallway=(6, 2),
        room_cells=room3_cells,
        policy={
            (7, 1): UP, (7, 2): UP, (7, 3): LEFT, (7, 4): LEFT, (7, 5): LEFT,
            (8, 1): UP, (8, 2): UP, (8, 3): LEFT, (8, 4): LEFT, (8, 5): LEFT,
            (9, 1): UP, (9, 2): UP, (9, 3): LEFT, (9, 4): LEFT, (9, 5): LEFT,
            (10, 1): UP, (10, 2): UP, (10, 3): LEFT, (10, 4): LEFT, (10, 5): LEFT,
            (11, 1): UP, (11, 2): UP, (11, 3): LEFT, (11, 4): LEFT, (11, 5): LEFT,
        }
    )
    options.append(room3_to_top)
    
    # Room 3 → Hallway (10, 6) [right hallway]
    room3_to_right = HallwayOption(
        name="Room3→Right",
        target_hallway=(10, 6),
        room_cells=room3_cells,
        policy={
            (7, 1): RIGHT, (7, 2): RIGHT, (7, 3): RIGHT, (7, 4): RIGHT, (7, 5): DOWN,
            (8, 1): RIGHT, (8, 2): RIGHT, (8, 3): RIGHT, (8, 4): RIGHT, (8, 5): DOWN,
            (9, 1): RIGHT, (9, 2): RIGHT, (9, 3): RIGHT, (9, 4): RIGHT, (9, 5): DOWN,
            (10, 1): RIGHT, (10, 2): RIGHT, (10, 3): RIGHT, (10, 4): RIGHT, (10, 5): RIGHT,
            (11, 1): UP, (11, 2): UP, (11, 3): UP, (11, 4): UP, (11, 5): UP,
        }
    )
    options.append(room3_to_right)
    
    # Room 4 (bottom-right): rows 8-11, cols 7-11
    # Can reach hallways at (7, 9) and (10, 6)
    
    # Room 4 → Hallway (7, 9) [top hallway]
    room4_cells = [(r, c) for r in range(8, 12) for c in range(7, 12)]
    room4_to_top = HallwayOption(
        name="Room4→Top",
        target_hallway=(7, 9),
        room_cells=room4_cells,
        policy={
            (8, 7): RIGHT, (8, 8): RIGHT, (8, 9): UP, (8, 10): LEFT, (8, 11): LEFT,
            (9, 7): UP, (9, 8): UP, (9, 9): UP, (9, 10): UP, (9, 11): UP,
            (10, 7): UP, (10, 8): UP, (10, 9): UP, (10, 10): UP, (10, 11): UP,
            (11, 7): UP, (11, 8): UP, (11, 9): UP, (11, 10): UP, (11, 11): UP,
        }
    )
    options.append(room4_to_top)
    
    # Room 4 → Hallway (10, 6) [left hallway]
    room4_to_left = HallwayOption(
        name="Room4→Left",
        target_hallway=(10, 6),
        room_cells=room4_cells,
        policy={
            (8, 7): DOWN, (8, 8): LEFT, (8, 9): LEFT, (8, 10): LEFT, (8, 11): LEFT,
            (9, 7): DOWN, (9, 8): LEFT, (9, 9): LEFT, (9, 10): LEFT, (9, 11): LEFT,
            (10, 7): LEFT, (10, 8): LEFT, (10, 9): LEFT, (10, 10): LEFT, (10, 11): LEFT,
            (11, 7): UP, (11, 8): LEFT, (11, 9): LEFT, (11, 10): LEFT, (11, 11): LEFT,
        }
    )
    options.append(room4_to_left)
    
    return options


def get_primitive_actions() -> List[int]:
    """Get the 4 primitive actions."""
    return [RIGHT, LEFT, UP, DOWN]


ACTION_NAMES = {RIGHT: "→", LEFT: "←", UP: "↑", DOWN: "↓"}

