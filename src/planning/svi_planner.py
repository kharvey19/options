"""Synchronous Value Iteration planner for Four Rooms with options.

This module implements planning using synchronous value iteration (SVI).
It can plan with either primitive actions only or with hallway options.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Union
from .hallway_options import HallwayOption, get_primitive_actions


class SVIPlanner:
    """Synchronous Value Iteration planner.
    
    This planner can work with either:
    - Primitive actions only (O = A)
    - Hallway options (O = H)
    - Both (O = A + H)
    
    Args:
        env: The Four Rooms environment
        options: List of options (can be primitive actions or hallway options)
        gamma: Discount factor (default: 0.99)
        theta: Convergence threshold (default: 1e-6)
    """
    
    def __init__(
        self,
        env: Any,
        options: Union[List[int], List[HallwayOption]],
        gamma: float = 0.99,
        theta: float = 1e-6,
    ):
        self.env = env
        self.options = options
        self.gamma = gamma
        self.theta = theta
        
        # Value function v(s) for all states
        self.values = np.zeros((13, 13))
        
        # Policy π(s) -> option index
        self.policy = np.zeros((13, 13), dtype=int)
        
        # Track convergence
        self.num_iterations = 0
        self.value_history = []
        
        # Track planning steps (weighted by initiation set size for options)
        self.planning_steps = 0
        self.planning_steps_history = []
        
        # Precompute SMDP models for all options (primitives and hallway options)
        self.option_models = self._compute_all_option_models()
    
    def plan(self, goal_state: Tuple[int, int], max_iterations: int = 1000, fixed_sweeps: bool = False) -> int:
        """Run synchronous value iteration until convergence or for fixed sweeps.
        
        Args:
            goal_state: The goal state (r, c)
            max_iterations: Maximum number of iterations (or exact count if fixed_sweeps=True)
            fixed_sweeps: If True, run exactly max_iterations sweeps without early stopping
            
        Returns:
            Number of iterations until convergence (or max_iterations if fixed_sweeps=True)
        """
        self.values = np.zeros((13, 13))
        self.policy = np.zeros((13, 13), dtype=int)
        self.num_iterations = 0
        self.value_history = []
        self.planning_steps = 0
        self.planning_steps_history = []
        
        for _ in range(max_iterations):
            delta = 0
            old_values = self.values.copy()
            sweep_steps = 0  # Planning steps for this sweep
            
            # Update value for each state
            for r in range(13):
                for c in range(13):
                    state = (r, c)
                    
                    # Skip walls
                    if not self._is_valid_state(state):
                        continue
                    
                    # Goal state has value 0 (already at goal)
                    if state == goal_state:
                        self.values[r, c] = 0
                        continue
                    
                    # Compute value for each available option (SMDP backup)
                    # Track best option with tie-breaking favoring options over primitives
                    best_value = -np.inf
                    best_option = -1
                    best_is_primitive = True
                    
                    for option_idx in range(len(self.options)):
                        model = self.option_models[option_idx]
                        
                        # Check if option can initiate from this state
                        if state not in model['I']:
                            continue
                        
                        # Count this (s,o) backup - each evaluation counts as 1 step
                        sweep_steps += 1
                        
                        # Compute Q(s, o) using SMDP backup
                        if model['is_primitive']:
                            # Primitive: use 1-step MDP backup
                            action = model['action']
                            value = self._compute_primitive_value(state, action, goal_state, old_values)
                            is_primitive = True
                        else:
                            # Hallway option: use multi-step SMDP backup
                            value = self._compute_option_value(state, option_idx, goal_state, old_values)
                            is_primitive = False
                        
                        # Tie-breaking: prefer options over primitives
                        # Update if: (1) strictly better, or (2) tied but switching from primitive to option
                        if value > best_value or (np.isclose(value, best_value, rtol=1e-9) and best_is_primitive and not is_primitive):
                            best_value = value
                            best_option = option_idx
                            best_is_primitive = is_primitive
                    
                    # Only update if best_value is finite (not -inf)
                    # If all options are invalid, keep old value
                    if np.isfinite(best_value) and best_option >= 0:
                        self.values[r, c] = best_value
                        self.policy[r, c] = best_option
                    else:
                        self.values[r, c] = old_values[r, c]
                    
                    # Track maximum change (only for finite values)
                    val_old = old_values[r, c]
                    val_new = self.values[r, c]
                    if np.isfinite(val_old) and np.isfinite(val_new):
                        delta = max(delta, abs(val_new - val_old))
            
            self.num_iterations += 1
            self.planning_steps += sweep_steps
            
            # Track delta (max-norm change) - this is the actual convergence criterion
            self.value_history.append(delta)
            self.planning_steps_history.append(self.planning_steps)
            
            # Check convergence (skip if running fixed sweeps)
            if not fixed_sweeps and delta < self.theta:
                break
        
        return self.num_iterations
    
    def _is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is valid (not a wall)."""
        r, c = state
        
        # Out of bounds
        if r < 0 or r >= 13 or c < 0 or c >= 13:
            return False
        
        # Outer walls
        if r == 0 or r == 12 or c == 0 or c == 12:
            return False
        
        # Inner walls (blocking the rooms)
        # Vertical wall at column 6 (except hallways at rows 3 and 10)
        if c == 6 and r not in [3, 10]:
            return False
        
        # Horizontal wall at row 6, columns 0-5 (except hallway at column 2)
        if r == 6 and c < 6 and c != 2:
            return False
        
        # Horizontal wall at row 7, columns 6-12 (except hallway at column 9)
        if r == 7 and c >= 6 and c != 9:
            return False
        
        return True
    
    def _compute_all_option_models(self):
        """Compute SMDP models for all options (primitives and hallway options).
        
        For primitives: 1-step SMDP with r_a(s) = R(s,a), P_a(s,s') = γ * P(s'|s,a)
        For hallway options: Multi-step SMDP using fundamental matrix
        
        Returns:
            Dictionary mapping option index to SMDP model
        """
        models = {}
        
        for opt_idx, option in enumerate(self.options):
            if isinstance(option, int):
                # Primitive action - build 1-step SMDP model
                models[opt_idx] = self._build_primitive_smdp_model(option)
            else:
                # Hallway option - build multi-step SMDP model
                models[opt_idx] = self._build_hallway_option_model(option)
        
        return models
    
    def _build_primitive_smdp_model(self, action: int):
        """Build SMDP model for a primitive action (1-step option).
        
        Returns:
            Dictionary with 'I' (all valid states), 'r', 'P' in SMDP form
        """
        # Primitives can initiate from all valid states
        I = []
        for r in range(13):
            for c in range(13):
                if self._is_valid_state((r, c)):
                    I.append((r, c))
        
        model = {
            'I': I,
            'is_primitive': True,
            'action': action,
        }
        
        return model
    
    def _compute_primitive_value(
        self,
        state: Tuple[int, int],
        action: int,
        goal_state: Tuple[int, int],
        V_prev: np.ndarray
    ) -> float:
        # Expected immediate reward (no gamma here)
        r_expect = 0.0
        cont = 0.0

        for next_state, prob_env in self._get_stochastic_transitions(state, action):
            reward = 1.0 if next_state == goal_state else 0.0
            r_expect += prob_env * reward              # E[r_{t+1}]
            r_next, c_next = next_state
            cont += prob_env * V_prev[r_next, c_next]  # E[V(s_{t+1})]

        return r_expect + self.gamma * cont

    
    def _build_hallway_option_model(self, option: HallwayOption):
        """Build SMDP model for a single hallway option using the fundamental matrix approach.
        
        For each option, computes:
        - r_o^s: Expected discounted reward E[Σ_{t=0}^{τ-1} γ^t r_t | s, o]
        - p_ss'^o: Expected discounted termination E[γ^τ 1(s_τ = s') | s, o]
        
        where τ is the (random) option duration and the discount γ^τ properly couples
        with the stochastic termination time.
        
        The fundamental matrix approach automatically handles this:
        - N = (I - γP_II)^(-1) = Σ_{k=0}^∞ γ^k P_II^k
        - N_ij represents expected discounted occupancy of state j from state i
        - Rewards: r_o^s = N @ r_I captures Σ_{t=0}^{τ-1} γ^t r(s_t)
        - Termination: γ * N * P_IB captures Σ_{k=1}^∞ γ^k * P(exit at step k to s')
        
        Returns:
            Dictionary with model components for this option
        """
        # Get initiation states (I) and boundary states (B)
        I = list(option.room_cells)
        B_set = {option.target_hallway}  # Start with target hallway
        extra_B = set()  # Will collect other exits
        
        n_I = len(I)
        state_to_idx = {s: i for i, s in enumerate(I)}
        
        # First pass: discover all possible termination destinations
        # (any valid state outside the room, including other doorways reached via stochasticity)
        for i, s in enumerate(I):
            try:
                action = option.get_action(s)
            except KeyError:
                continue
            
            transitions = self._get_stochastic_transitions(s, action)
            for next_state, prob in transitions:
                # If next_state is outside the room and valid, it's a termination destination
                if next_state not in state_to_idx and self._is_valid_state(next_state):
                    extra_B.add(next_state)
        
        # Extend boundary set and index it
        B = list(B_set | extra_B)
        b_index = {b: j for j, b in enumerate(B)}
        
        # Build P_II: policy-induced transition matrix within I
        # P_II[i, j] = probability of transitioning from I[i] to I[j]
        P_II = np.zeros((n_I, n_I))
        
        # Build P_IB: transitions from I to B (all termination destinations)
        # P_IB[i, b] = probability of transitioning from I[i] to B[b]
        P_IB = np.zeros((n_I, len(B)))
        
        for i, s in enumerate(I):
            # Get the action prescribed by the option's policy
            try:
                action = option.get_action(s)
            except KeyError:
                # State not in policy - leave transitions as zero
                continue
            
            # Get stochastic transitions for this action
            transitions = self._get_stochastic_transitions(s, action)
            
            for next_state, prob in transitions:
                if next_state in state_to_idx:
                    # Transition within I
                    j = state_to_idx[next_state]
                    P_II[i, j] += prob
                elif self._is_valid_state(next_state):
                    # Termination to ANY valid outside state (hallway or other)
                    P_IB[i, b_index[next_state]] += prob
                else:
                    # Invalid move (wall) → stay in place (bounce)
                    # Add self-loop to make P_II properly stochastic
                    P_II[i, i] += prob
        
        # Compute fundamental matrix: fund_matrix = (I - γ * P_II)^(-1)
        # fund_matrix represents expected discounted occupancy: fund_matrix_ij = E[Σ_{t=0}^{τ-1} γ^t 1(s_t=j) | s_0=i]
        I_matrix = np.eye(n_I)
        try:
            fund_matrix = np.linalg.inv(I_matrix - self.gamma * P_II)
        except np.linalg.LinAlgError:
            # Singular matrix - option model is degenerate
            # Use a very small identity to regularize
            fund_matrix = np.linalg.inv(I_matrix - self.gamma * P_II + 1e-12 * I_matrix)
        
        # Store the model: for each state in I, compute r_o^s and p_ss'^o
        reward_dict: Dict[Tuple[int, int], np.ndarray] = {}
        termination_dict: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        
        for i, s in enumerate(I):
            # Store fund_matrix[i, :] for computing expected discounted reward: r_o^s = [fund_matrix * r_I]_s
            # Note: r_I is goal-dependent, so we'll compute fund_matrix * r_I in _compute_option_value
            # This captures: E[Σ_{t=0}^{τ-1} γ^t r(s_t)] where each γ^t is coupled with time t
            reward_dict[s] = fund_matrix[i, :]
            
            # Expected discounted termination distribution: p_ss'^o = E[γ^τ 1(s_τ = s')]
            # = Σ_{k=1}^∞ γ^k * P(exit at step k to s')
            # = Σ_{k=1}^∞ γ^k * [P_II^{k-1}]_{i,:} @ P_IB
            # = γ * [Σ_{k=0}^∞ γ^k P_II^k] @ P_IB
            # = γ * fund_matrix @ P_IB
            # This properly couples discount γ^k with option duration k
            termination_probs = self.gamma * fund_matrix[i, :] @ P_IB
            termination_dict[s] = {B[b]: termination_probs[b] for b in range(len(B))}
        
        return {
            'I': I,
            'B': B,
            'state_to_idx': state_to_idx,
            'N_row': reward_dict,  # N[i, :] for each state
            'termination': termination_dict,
            'r_I_indices': state_to_idx,  # For mapping states to r_I indices
            'is_primitive': False,
        }
    
    def _compute_option_value(
        self,
        state: Tuple[int, int],
        option_idx: int,
        goal_state: Tuple[int, int],
        V_prev: np.ndarray
    ) -> float:
        """Compute the value of executing an option from state using SMDP model.
        
        SMDP Bellman backup for options:
            V(s) = r_o^s + Σ_s' p_ss'^o * V(s')
        
        where:
        - r_o^s = E[Σ_{t=0}^{τ-1} γ^t r_t | s, o]  (expected discounted reward)
        - p_ss'^o = E[γ^τ 1(s_τ = s') | s, o]       (expected discounted termination)
        - τ is the random option duration
        
        The discount γ^τ is properly coupled with the stochastic termination time
        through the fundamental matrix (computed in _compute_option_models).
        """
        # Get the option and its model
        model = self.option_models[option_idx]
        
        if state not in model['N_row']:
            # State not in option's initiation set
            return -np.inf
        
        # Get the hallway option (must be a HallwayOption, not a primitive)
        hallway_opt = self.options[option_idx]
        assert isinstance(hallway_opt, HallwayOption), "Expected HallwayOption for multi-step backup"
        
        # Compute goal-dependent immediate reward vector r_I
        # r_I[i] = E[r(s_i, π(s_i), s') | s_i]
        I = model['I']
        n_I = len(I)
        r_I = np.zeros(n_I)
        
        # For each state in I, compute expected reward under option's policy
        for i, s in enumerate(I):
            try:
                action = hallway_opt.get_action(s)
            except KeyError:
                continue
            
            # Get stochastic transitions
            transitions = self._get_stochastic_transitions(s, action)
            
            for next_state, prob in transitions:
                # Reward is 1 if reaching goal, 0 otherwise
                reward = 1.0 if next_state == goal_state else 0.0
                r_I[i] += prob * reward
        
        # Expected discounted reward: r_o^s = [N * r_I]_s
        # This computes E[Σ_{t=0}^{τ-1} γ^t r(s_t)] where γ^t is coupled with time t
        N_row = model['N_row'][state]  # This is N[i, :]
        r_o_s = N_row @ r_I
        
        # Expected discounted value from termination: Σ_s' p_ss'^o * V(s')
        # where p_ss'^o = E[γ^τ 1(s_τ = s')] couples γ^τ with option duration τ
        # Note: r_I already includes rewards for reaching goal during option execution,
        # so we only add the continuation value V(s') here (no double counting)
        termination_value = 0.0
        for s_prime, prob in model['termination'][state].items():
            r_prime, c_prime = s_prime
            termination_value += prob * V_prev[r_prime, c_prime]
        
        # SMDP Bellman backup: total expected discounted value
        return r_o_s + termination_value
    
    def _get_stochastic_transitions(
        self,
        state: Tuple[int, int],
        action: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get stochastic transitions for a primitive action.
        
        With 2/3 probability: intended direction
        With 1/3 probability: one of the other 3 directions (1/9 each)
        """
        transitions = []
        
        # All possible actions
        all_actions = get_primitive_actions()
        
        for a in all_actions:
            if a == action:
                prob = 2.0 / 3.0
            else:
                prob = 1.0 / 9.0
            
            next_state = self._get_deterministic_next_state(state, a)
            transitions.append((next_state, prob))
        
        return transitions
    
    def _get_deterministic_next_state(
        self,
        state: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        """Get the next state from taking an action (deterministic version)."""
        r, c = state
        
        # Map action to movement
        if action == 0:  # RIGHT
            next_state = (r, c + 1)
        elif action == 1:  # LEFT
            next_state = (r, c - 1)
        elif action == 2:  # UP
            next_state = (r - 1, c)
        elif action == 3:  # DOWN
            next_state = (r + 1, c)
        else:
            next_state = state
        
        # If next state is a wall, stay in current state
        if not self._is_valid_state(next_state):
            return state
        
        return next_state