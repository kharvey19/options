"""Plot efficiency metrics for options vs primitives planning.

This script creates three plots demonstrating the benefits of options:
1. Value function error vs. iteration – shows faster propagation of value with options
2. Cumulative primitive steps vs. value error – demonstrates efficiency per primitive update
3. Value of start state vs. planning time – shows how quickly the agent learns useful behavior
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from src.env import FourRooms
from src.planning.svi_planner import SVIPlanner
from src.planning.hallway_options import get_hallway_options, get_primitive_actions


def compute_value_error(V_current: np.ndarray, V_true: np.ndarray, valid_states: List[Tuple[int, int]]) -> float:
    """Compute the L2 error between current and true value functions.
    
    Args:
        V_current: Current value function array
        V_true: True/optimal value function array
        valid_states: List of valid (non-wall) states
        
    Returns:
        L2 error normalized by number of states
    """
    errors = []
    for state in valid_states:
        r, c = state
        error = abs(V_current[r, c] - V_true[r, c])
        errors.append(error ** 2)
    
    return np.sqrt(np.mean(errors))


def collect_convergence_with_snapshots(
    planner: SVIPlanner,
    goal_state: Tuple[int, int],
    true_values: np.ndarray,
    valid_states: List[Tuple[int, int]],
    start_state: Tuple[int, int],
    max_iterations: int = 100
) -> Dict[str, np.ndarray]:
    """Run planning and collect snapshots at each iteration.
    
    Args:
        planner: The SVI planner
        goal_state: The goal state tuple
        true_values: The true/optimal value function
        valid_states: List of valid states
        start_state: A specific start state to track
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary with iteration data
    """
    planner.values = np.zeros((13, 13))
    planner.policy = np.zeros((13, 13), dtype=int)
    planner.value_history = []
    planner.planning_steps = 0
    planner.planning_steps_history = []
    
    # Track metrics at each iteration
    iterations = []
    errors = []
    planning_steps = []
    start_values = []
    
    for i in range(max_iterations):
        # Compute current metrics before update
        current_error = compute_value_error(planner.values, true_values, valid_states)
        start_value = planner.values[start_state]
        
        iterations.append(i)
        errors.append(current_error)
        planning_steps.append(planner.planning_steps)
        start_values.append(start_value)
        
        # Perform one iteration
        delta = 0
        old_values = planner.values.copy()
        sweep_steps = 0
        
        # Update value for each state (one sweep)
        for r in range(13):
            for c in range(13):
                state = (r, c)
                
                # Skip walls
                if not planner._is_valid_state(state):  # type: ignore[attr-defined]
                    continue
                
                # Goal state has value 0
                if state == goal_state:
                    planner.values[r, c] = 0
                    continue
                
                # Compute value for each available option
                best_value = -np.inf
                best_option = -1
                
                for option_idx in range(len(planner.options)):
                    model = planner.option_models[option_idx]
                    
                    # Check if option can initiate from this state
                    if state not in model['I']:
                        continue
                    
                    # Count this backup
                    sweep_steps += 1
                    
                    # Compute Q(s, o)
                    if model['is_primitive']:
                        action = model['action']
                        value = planner._compute_primitive_value(state, action, goal_state, old_values)  # type: ignore[attr-defined]
                    else:
                        value = planner._compute_option_value(state, option_idx, goal_state, old_values)  # type: ignore[attr-defined]
                    
                    if value > best_value:
                        best_value = value
                        best_option = option_idx
                
                # Update value
                if np.isfinite(best_value) and best_option >= 0:
                    planner.values[r, c] = best_value
                    planner.policy[r, c] = best_option
                else:
                    planner.values[r, c] = old_values[r, c]
                
                # Track maximum change
                val_old = old_values[r, c]
                val_new = planner.values[r, c]
                if np.isfinite(val_old) and np.isfinite(val_new):
                    delta = max(delta, abs(val_new - val_old))
        
        planner.planning_steps += sweep_steps
        planner.value_history.append(delta)
        planner.planning_steps_history.append(planner.planning_steps)
        
        # Check convergence
        if delta < planner.theta:
            # Add final snapshot
            current_error = compute_value_error(planner.values, true_values, valid_states)
            start_value = planner.values[start_state]
            iterations.append(i + 1)
            errors.append(current_error)
            planning_steps.append(planner.planning_steps)
            start_values.append(start_value)
            break
    
    return {
        'iterations': np.array(iterations),
        'errors': np.array(errors),
        'planning_steps': np.array(planning_steps),
        'start_values': np.array(start_values),
    }


def plot_efficiency_metrics():
    """Generate the three efficiency metric plots."""
    print("=" * 80)
    print("Plotting Efficiency Metrics for Options vs Primitives")
    print("=" * 80)
    
    # Setup environment and goal
    env = FourRooms()
    goal_state = (3, 6)  # Hallway between Room 1 and Room 2
    start_state = (9, 9)  # Far corner of Room 4 - good for demonstrating value propagation
    print(f"Goal state: {goal_state}")
    print(f"Start state: {start_state}")
    
    # Get all valid states
    valid_states = []
    for r in range(13):
        for c in range(13):
            if r == 0 or r == 12 or c == 0 or c == 12:
                continue
            if c == 6 and r not in [3, 10]:
                continue
            if r == 6 and c < 6 and c != 2:
                continue
            if r == 7 and c >= 6 and c != 9:
                continue
            valid_states.append((r, c))
    
    print(f"Valid states: {len(valid_states)}")
    
    # Step 1: Compute true/optimal value functions by running to full convergence
    print("\n" + "=" * 80)
    print("Computing true value functions (running to full convergence)...")
    print("=" * 80)
    
    # True values with primitives
    print("\n1. Computing true values with primitives...")
    primitives = get_primitive_actions()
    planner_prim_true = SVIPlanner(env, primitives, gamma=0.99, theta=1e-10)
    iters_true_prim = planner_prim_true.plan(goal_state, max_iterations=500)
    V_true_prim = planner_prim_true.values.copy()
    print(f"   Converged in {iters_true_prim} iterations")
    print(f"   Value at start state: {V_true_prim[start_state]:.6f}")
    
    # True values with options
    print("\n2. Computing true values with options...")
    options = get_hallway_options()
    planner_opt_true = SVIPlanner(env, options, gamma=0.99, theta=1e-10)
    iters_true_opt = planner_opt_true.plan(goal_state, max_iterations=500)
    V_true_opt = planner_opt_true.values.copy()
    print(f"   Converged in {iters_true_opt} iterations")
    print(f"   Value at start state: {V_true_opt[start_state]:.6f}")
    
    # True values with both
    print("\n3. Computing true values with both...")
    both = get_primitive_actions() + get_hallway_options()
    planner_both_true = SVIPlanner(env, both, gamma=0.99, theta=1e-10)  # type: ignore[arg-type]
    iters_true_both = planner_both_true.plan(goal_state, max_iterations=500)
    V_true_both = planner_both_true.values.copy()
    print(f"   Converged in {iters_true_both} iterations")
    print(f"   Value at start state: {V_true_both[start_state]:.6f}")
    
    # Verify they converge to same values
    error_prim_opt = compute_value_error(V_true_prim, V_true_opt, valid_states)
    error_prim_both = compute_value_error(V_true_prim, V_true_both, valid_states)
    print(f"\nValue function agreement:")
    print(f"   Primitives vs Options: {error_prim_opt:.2e}")
    print(f"   Primitives vs Both: {error_prim_both:.2e}")
    
    # Use primitives as ground truth (all should converge to same values)
    V_true = V_true_prim
    
    # Step 2: Collect convergence data with snapshots
    print("\n" + "=" * 80)
    print("Collecting convergence data with snapshots...")
    print("=" * 80)
    
    print("\n1. Running primitives...")
    planner_prim = SVIPlanner(env, primitives, gamma=0.99, theta=1e-6)
    data_prim = collect_convergence_with_snapshots(
        planner_prim, goal_state, V_true, valid_states, start_state, max_iterations=100
    )
    print(f"   Converged in {len(data_prim['iterations'])-1} iterations")
    print(f"   Final error: {data_prim['errors'][-1]:.2e}")
    print(f"   Final start value: {data_prim['start_values'][-1]:.6f}")
    
    print("\n2. Running options...")
    planner_opt = SVIPlanner(env, options, gamma=0.99, theta=1e-6)
    data_opt = collect_convergence_with_snapshots(
        planner_opt, goal_state, V_true, valid_states, start_state, max_iterations=100
    )
    print(f"   Converged in {len(data_opt['iterations'])-1} iterations")
    print(f"   Final error: {data_opt['errors'][-1]:.2e}")
    print(f"   Final start value: {data_opt['start_values'][-1]:.6f}")
    
    print("\n3. Running both...")
    planner_both = SVIPlanner(env, both, gamma=0.99, theta=1e-6)  # type: ignore[arg-type]
    data_both = collect_convergence_with_snapshots(
        planner_both, goal_state, V_true, valid_states, start_state, max_iterations=100
    )
    print(f"   Converged in {len(data_both['iterations'])-1} iterations")
    print(f"   Final error: {data_both['errors'][-1]:.2e}")
    print(f"   Final start value: {data_both['start_values'][-1]:.6f}")
    
    # Step 3: Create the three plots
    print("\n" + "=" * 80)
    print("Creating plots...")
    print("=" * 80)
    
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Value function error vs. iteration
    ax1 = axes[0]
    ax1.semilogy(data_prim['iterations'], data_prim['errors'], 
                 'o-', label='Primitives', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    ax1.semilogy(data_opt['iterations'], data_opt['errors'], 
                 's-', label='Options', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    ax1.semilogy(data_both['iterations'], data_both['errors'], 
                 '^-', label='Both', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    
    ax1.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax1.set_ylabel('Value Function Error (L2)', fontsize=12, weight='bold')
    ax1.set_title('Value Error vs Iteration\n(Faster value propagation with options)', 
                  fontsize=12, weight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(left=-1)
    
    # Plot 2: Cumulative primitive steps vs. value error
    ax2 = axes[1]
    ax2.semilogy(data_prim['planning_steps'], data_prim['errors'], 
                 'o-', label='Primitives', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    ax2.semilogy(data_opt['planning_steps'], data_opt['errors'], 
                 's-', label='Options', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    ax2.semilogy(data_both['planning_steps'], data_both['errors'], 
                 '^-', label='Both', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    
    ax2.set_xlabel('Cumulative Planning Steps', fontsize=12, weight='bold')
    ax2.set_ylabel('Value Function Error (L2)', fontsize=12, weight='bold')
    ax2.set_title('Value Error vs Planning Work\n(Better efficiency per update)', 
                  fontsize=12, weight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(left=-100)
    
    # Plot 3: Value of start state vs. planning steps
    ax3 = axes[2]
    ax3.plot(data_prim['planning_steps'], data_prim['start_values'], 
             'o-', label='Primitives', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    ax3.plot(data_opt['planning_steps'], data_opt['start_values'], 
             's-', label='Options', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    ax3.plot(data_both['planning_steps'], data_both['start_values'], 
             '^-', label='Both', linewidth=2, markersize=5, alpha=0.8, markevery=2)
    
    # Add horizontal line for true value
    true_start_value = V_true[start_state]
    ax3.axhline(y=true_start_value, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'True value ({true_start_value:.3f})')
    
    ax3.set_xlabel('Cumulative Planning Steps', fontsize=12, weight='bold')
    ax3.set_ylabel(f'Value of Start State {start_state}', fontsize=12, weight='bold')
    ax3.set_title('Start State Value vs Planning Time\n(Faster learning of useful behavior)', 
                  fontsize=12, weight='bold')
    ax3.legend(fontsize=10, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=-100)
    
    plt.tight_layout()
    plt.savefig('plots/planning_efficiency_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved to: plots/planning_efficiency_metrics.png")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    prim_iters = len(data_prim['iterations']) - 1
    opt_iters = len(data_opt['iterations']) - 1
    both_iters = len(data_both['iterations']) - 1
    
    prim_steps = data_prim['planning_steps'][-1]
    opt_steps = data_opt['planning_steps'][-1]
    both_steps = data_both['planning_steps'][-1]
    
    print(f"\n{'Method':<15} {'Iterations':<12} {'Planning Steps':<15} {'Final Error':<15}")
    print("-" * 80)
    print(f"{'Primitives':<15} {prim_iters:<12} {prim_steps:<15} {data_prim['errors'][-1]:<15.2e}")
    print(f"{'Options':<15} {opt_iters:<12} {opt_steps:<15} {data_opt['errors'][-1]:<15.2e}")
    print(f"{'Both':<15} {both_iters:<12} {both_steps:<15} {data_both['errors'][-1]:<15.2e}")
    
    print(f"\nSpeedups:")
    print(f"  Iterations: Options are {prim_iters/opt_iters:.2f}x faster, Both are {prim_iters/both_iters:.2f}x faster")
    print(f"  Steps: Options are {prim_steps/opt_steps:.2f}x more efficient, Both are {prim_steps/both_steps:.2f}x more efficient")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    plot_efficiency_metrics()

