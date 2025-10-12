"""Test script to validate SMDP backup implementation.

This script performs several checks:
1. Mass check termination probabilities - should sum to E[γ^τ] ≤ 1, not near 0
2. Verify no empty P_II rows - every state should have policy action or self-loop
3. Plot convergence with global color scale to visualize options vs primitives
"""


import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent

sys.path.insert(0, str(project_root))


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.env.env import FourRooms
from src.planning.svi_planner import SVIPlanner
from src.planning.hallway_options import get_hallway_options, get_primitive_actions



def test_termination_probabilities():
    """Test that termination probabilities sum to reasonable values (E[γ^τ])."""
    print("=" * 80)
    print("TEST 1: Termination Probability Mass Check")
    print("=" * 80)
    
    env = FourRooms()
    options = get_hallway_options()
    gamma = 0.99
    
    planner = SVIPlanner(env, options, gamma=gamma, theta=1e-6)
    
    # Check each option's termination probabilities
    for opt_idx, option in enumerate(options):
        print(f"\nOption {opt_idx}: {option.name}")
        print(f"  Target hallway: {option.target_hallway}")
        print(f"  Room has {len(option.room_cells)} cells")
        
        model = planner.option_models[opt_idx]
        
        # Sample a handful of states from the initiation set
        sample_states = list(option.room_cells)[:10]  # First 10 states
        
        termination_sums = []
        for state in sample_states:
            if state in model['termination']:
                term_probs = model['termination'][state]
                prob_sum = sum(term_probs.values())
                termination_sums.append(prob_sum)
                
                # Print details for first few states
                if len(termination_sums) <= 3:
                    print(f"    State {state}: Σ p_ss'^o = {prob_sum:.6f}")
                    print(f"      Terminates to {len(term_probs)} states:")
                    for s_prime, prob in sorted(term_probs.items(), key=lambda x: -x[1])[:5]:  # type: ignore[arg-type,return-value]
                        if prob > 0.001:
                            print(f"        {s_prime}: {prob:.6f}")
        
        if termination_sums:
            avg_sum = np.mean(termination_sums)
            min_sum = np.min(termination_sums)
            max_sum = np.max(termination_sums)
            
            print(f"  Summary of {len(termination_sums)} states:")
            print(f"    Average Σ p_ss'^o = {avg_sum:.6f}")
            print(f"    Min = {min_sum:.6f}, Max = {max_sum:.6f}")
            
            # Check if reasonable (should be close to E[γ^τ], which is ≤ 1)
            if avg_sum < 0.1:
                print(f"    ⚠️  WARNING: Average sum is very low! Expected close to E[γ^τ] ≤ 1")
            elif avg_sum > 1.01:
                print(f"    ⚠️  WARNING: Average sum > 1! Something is wrong.")
            else:
                print(f"    ✓ PASS: Termination probabilities sum to reasonable value")


def test_no_empty_rows():
    """Test that no P_II rows are empty (all states have policy or self-loop)."""
    print("\n" + "=" * 80)
    print("TEST 2: Check for Empty P_II Rows")
    print("=" * 80)
    
    env = FourRooms()
    options = get_hallway_options()
    gamma = 0.99
    
    # We'll rebuild the models with extra checking
    planner = SVIPlanner(env, options, gamma=gamma, theta=1e-6)
    
    for opt_idx, option in enumerate(options):
        print(f"\nOption {opt_idx}: {option.name}")
        
        model = planner.option_models[opt_idx]
        I = model['I']
        
        # Check each state's policy
        states_without_action = []
        for state in I:
            try:
                _ = option.get_action(state)
                # Has action - good
            except KeyError:
                states_without_action.append(state)
        
        if states_without_action:
            print(f"  ⚠️  Found {len(states_without_action)} states without policy actions:")
            for s in states_without_action[:5]:
                print(f"    {s}")
            if len(states_without_action) > 5:
                print(f"    ... and {len(states_without_action) - 5} more")
        else:
            print(f"  ✓ PASS: All {len(I)} states have policy actions")


def collect_values_by_fixed_sweeps(
    planner: SVIPlanner,
    sweep_counts: List[int],
    goal_state: Tuple[int, int]
) -> List[np.ndarray]:
    """Collect value snapshots at specific sweep counts (paper-like).
    
    This matches Sutton et al.'s methodology: run exactly T sweeps and snapshot,
    without early stopping on convergence threshold.
    
    Args:
        planner: The SVI planner
        sweep_counts: List of sweep counts (e.g., [1, 2, 3, 5, 10, 15, 20, 30])
        goal_state: The goal state tuple
        
    Returns:
        List of value function snapshots (one per sweep count)
    """
    out = []
    
    for num_sweeps in sweep_counts:
        # Reset and run for exactly num_sweeps iterations
        planner.values = np.zeros((13, 13))
        planner.policy = np.zeros((13, 13), dtype=int)
        planner.plan(goal_state, max_iterations=num_sweeps, fixed_sweeps=True)
        
        v = planner.values.copy()
        # mask walls and mark untouched states slightly below 0
        for r in range(13):
            for c in range(13):
                if not planner._is_valid_state((r, c)):  # type: ignore[attr-defined]
                    v[r, c] = np.nan
                elif (r, c) != goal_state and v[r, c] == 0:
                    v[r, c] = -0.01
        out.append(v)
    
    return out


def plot_convergence_comparison():
    """Plot convergence with global color scale to visualize options vs primitives."""
    print("\n" + "=" * 80)
    print("TEST 3: Convergence Visualization")
    print("=" * 80)
    
    env = FourRooms()
    # Use a hallway as the goal to demonstrate cross-room value propagation
    goal_state = (3, 6)  # Hallway between Room 1 and Room 2
    print(f"Goal state: {goal_state} (hallway between Room 1 and Room 2)")
    
    # Plan with primitives
    print("\nPlanning with primitive actions...")
    primitives = get_primitive_actions()
    planner_primitive = SVIPlanner(env, primitives, gamma=0.99, theta=1e-6)
    iters_primitive = planner_primitive.plan(goal_state, max_iterations=100)
    print(f"  Converged in {iters_primitive} iterations ({planner_primitive.planning_steps} planning steps)")
    
    # Plan with options
    print("Planning with hallway options...")
    options = get_hallway_options()
    planner_options = SVIPlanner(env, options, gamma=0.99, theta=1e-6)
    iters_options = planner_options.plan(goal_state, max_iterations=100)
    print(f"  Converged in {iters_options} iterations ({planner_options.planning_steps} planning steps)")
    
    # Plan with both
    print("Planning with both primitives and options...")
    both = get_primitive_actions() + get_hallway_options()
    planner_both = SVIPlanner(env, both, gamma=0.99, theta=1e-6)  # type: ignore[arg-type]
    iters_both = planner_both.plan(goal_state, max_iterations=100)
    print(f"  Converged in {iters_both} iterations ({planner_both.planning_steps} planning steps)")
    
    # --- NEW: iteration-by-iteration visualization with 3 groups (Primitives / Options / Both) ---
    print("\nCreating iteration-by-iteration visualization with global color scale (3 groups)...")
    
    # Rebuild planners for clean snapshots
    planner_prim_snapshot = SVIPlanner(env, primitives, gamma=0.99, theta=1e-6)
    planner_opt_snapshot = SVIPlanner(env, options, gamma=0.99, theta=1e-6)
    planner_both_snapshot = SVIPlanner(env, both, gamma=0.99, theta=1e-6)  # type: ignore[arg-type]

    # Use fixed sweeps matching the paper (Sutton et al. 1999)
    snapshot_sweeps = [1, 2, 3, 5, 10, 15, 20, 30]
    
    print(f"  Collecting snapshots at sweeps: {snapshot_sweeps}")
    all_v_prim = collect_values_by_fixed_sweeps(planner_prim_snapshot, snapshot_sweeps, goal_state)
    all_v_opt  = collect_values_by_fixed_sweeps(planner_opt_snapshot, snapshot_sweeps, goal_state)
    all_v_both = collect_values_by_fixed_sweeps(planner_both_snapshot, snapshot_sweeps, goal_state)

    # Global color limits across all groups
    vmin = -0.01
    vmax = max(
        max(np.nanmax(v) for v in all_v_prim),
        max(np.nanmax(v) for v in all_v_opt),
        max(np.nanmax(v) for v in all_v_both),
    )

    import copy
    cmap = copy.copy(plt.colormaps['viridis'])
    cmap.set_bad(color='white')  # walls

    # Grid: 3 rows (one per group) × 8 columns (one per snapshot)
    n_rows = 3
    n_cols = len(snapshot_sweeps)  # 8 snapshots
    groups = [
        ("Primitives", all_v_prim),
        ("Options",    all_v_opt),
        ("Both",       all_v_both),
    ]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols*2.0, n_rows*2.5),
        constrained_layout=True
    )
    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes[:, None]  # type: ignore[index]

    im = None  # Initialize for type checker
    for row_idx, (gname, gvals) in enumerate(groups):
        for col_idx, num_sweeps in enumerate(snapshot_sweeps):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(gvals[col_idx], cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect='equal', interpolation='nearest')
            # Add star at goal
            goal_r, goal_c = goal_state
            ax.scatter([goal_c], [goal_r], marker='*', s=150, 
                      color='red', edgecolors='white', linewidths=1.5, zorder=10)
            
            # Only show title on top row
            if row_idx == 0:
                ax.set_title(f"Iteration {num_sweeps}", fontsize=10, weight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)

        # Group label on the far-left of each row
        axes[row_idx, 0].set_ylabel(gname, rotation=90, fontsize=12, weight='bold', labelpad=10)

    # One compact colorbar for all tiles
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label('Value')

    plt.suptitle('Value Function Evolution with Goal at Hallway (3,6)',
                 fontsize=14, weight='bold')
    plt.savefig('plots/value_evolution_hallway.png', dpi=150)
    print("✓ Evolution plot (3 groups, fixed sweeps) saved to: plots/value_evolution_hallway.png")
    
if __name__ == "__main__":
    print("SMDP Backup Validation Test Suite")
    print("=" * 80)
    
    # Run all tests
    test_termination_probabilities()
    test_no_empty_rows()
    plot_convergence_comparison()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

