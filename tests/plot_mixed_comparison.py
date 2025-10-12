"""Compare planning with primitives only vs. primitives + options together.

This script demonstrates the comparison between:
- Planner 1 (O = A): Uses only primitive actions (up, down, left, right)
- Planner 2 (O = H): Uses 8 hallway options only
- Planner 3 (O = A ∪ H): Uses both primitives AND hallway options together

All planners use Synchronous Value Iteration (SVI) on the same MDP.
We test on a hallway goal (G1).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import FourRooms
from src.planning.hallway_options import get_hallway_options, get_primitive_actions
from src.planning.svi_planner import SVIPlanner


def run_comparison():
    """Run the comparison experiment."""
    env = FourRooms()
    
    # Define goal state
    # G1: Hallway cell (one of the 4 hallway positions)
    goal_g1 = (3, 6)  # Right hallway of room 1
    
    print("=" * 80)
    print("PLANNING WITH MIXED OPTIONS: Synchronous Value Iteration Experiment")
    print("=" * 80)
    print()
    print("Environment: Four Rooms (13x13 grid with stochastic transitions)")
    print("Algorithm: Synchronous Value Iteration (SVI)")
    print()
    print("Comparison:")
    print("  - Planner 1 (O = A):       Uses 4 primitive actions")
    print("  - Planner 2 (O = H):       Uses 8 hallway options")
    print("  - Planner 3 (O = A ∪ H):   Uses both primitives AND options (12 total)")
    print()
    
    # Get options
    primitive_actions = get_primitive_actions()
    hallway_options = get_hallway_options()
    mixed_options = primitive_actions + hallway_options  # Combine both
    
    print(f"Primitive Actions:")
    action_names = ["→ (Right)", "← (Left)", "↑ (Up)", "↓ (Down)"]
    for i, _ in enumerate(primitive_actions):
        print(f"  {i}: {action_names[i]}")
    print(f"Hallway Options:")
    for i, opt in enumerate(hallway_options):
        print(f"  {i}: {opt.name} → {opt.target_hallway}")
    print(f"Mixed: Primitives + Options")
    
    # Results storage
    results = {}
    
    # Test G1 (Hallway) goal only
    for goal_name, goal_state in [("G1 (Hallway)", goal_g1)]:
        print(f"\n{'=' * 80}")
        print(f"Goal: {goal_name} at position {goal_state}")
        print(f"{'=' * 80}")

        planner_primitive = SVIPlanner(
            env=env,
            options=primitive_actions,
            gamma=0.99,
            theta=1e-6
        )
        iterations_primitive = planner_primitive.plan(goal_state)
        steps_primitive = planner_primitive.planning_steps
  
        planner_options = SVIPlanner(
            env=env,
            options=hallway_options,
            gamma=0.99,
            theta=1e-6
        )
        iterations_options = planner_options.plan(goal_state)
        steps_options = planner_options.planning_steps
 
        planner_mixed = SVIPlanner(
            env=env,
            options=mixed_options,  # type: ignore[arg-type]
            gamma=0.99,
            theta=1e-6
        )
        iterations_mixed = planner_mixed.plan(goal_state)
        steps_mixed = planner_mixed.planning_steps
        # Comparison
        speedup_options = iterations_primitive / iterations_options if iterations_options > 0 else 1.0
        speedup_mixed = iterations_primitive / iterations_mixed if iterations_mixed > 0 else 1.0
        
        steps_speedup_options = steps_primitive / steps_options if steps_options > 0 else 1.0
        steps_speedup_mixed = steps_primitive / steps_mixed if steps_mixed > 0 else 1.0
        
        print(f"\n{'─' * 80}")
        print(f"RESULTS FOR {goal_name}:")
        print(f"{'─' * 80}")
        print(f"  Primitive Actions (O = A):  {iterations_primitive:4d} iterations, {steps_primitive:6d} steps, max_value: {np.max(planner_primitive.values):.6f}")
        print(f"  Hallway Options (O = H):    {iterations_options:4d} iterations, {steps_options:6d} steps, max_value: {np.max(planner_options.values):.6f}")
        print(f"  Mixed (O = A ∪ H):          {iterations_mixed:4d} iterations, {steps_mixed:6d} steps, max_value: {np.max(planner_mixed.values):.6f}")
        print(f"\n  Speedup (iterations):")
        print(f"    Options vs Primitive:  {speedup_options:.2f}x faster")
        print(f"    Mixed vs Primitive:    {speedup_mixed:.2f}x faster")
        print(f"\n  Speedup (planning steps):")
        print(f"    Options vs Primitive:  {steps_speedup_options:.2f}x faster")
        print(f"    Mixed vs Primitive:    {steps_speedup_mixed:.2f}x faster")        
        # Store results
        results[goal_name] = {
            'primitive_iterations': iterations_primitive,
            'options_iterations': iterations_options,
            'mixed_iterations': iterations_mixed,
            'primitive_steps': steps_primitive,
            'options_steps': steps_options,
            'mixed_steps': steps_mixed,
            'primitive_max_value': np.max(planner_primitive.values),
            'options_max_value': np.max(planner_options.values),
            'mixed_max_value': np.max(planner_mixed.values),
            'speedup_options': speedup_options,
            'speedup_mixed': speedup_mixed,
            'steps_speedup_options': steps_speedup_options,
            'steps_speedup_mixed': steps_speedup_mixed,
            'primitive_planner': planner_primitive,
            'options_planner': planner_options,
            'mixed_planner': planner_mixed,
        }
    
    return results


def visualize_results(results: Dict[str, Any]) -> None:
    """Visualize the comparison results."""
    _, (ax_iter, ax_steps) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get results for G1 (Hallway)
    goal_name = "G1 (Hallway)"
    res = results[goal_name]
    
    # Left plot: Iterations comparison
    planners = ['Primitive\n(O=A)', 'Options\n(O=H)', 'Mixed\n(O=A∪H)']
    iterations = [
        res['primitive_iterations'],
        res['options_iterations'],
        res['mixed_iterations']
    ]
    colors_bar = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    
    bars = ax_iter.bar(planners, iterations, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_iter.set_ylabel('Iterations to Convergence', fontsize=11, fontweight='bold')
    ax_iter.set_title(f'{goal_name}: Convergence (Iterations)', fontsize=12, fontweight='bold')
    ax_iter.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax_iter.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add speedup annotations
    ax_iter.text(0.5, 0.95, f'Options: {res["speedup_options"]:.2f}x, Mixed: {res["speedup_mixed"]:.2f}x',
               transform=ax_iter.transAxes,
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
               fontsize=9, fontweight='bold')
    
    # Right plot: Planning steps comparison
    steps = [
        res['primitive_steps'],
        res['options_steps'],
        res['mixed_steps']
    ]
    
    bars_steps = ax_steps.bar(planners, steps, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_steps.set_ylabel('Planning Steps', fontsize=11, fontweight='bold')
    ax_steps.set_title(f'{goal_name}: Planning Steps', fontsize=12, fontweight='bold')
    ax_steps.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars_steps, steps):
        height = bar.get_height()
        ax_steps.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add speedup annotations
    ax_steps.text(0.5, 0.95, f'Options: {res["steps_speedup_options"]:.2f}x, Mixed: {res["steps_speedup_mixed"]:.2f}x',
               transform=ax_steps.transAxes,
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/katherineharvey/options/plots/mixed_planning_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved to: plots/mixed_planning_comparison.png\n")
    plt.show()


if __name__ == "__main__":
    results = run_comparison()
    
    print("\n" + "=" * 80)
    visualize_results(results)

