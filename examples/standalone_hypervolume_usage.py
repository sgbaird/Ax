#!/usr/bin/env python3
"""
Example usage of the standalone hypervolume calculation function.

This demonstrates how to use the hypervolume functionality that has been
extracted from Ax and made standalone for easy copy-paste usage.
"""

import numpy as np
from ax.utils.hypervolume import (
    compute_hypervolume,
    compute_hypervolume_trace,
    infer_reference_point,
)


def example_basic_usage():
    """Basic hypervolume computation examples."""
    print("=== Basic Hypervolume Computation ===")
    
    # Example 1: Two objectives, simple case
    print("\n1. Two objectives with known reference point:")
    points = np.array([
        [2.0, 3.0],  # Point 1: good in objective 1, excellent in objective 2
        [3.0, 2.0],  # Point 2: excellent in objective 1, good in objective 2
        [1.5, 1.5],  # Point 3: moderate in both (will be dominated)
    ])
    reference_point = np.array([0.0, 0.0])
    
    hv = compute_hypervolume(points, reference_point)
    print(f"   Points: {points.tolist()}")
    print(f"   Reference: {reference_point.tolist()}")
    print(f"   Hypervolume: {hv:.3f}")
    
    # Example 2: With infeasible points
    print("\n2. With feasible/infeasible points:")
    feasible_mask = np.array([True, True, False])  # Third point is infeasible
    hv_feasible = compute_hypervolume(points, reference_point, feasible_mask)
    print(f"   Feasible mask: {feasible_mask.tolist()}")
    print(f"   Hypervolume (feasible only): {hv_feasible:.3f}")


def example_inferred_reference_point():
    """Example of inferring reference point from data."""
    print("\n=== Inferring Reference Point ===")
    
    # Multi-objective optimization results
    optimization_history = np.array([
        [1.2, 0.8, 0.9],  # Early results
        [1.5, 0.6, 1.1],
        [1.8, 0.4, 1.3],
        [2.1, 0.3, 1.5],  # Later, better results
        [2.0, 0.2, 1.4],
    ])
    
    print(f"Optimization history shape: {optimization_history.shape}")
    print(f"Objectives: [obj1_maximize, obj2_minimize, obj3_maximize]")
    
    # For mixed maximize/minimize objectives, we need to flip signs
    # for minimize objectives to make everything maximization
    processed_points = optimization_history.copy()
    processed_points[:, 1] *= -1  # Flip obj2 since it's minimize
    
    # Infer reference point
    ref_point = infer_reference_point(processed_points)
    print(f"Inferred reference point: {ref_point}")
    
    # Compute hypervolume
    hv = compute_hypervolume(processed_points, ref_point)
    print(f"Total hypervolume: {hv:.3f}")


def example_hypervolume_trace():
    """Example of tracking hypervolume over optimization progress."""
    print("\n=== Hypervolume Trace (Progress Tracking) ===")
    
    # Simulate optimization progress
    optimization_sequence = np.array([
        [1.0, 1.0],    # Initial point
        [1.5, 2.0],    # Improvement in obj2
        [1.2, 1.8],    # Dominated point (won't improve HV much)
        [2.5, 1.5],    # Improvement in obj1
        [2.0, 2.5],    # Improvement in obj2
        [3.0, 2.0],    # Final best point
    ])
    reference_point = np.array([0.0, 0.0])
    
    print("Optimization sequence:")
    for i, point in enumerate(optimization_sequence):
        print(f"   Step {i+1}: {point}")
    
    # Cumulative hypervolume (recommended for progress tracking)
    cumulative_hv = compute_hypervolume_trace(
        optimization_sequence, reference_point, cumulative=True
    )
    
    print("\nCumulative hypervolume progress:")
    for i, hv in enumerate(cumulative_hv):
        print(f"   After step {i+1}: {hv:.3f}")
    
    # Individual contributions
    individual_hv = compute_hypervolume_trace(
        optimization_sequence, reference_point, cumulative=False
    )
    
    print("\nIndividual point contributions:")
    for i, hv in enumerate(individual_hv):
        print(f"   Point {i+1}: {hv:.3f}")


def example_real_world_scenario():
    """Real-world scenario: multi-objective hyperparameter optimization."""
    print("\n=== Real-World Example: Hyperparameter Optimization ===")
    
    # Simulated results from hyperparameter optimization
    # Objectives: [accuracy, -training_time, -model_size]
    # (negative because we want to minimize time and size)
    results = np.array([
        [0.85, -120, -50],   # Fast, small model, decent accuracy
        [0.92, -300, -200],  # Slow, large model, high accuracy  
        [0.88, -180, -80],   # Balanced approach
        [0.94, -450, -350],  # Very slow, very large, highest accuracy
        [0.90, -200, -100],  # Another balanced point
        [0.87, -150, -70],   # Slightly dominated
    ])
    
    # Some experiments might have failed
    feasible = np.array([True, True, True, True, False, True])
    
    print("Hyperparameter optimization results:")
    objectives = ["Accuracy", "Training Time (neg)", "Model Size (neg)"]
    print(f"Objectives: {objectives}")
    for i, (result, feas) in enumerate(zip(results, feasible)):
        status = "✓" if feas else "✗"
        print(f"   Config {i+1}: {result} [{status}]")
    
    # Infer reference point from feasible results only
    ref_point = infer_reference_point(results, feasible)
    print(f"\nInferred reference point: {ref_point}")
    
    # Compute final hypervolume
    final_hv = compute_hypervolume(results, ref_point, feasible)
    print(f"Final hypervolume: {final_hv:.3f}")
    
    # Track progress over time
    hv_progress = compute_hypervolume_trace(results, ref_point, feasible, cumulative=True)
    print("\nHypervolume progress:")
    for i, hv in enumerate(hv_progress):
        status = "✓" if feasible[i] else "✗ (failed)"
        print(f"   After config {i+1}: {hv:.3f} [{status}]")


def example_higher_dimensions():
    """Example with higher-dimensional objectives."""
    print("\n=== Higher Dimensional Example (5 objectives) ===")
    
    # 5-objective optimization problem
    points_5d = np.array([
        [1.0, 2.0, 1.5, 1.8, 1.2],
        [1.5, 1.8, 1.8, 1.5, 1.5], 
        [1.2, 1.9, 1.6, 1.7, 1.3],
        [1.8, 1.5, 2.0, 1.2, 1.7],
    ])
    ref_5d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"5D points shape: {points_5d.shape}")
    hv_5d = compute_hypervolume(points_5d, ref_5d)
    print(f"5D hypervolume: {hv_5d:.3f}")
    
    # Note: For very high dimensions (>6), specialized algorithms
    # like WFG would be more efficient, but this works for moderate cases


if __name__ == "__main__":
    example_basic_usage()
    example_inferred_reference_point()
    example_hypervolume_trace()
    example_real_world_scenario()
    example_higher_dimensions()
    
    print("\n" + "="*60)
    print("COPY-PASTE INSTRUCTIONS:")
    print("="*60)
    print("To use this hypervolume function independently:")
    print("1. Copy the file ax/utils/hypervolume.py")
    print("2. Only requires numpy (no heavy dependencies)")
    print("3. Use compute_hypervolume() for single calculations")
    print("4. Use compute_hypervolume_trace() for progress tracking")
    print("5. Use infer_reference_point() when you don't know the reference")
    print("="*60)