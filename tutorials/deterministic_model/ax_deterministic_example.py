#!/usr/bin/env python3
"""
Ax-based Implementation: Using DeterministicModel with Mixed Optimization

This script demonstrates how to use DeterministicModel concepts in Ax for optimization 
problems where some objectives or constraints are analytically known.

Addresses GitHub issues #935 and #1192 in facebook/Ax repository using Ax's high-level API.

Author: Generated for Ax tutorial
"""

import torch
import numpy as np
from typing import Callable, Dict, Any, List, Tuple

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def analytical_cost_function(x: float, y: float) -> float:
    """Analytical cost function: f(x,y) = x² + y²"""
    return float(x**2 + y**2)


def black_box_performance(x: float, y: float) -> float:
    """Black-box performance function that we want to maximize."""
    base_value = -((x - 0.3)**2 + (y - 0.3)**2) + 0.5
    noise = 0.1 * np.random.randn()
    return float(base_value + noise)


def constraint_function(x: float, y: float) -> float:
    """Black-box constraint function: g(x,y) = x + y - 0.8 ≤ 0"""
    return float(x + y - 0.8 + 0.05 * np.random.randn())


def demonstrate_ax_style_approach():
    """Demonstrate Ax-style approach for mixed optimization using simple evaluate function."""
    
    print("=" * 60)
    print("Ax-Style DeterministicModel Example (Simple Approach)")
    print("=" * 60)
    
    print("\nThis example shows the conceptual approach for using Ax with mixed")
    print("analytical and black-box functions. In a full implementation, you would:")
    print("1. Create custom Metric classes for analytical functions")
    print("2. Use Ax's optimization loops with mixed metric types")
    print("3. Let Ax handle the optimization automatically")
    
    # Step 1: Define evaluation function that Ax would use
    def mixed_evaluation_function(parameterization: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Evaluation function for Ax optimization."""
        x = parameterization["x"]
        y = parameterization["y"]
        
        return {
            "cost": (analytical_cost_function(x, y), 0.0),  # Exact, no uncertainty
            "performance": (black_box_performance(x, y), 0.05),  # With uncertainty
            "constraint": (constraint_function(x, y), 0.02),  # With uncertainty
        }
    
    print("\n1. Evaluation Function Structure:")
    print("   def mixed_evaluation_function(parameterization):")
    print("       x, y = parameterization['x'], parameterization['y']")
    print("       return {")
    print("           'cost': (analytical_cost_function(x, y), 0.0),  # Exact")
    print("           'performance': (black_box_performance(x, y), 0.05),  # Uncertain")
    print("           'constraint': (constraint_function(x, y), 0.02),  # Uncertain")
    print("       }")
    
    # Step 2: Simulate optimization trials
    print("\n2. Simulated Optimization Results:")
    print(f"   {'Trial':<6} {'x':<8} {'y':<8} {'Cost':<10} {'Performance':<12} {'Constraint':<12}")
    print("   " + "-" * 62)
    
    # Generate some trial points
    trial_points = [
        {"x": 0.2, "y": 0.3},
        {"x": 0.4, "y": 0.2},
        {"x": 0.3, "y": 0.3},
        {"x": 0.1, "y": 0.4},
        {"x": 0.35, "y": 0.25},
    ]
    
    results = []
    for i, params in enumerate(trial_points):
        metrics = mixed_evaluation_function(params)
        results.append((params, metrics))
        
        cost_val = metrics["cost"][0]
        perf_val = metrics["performance"][0]
        const_val = metrics["constraint"][0]
        
        print(f"   {i+1:<6} {params['x']:<8.2f} {params['y']:<8.2f} {cost_val:<10.6f} {perf_val:<12.4f} {const_val:<12.4f}")
    
    # Step 3: Analysis
    print("\n3. Analysis:")
    
    # Find best feasible point (constraint <= 0)
    feasible_results = [(params, metrics) for params, metrics in results if metrics["constraint"][0] <= 0]
    
    if feasible_results:
        best_params, best_metrics = max(feasible_results, key=lambda x: x[1]["performance"][0])
        print(f"   Best feasible point: x={best_params['x']:.3f}, y={best_params['y']:.3f}")
        print(f"   Cost (analytical): {best_metrics['cost'][0]:.6f}")
        print(f"   Performance: {best_metrics['performance'][0]:.4f}")
        print(f"   Constraint: {best_metrics['constraint'][0]:.4f}")
    else:
        print("   No feasible points found in this simulation")
    
    # Step 4: Key benefits
    print("\n4. Key Benefits of This Approach:")
    print("   ✓ Analytical cost function computed exactly (zero error)")
    print("   ✓ No surrogate model needed for known functions")
    print("   ✓ Uncertainty correctly specified for each metric type")
    print("   ✓ Ready for integration with Ax's optimization algorithms")
    
    return results


def demonstrate_advanced_ax_concepts():
    """Demonstrate advanced Ax concepts for DeterministicModel integration."""
    
    print("\n" + "=" * 60)
    print("Advanced Ax Integration Concepts")
    print("=" * 60)
    
    print("\n1. **Custom Metric Classes**:")
    print("   ```python")
    print("   from ax.core.metric import Metric")
    print("   from ax.core.data import Data")
    print("   ")
    print("   class AnalyticalCostMetric(Metric):")
    print("       def fetch_trial_data(self, trial, **kwargs):")
    print("           # Compute analytical function directly")
    print("           arm = trial.arm")
    print("           x, y = arm.parameters['x'], arm.parameters['y']")
    print("           cost = x**2 + y**2  # Analytical function")
    print("           return Data([{")
    print("               'arm_name': arm.name,")
    print("               'metric_name': self.name,")
    print("               'mean': cost,")
    print("               'sem': 0.0,  # Zero uncertainty for deterministic")
    print("           }])")
    print("   ```")
    
    print("\n2. **Experiment Setup**:")
    print("   ```python")
    print("   from ax.core.experiment import Experiment")
    print("   from ax.core.optimization_config import OptimizationConfig")
    print("   from ax.core.objective import Objective")
    print("   ")
    print("   experiment = Experiment(")
    print("       name='mixed_optimization',")
    print("       search_space=search_space,")
    print("       optimization_config=OptimizationConfig(")
    print("           objective=Objective(performance_metric, minimize=False),")
    print("           outcome_constraints=[constraint_metric <= 0],")
    print("       ),")
    print("       tracking_metrics=[cost_metric],  # Track but don't optimize")
    print("   )")
    print("   ```")
    
    print("\n3. **Generation Strategy**:")
    print("   ```python")
    print("   from ax.modelbridge.dispatch_utils import choose_generation_strategy")
    print("   ")
    print("   generation_strategy = choose_generation_strategy(")
    print("       search_space=experiment.search_space,")
    print("       optimization_config=experiment.optimization_config,")
    print("   )")
    print("   ```")
    
    print("\n4. **Optimization Loop**:")
    print("   ```python")
    print("   for i in range(num_trials):")
    print("       trial = experiment.new_trial(")
    print("           generator_run=generation_strategy.gen(experiment)")
    print("       )")
    print("       trial.run().mark_completed()")
    print("       data = experiment.fetch_data()")
    print("       generation_strategy.gen(experiment, new_data=data)")
    print("   ```")
    
    print("\n5. **Alternative: Simple Optimization Function**:")
    print("   ```python")
    print("   from ax.service.managed_loop import optimize")
    print("   ")
    print("   best_parameters, values, experiment, model = optimize(")
    print("       parameters=[")
    print("           {'name': 'x', 'type': 'range', 'bounds': [0.0, 1.0]},")
    print("           {'name': 'y', 'type': 'range', 'bounds': [0.0, 1.0]},")
    print("       ],")
    print("       evaluation_function=mixed_evaluation_function,")
    print("       minimize=False,  # Maximize performance")
    print("       total_trials=20,")
    print("   )")
    print("   ```")


def compare_approaches():
    """Compare different approaches for handling analytical functions."""
    
    print("\n" + "=" * 60)
    print("Approach Comparison")
    print("=" * 60)
    
    approaches = [
        {
            "name": "Ax with Custom Metrics",
            "complexity": "Low",
            "deterministic_support": "Indirect",
            "integration": "Native",
            "use_case": "Production systems",
            "pros": ["Easy implementation", "Full Ax features", "Automatic optimization"],
            "cons": ["No direct DeterministicModel", "Less explicit"],
        },
        {
            "name": "Ax with Custom Surrogate",
            "complexity": "High", 
            "deterministic_support": "Direct",
            "integration": "Manual",
            "use_case": "Advanced research",
            "pros": ["Direct DeterministicModel", "Full control", "Explicit modeling"],
            "cons": ["Complex implementation", "Requires Ax internals knowledge"],
        },
        {
            "name": "Pure BoTorch",
            "complexity": "Medium",
            "deterministic_support": "Direct",
            "integration": "Manual",
            "use_case": "Research & prototyping",
            "pros": ["Direct model access", "Clear concepts", "Flexible"],
            "cons": ["Manual optimization loop", "No high-level features"],
        },
    ]
    
    print(f"\n{'Approach':<25} {'Complexity':<12} {'Det.Model':<12} {'Integration':<12} {'Use Case':<20}")
    print("-" * 85)
    
    for approach in approaches:
        print(f"{approach['name']:<25} {approach['complexity']:<12} {approach['deterministic_support']:<12} {approach['integration']:<12} {approach['use_case']:<20}")
    
    print("\n**Recommendations:**")
    print("1. **Start with Ax Custom Metrics** - easiest and most practical")
    print("2. **Use BoTorch directly** - for understanding concepts")
    print("3. **Consider Custom Surrogate** - only for advanced needs")


def main():
    """Main demonstration function."""
    
    print("Welcome to the Ax-based DeterministicModel Tutorial!")
    print("This tutorial demonstrates how to handle mixed analytical/black-box optimization")
    print("using Ax-compatible patterns.")
    
    # Demonstrate the simple approach
    results = demonstrate_ax_style_approach()
    
    # Show advanced concepts
    demonstrate_advanced_ax_concepts()
    
    # Compare approaches
    compare_approaches()
    
    print("\n" + "=" * 60)
    print("Summary and Next Steps")
    print("=" * 60)
    
    print("\n**What We Demonstrated:**")
    print("✓ Evaluation function structure for mixed optimization")
    print("✓ Handling analytical functions with zero uncertainty")
    print("✓ Conceptual Ax integration patterns")
    print("✓ Comparison of different implementation approaches")
    
    print("\n**Key Insights:**")
    print("1. **Analytical functions** should return (value, 0.0) to indicate no uncertainty")
    print("2. **Black-box functions** should return (value, sem) with appropriate uncertainty")
    print("3. **Ax handles mixed types** automatically when structured properly")
    print("4. **Custom Metric classes** provide the most flexible integration")
    
    print("\n**Next Steps:**")
    print("1. Implement custom Metric classes for your analytical functions")
    print("2. Create an Ax experiment with your specific search space")
    print("3. Use Ax's generation strategies for efficient optimization")
    print("4. Refer to the BoTorch example (minimal_example.py) for core concepts")
    
    print("\n**Files to Study:**")
    print("- `ax_deterministic_example.py`: This file (Ax concepts and patterns)")
    print("- `minimal_example.py`: BoTorch implementation (core DeterministicModel usage)")
    print("- `README.md`: Complete documentation and usage guidance")
    
    print("\n" + "=" * 60)
    print("Tutorial completed successfully!")
    print("For a working BoTorch example, run: python minimal_example.py")
    print("=" * 60)


if __name__ == "__main__":
    main()