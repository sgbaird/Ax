#!/usr/bin/env python3
"""
Ax-based Implementation: Using DeterministicModel with Mixed Optimization

This script demonstrates how to use AxClient with mixed analytical/black-box optimization 
where some objectives or constraints are analytically known.

Addresses GitHub issues #935 and #1192 in facebook/Ax repository.

Author: Generated for Ax tutorial
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple

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


def evaluation_function(parameterization: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    Evaluation function for Ax optimization.
    Returns a dictionary with metrics and their (mean, sem) values.
    
    The key insight for DeterministicModel usage:
    - Analytical functions return (value, 0.0) indicating zero uncertainty
    - Black-box functions return (value, sem) with appropriate uncertainty
    """
    x = parameterization["x"]
    y = parameterization["y"]
    
    return {
        "cost": (analytical_cost_function(x, y), 0.0),  # Analytical - zero uncertainty
        "performance": (black_box_performance(x, y), 0.05),  # Black-box with uncertainty  
        "constraint": (constraint_function(x, y), 0.02),  # Black-box with uncertainty
    }


def run_optimization_simple() -> None:
    """Run mixed analytical/black-box optimization using simple Ax API."""
    
    print("Mixed Analytical/Black-box Optimization using AxClient")
    print("=" * 65)
    
    try:
        # Try to import AxClient
        from ax.service.ax_client import AxClient, ObjectiveProperties
        
        # Create AxClient for optimization
        ax_client = AxClient()
        
        ax_client.create_experiment(
            name="deterministic_model_optimization",
            parameters=[
                {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            objectives={
                "performance": ObjectiveProperties(minimize=False),  # Maximize performance
            },
            outcome_constraints=[
                "constraint <= 0",  # Constraint must be <= 0
            ],
            tracking_metrics=["cost"],  # Track cost but don't optimize it directly
        )
        
        print(f"{'Trial':<6} {'x':<8} {'y':<8} {'Cost':<10} {'Performance':<12} {'Constraint':<12}")
        print("-" * 65)
        
        best_trial = None
        best_performance = float('-inf')
        
        # Run optimization trials
        for trial_idx in range(15):
            # Get next trial parameters
            parameterization, trial_index = ax_client.get_next_trial()
            
            # Evaluate the trial
            results = evaluation_function(parameterization)
            
            # Complete the trial with results
            ax_client.complete_trial(trial_index=trial_index, raw_data=results)
            
            # Extract values for display
            x = parameterization["x"]
            y = parameterization["y"] 
            cost = results["cost"][0]
            performance = results["performance"][0]
            constraint = results["constraint"][0]
            
            print(f"{trial_idx+1:<6} {x:<8.3f} {y:<8.3f} {cost:<10.4f} {performance:<12.4f} {constraint:<12.4f}")
            
            # Track best feasible trial
            if constraint <= 0 and performance > best_performance:
                best_performance = performance
                best_trial = {
                    "trial": trial_idx + 1,
                    "params": parameterization,
                    "metrics": results
                }
        
        # Display results
        print("\nOptimization Results:")
        print("=" * 40)
        
        if best_trial:
            print(f"Best feasible trial: #{best_trial['trial']}")
            print(f"  Parameters: x={best_trial['params']['x']:.4f}, y={best_trial['params']['y']:.4f}")
            print(f"  Cost (analytical): {best_trial['metrics']['cost'][0]:.6f}")
            print(f"  Performance: {best_trial['metrics']['performance'][0]:.4f}")
            print(f"  Constraint: {best_trial['metrics']['constraint'][0]:.4f}")
        else:
            print("No feasible trials found.")
        
        # Get best point from model
        try:
            best_params, best_values = ax_client.get_best_point()
            print(f"\nModel-predicted best point:")
            print(f"  Parameters: {best_params}")
            print(f"  Predicted values: {best_values}")
        except Exception as e:
            print(f"Could not get model prediction: {e}")

        print("\nKey Insights:")
        print("✓ Analytical functions specified with zero uncertainty (0.0 SEM)")
        print("✓ Black-box functions include appropriate uncertainty estimates") 
        print("✓ AxClient handles mixed uncertainty types automatically")
        print("✓ Cost function computed exactly, no surrogate model needed")
        
        return True
        
    except ImportError as e:
        print(f"Could not import Ax components: {e}")
        return False


def demonstrate_direct_deterministic_model():
    """
    Demonstrate how to use DeterministicModel directly.
    
    This shows the core DeterministicModel concepts that would be integrated
    into Ax workflows through custom generation strategies.
    """
    
    try:
        # Import BoTorch models
        from botorch.models.deterministic import GenericDeterministicModel
        from botorch.models.model import ModelList
        from botorch.models import SingleTaskGP
        
        print("\nDemonstrating Direct DeterministicModel Integration:")
        print("=" * 55)
        
        # Create a torch version of analytical function
        def torch_analytical_cost(X: torch.Tensor) -> torch.Tensor:
            """Convert analytical cost to torch tensor format."""
            return (X**2).sum(dim=-1, keepdim=True)
        
        # Test the DeterministicModel directly
        det_model = GenericDeterministicModel(f=torch_analytical_cost)
        
        # Test with sample input
        test_input = torch.tensor([[0.3, 0.4]])
        analytical_result = torch_analytical_cost(test_input)
        model_result = det_model(test_input)
        
        print(f"Direct function:      {analytical_result.item():.6f}")
        print(f"DeterministicModel:   {model_result.item():.6f}") 
        print(f"Results match:        {torch.allclose(analytical_result, model_result)}")
        
        # Demonstrate ModelList integration
        print(f"\nDemonstrating ModelList with DeterministicModel:")
        
        # Create some dummy data for GP models
        X_dummy = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        Y_performance = torch.tensor([[0.1], [0.3], [0.2]])  
        Y_constraint = torch.tensor([[-0.1], [0.05], [-0.2]])
        
        # Create models
        performance_gp = SingleTaskGP(X_dummy, Y_performance)
        constraint_gp = SingleTaskGP(X_dummy, Y_constraint)
        
        # Create ModelList combining deterministic and probabilistic models
        mixed_model = ModelList(
            det_model,        # Model 0: Analytical cost (deterministic)
            performance_gp,   # Model 1: Performance (probabilistic)
            constraint_gp     # Model 2: Constraint (probabilistic)
        )
        
        print(f"Created ModelList with {len(mixed_model.models)} sub-models:")
        for i, model in enumerate(mixed_model.models):
            print(f"  Model {i}: {type(model).__name__}")
        
        # Test mixed model evaluation
        test_points = torch.tensor([[0.2, 0.3], [0.5, 0.4]])
        
        print(f"\nTesting mixed model on sample points:")
        print(f"{'Point':<15} {'Cost':<10} {'Perf Mean':<10} {'Perf Std':<10} {'Const Mean':<10} {'Const Std':<10}")
        print("-" * 70)
        
        for i, point in enumerate(test_points):
            point_tensor = point.unsqueeze(0)  # Add batch dimension
            
            # Evaluate with mixed model
            cost_output = mixed_model.models[0](point_tensor)
            performance_output = mixed_model.models[1].posterior(point_tensor)
            constraint_output = mixed_model.models[2].posterior(point_tensor)
            
            point_str = f"({point[0]:.1f}, {point[1]:.1f})"
            cost_val = cost_output.item()
            perf_mean = performance_output.mean.item()
            perf_std = performance_output.variance.sqrt().item()
            const_mean = constraint_output.mean.item()
            const_std = constraint_output.variance.sqrt().item()
            
            print(f"{point_str:<15} {cost_val:<10.4f} {perf_mean:<10.4f} {perf_std:<10.4f} {const_mean:<10.4f} {const_std:<10.4f}")
        
        print(f"\nKey Benefits of DeterministicModel:")
        print("✓ Zero approximation error for analytical functions")
        print("✓ No training data needed for known functions") 
        print("✓ Perfect predictions without uncertainty")
        print("✓ Can be combined with GP models in ModelList")
        
        print(f"\nIntegration with Ax:")
        print("• Use custom generation strategies with ModelList")
        print("• Specify analytical functions with zero uncertainty in evaluation")
        print("• Combine exact and approximate models seamlessly")
        
        return True
        
    except ImportError as e:
        print(f"Could not import BoTorch components: {e}")
        return False


def run_manual_optimization():
    """Fallback manual optimization if Ax imports fail."""
    
    print("Manual Optimization (Fallback Implementation)")
    print("=" * 50)
    print("Running manual optimization to demonstrate concepts...")
    
    print(f"{'Trial':<6} {'x':<8} {'y':<8} {'Cost':<10} {'Performance':<12} {'Constraint':<12}")
    print("-" * 65)
    
    best_trial = None
    best_performance = float('-inf')
    
    # Simple grid search + random sampling
    for trial_idx in range(15):
        # Generate random parameters
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        
        parameterization = {"x": x, "y": y}
        results = evaluation_function(parameterization)
        
        cost = results["cost"][0]
        performance = results["performance"][0]
        constraint = results["constraint"][0]
        
        print(f"{trial_idx+1:<6} {x:<8.3f} {y:<8.3f} {cost:<10.4f} {performance:<12.4f} {constraint:<12.4f}")
        
        # Track best feasible trial
        if constraint <= 0 and performance > best_performance:
            best_performance = performance
            best_trial = {
                "trial": trial_idx + 1,
                "params": parameterization,
                "metrics": results
            }
    
    print("\nOptimization Results:")
    print("=" * 40)
    
    if best_trial:
        print(f"Best feasible trial: #{best_trial['trial']}")
        print(f"  Parameters: x={best_trial['params']['x']:.4f}, y={best_trial['params']['y']:.4f}")
        print(f"  Cost (analytical): {best_trial['metrics']['cost'][0]:.6f}")
        print(f"  Performance: {best_trial['metrics']['performance'][0]:.4f}")
        print(f"  Constraint: {best_trial['metrics']['constraint'][0]:.4f}")
    else:
        print("No feasible trials found.")


def main():
    """Main demonstration function."""
    
    print("Ax-based DeterministicModel Example")
    print("This tutorial shows how to use Ax with mixed analytical/black-box optimization.\n")
    
    # Try to run with AxClient
    success = run_optimization_simple()
    
    if not success:
        print("\nFalling back to manual optimization...")
        run_manual_optimization()
    
    # Show direct DeterministicModel concepts
    print("\n" + "="*70)
    demonstrate_direct_deterministic_model()
    
    print("\nNext Steps:")
    print("1. For more complex DeterministicModel integration, see minimal_example.py")
    print("2. For production use, consider custom generation strategies with ModelList")
    print("3. The evaluation function approach shown here works for most use cases")


if __name__ == "__main__":
    main()