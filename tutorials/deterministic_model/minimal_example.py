#!/usr/bin/env python3
"""
Minimal Working Example: Using DeterministicModel in BoTorch

This script demonstrates how to use DeterministicModel for optimization problems
where some objectives or constraints are analytically known.

Addresses GitHub issues #935 and #1192 in facebook/Ax repository.

Author: Generated for Ax tutorial
"""

import torch
import numpy as np
from typing import Callable

# Core BoTorch imports
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models import SingleTaskGP
from botorch.models.model import ModelList
from torch.quasirandom import SobolEngine

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def analytical_cost_function(x: torch.Tensor) -> torch.Tensor:
    """
    Analytical cost function: f(x,y) = x² + y²
    
    This could represent monetary cost, energy consumption, etc.
    Args:
        x: Input tensor of shape (..., 2) where columns are [x, y]
    Returns:
        Cost values of shape (..., 1)
    """
    return (x**2).sum(dim=-1, keepdim=True)


def black_box_performance(x: torch.Tensor) -> torch.Tensor:
    """
    Black-box performance function that we want to maximize.
    
    Simulates an expensive-to-evaluate function like simulation results.
    For demonstration: f(x,y) = -(x-0.3)² - (y-0.3)² + 0.5 + noise
    """
    base_value = -((x[..., 0:1] - 0.3)**2 + (x[..., 1:2] - 0.3)**2) + 0.5
    noise = 0.1 * torch.randn_like(base_value)
    return base_value + noise


def constraint_function(x: torch.Tensor) -> torch.Tensor:
    """
    Black-box constraint function: g(x,y) = x + y - 0.8 ≤ 0
    """
    return x.sum(dim=-1, keepdim=True) - 0.8 + 0.05 * torch.randn(x.shape[0], 1)


def main():
    """Main demonstration function."""
    
    print("=" * 60)
    print("DeterministicModel Minimal Working Example")
    print("=" * 60)
    
    # Step 1: Create DeterministicModel for analytical function
    print("\n1. Creating DeterministicModel for analytical cost function...")
    deterministic_cost_model = GenericDeterministicModel(f=analytical_cost_function)
    
    # Test the deterministic model
    test_input = torch.tensor([[0.4, 0.5]])
    analytical_result = analytical_cost_function(test_input)
    model_result = deterministic_cost_model(test_input)
    
    print(f"   Direct function call: {analytical_result.item():.6f}")
    print(f"   DeterministicModel call: {model_result.item():.6f}")
    print(f"   Results match: {torch.allclose(analytical_result, model_result)}")
    
    # Step 2: Generate initial data for black-box functions
    print("\n2. Generating initial data for black-box functions...")
    n_initial = 8
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    
    # Sobol sampling for initial points
    sobol = SobolEngine(dimension=2, scramble=True, seed=42)
    initial_X = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_initial)
    
    # Evaluate black-box functions at initial points
    initial_performance = black_box_performance(initial_X)
    initial_constraints = constraint_function(initial_X)
    
    print(f"   Generated {n_initial} initial points")
    print(f"   X shape: {initial_X.shape}")
    print(f"   Performance range: [{initial_performance.min().item():.3f}, {initial_performance.max().item():.3f}]")
    print(f"   Constraint range: [{initial_constraints.min().item():.3f}, {initial_constraints.max().item():.3f}]")
    
    # Step 3: Create GP models for black-box functions
    print("\n3. Creating GP models for black-box functions...")
    performance_gp = SingleTaskGP(initial_X, initial_performance)
    constraint_gp = SingleTaskGP(initial_X, initial_constraints)
    
    # Step 4: Create ModelList combining deterministic and probabilistic models
    print("\n4. Creating mixed ModelList...")
    mixed_model = ModelList(
        deterministic_cost_model,  # Model 0: Analytical cost (deterministic)
        performance_gp,            # Model 1: Performance (probabilistic)
        constraint_gp              # Model 2: Constraint (probabilistic)
    )
    
    print(f"   Mixed model created with {len(mixed_model.models)} sub-models:")
    for i, model in enumerate(mixed_model.models):
        print(f"     Model {i}: {type(model).__name__}")
    
    # Step 5: Test the mixed model
    print("\n5. Testing mixed model evaluation...")
    test_points = torch.tensor([[0.2, 0.3], [0.5, 0.4], [0.8, 0.1]])
    
    print(f"   {'Point':<15} {'Cost':<10} {'Perf Mean':<10} {'Perf Std':<10} {'Const Mean':<10} {'Const Std':<10}")
    print("   " + "-" * 70)
    
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
        
        print(f"   {point_str:<15} {cost_val:<10.4f} {perf_mean:<10.4f} {perf_std:<10.4f} {const_mean:<10.4f} {const_std:<10.4f}")
    
    # Step 6: Comparison with GP for analytical function
    print("\n6. Comparing DeterministicModel vs GP for analytical function...")
    
    # Create GP for analytical function (inefficient approach)
    analytical_values = analytical_cost_function(initial_X)
    gp_for_analytical = SingleTaskGP(initial_X, analytical_values)
    
    # Test on new points
    test_comparison_points = torch.tensor([[0.2, 0.7], [0.8, 0.3]])
    true_values = analytical_cost_function(test_comparison_points)
    
    # DeterministicModel predictions (exact)
    det_predictions = deterministic_cost_model(test_comparison_points)
    
    # GP predictions (approximate)
    gp_predictions = gp_for_analytical.posterior(test_comparison_points)
    
    print(f"   {'Point':<15} {'True Value':<12} {'Deterministic':<12} {'GP Mean':<12} {'GP Std':<12}")
    print("   " + "-" * 60)
    
    for i in range(len(test_comparison_points)):
        point_str = f"({test_comparison_points[i, 0]:.1f}, {test_comparison_points[i, 1]:.1f})"
        true_val = true_values[i].item()
        det_val = det_predictions[i].item()
        gp_mean = gp_predictions.mean[i].item()
        gp_std = gp_predictions.variance[i].sqrt().item()
        
        print(f"   {point_str:<15} {true_val:<12.6f} {det_val:<12.6f} {gp_mean:<12.6f} {gp_std:<12.6f}")
    
    # Calculate errors
    det_error = torch.abs(det_predictions - true_values).mean()
    gp_error = torch.abs(gp_predictions.mean - true_values).mean()
    
    print(f"\n   Mean Absolute Error:")
    print(f"     DeterministicModel: {det_error:.10f}")
    print(f"     Gaussian Process:   {gp_error:.6f}")
    
    # Step 7: Summary
    print("\n7. Summary and Key Benefits:")
    print("   ✓ DeterministicModel provides exact predictions for analytical functions")
    print("   ✓ No approximation error compared to GP surrogate models")
    print("   ✓ No need to 'learn' functions you already know analytically")
    print("   ✓ Seamless integration with probabilistic models in ModelList")
    print("   ✓ Efficient for cost functions, physical constraints, etc.")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    

if __name__ == "__main__":
    main()