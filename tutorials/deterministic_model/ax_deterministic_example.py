#!/usr/bin/env python3
"""
Ax-based Implementation: Using DeterministicModel with Mixed Optimization

This script demonstrates how to properly integrate DeterministicModel with Ax 
for mixed analytical/black-box optimization, following the SEBO pattern.

Addresses GitHub issues #935 and #1192 in facebook/Ax repository.

Author: Generated for Ax tutorial
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MixedOptimizationAcquisition:
    """
    Custom acquisition class that integrates DeterministicModel with Ax.
    
    This follows the SEBO pattern to demonstrate how to properly combine
    analytical (deterministic) and black-box (probabilistic) models in Ax.
    """
    
    def __init__(
        self, 
        surrogate,
        search_space_digest, 
        torch_opt_config,
        analytical_function: callable,
        botorch_acqf_class=None,
        options: Optional[Dict[str, Any]] = None
    ):
        from ax.generators.torch.botorch_modular.acquisition import Acquisition
        from botorch.models.deterministic import GenericDeterministicModel
        from botorch.models.model import ModelList
        from botorch.utils.datasets import SupervisedDataset
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        
        # Default acquisition function if none provided
        if botorch_acqf_class is None:
            botorch_acqf_class = qLogNoisyExpectedHypervolumeImprovement
            
        tkwargs = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = {} if options is None else options
        
        # Create deterministic model for analytical function
        self.analytical_function = analytical_function
        self.deterministic_model = GenericDeterministicModel(f=analytical_function)
        
        # Create modified surrogate following SEBO pattern
        surrogate_modified = deepcopy(surrogate)
        
        # Get training data from original surrogate
        X_train = surrogate_modified.Xs[0].clone()
        
        # Evaluate analytical function on training data
        Y_analytical = self.deterministic_model(X_train)
        
        # Add training data for deterministic model with zero variance (noiseless)
        surrogate_modified._training_data.append(
            SupervisedDataset(
                X=X_train,
                Y=Y_analytical,
                Yvar=torch.zeros(X_train.shape[0], 1, **tkwargs),  # Zero variance for deterministic
                feature_names=surrogate_modified.training_data[0].feature_names,
                outcome_names=["analytical_cost"]
            )
        )
        
        # Update model to ModelList combining original model + deterministic model
        surrogate_modified._model = ModelList(surrogate.model, self.deterministic_model)
        
        # Update torch config to handle additional objective
        torch_opt_config_modified = self._transform_torch_config(
            torch_opt_config=torch_opt_config, **tkwargs
        )
        
        # Initialize the parent acquisition with modified surrogate
        self.acquisition = Acquisition(
            surrogate=surrogate_modified,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config_modified,
            botorch_acqf_class=botorch_acqf_class,
            options=options,
        )
        
    def _transform_torch_config(self, torch_opt_config, **tkwargs):
        """Transform torch config to include analytical function as additional outcome."""
        from ax.generators.torch_base import TorchOptConfig
        
        # Add weight for analytical objective (negative to minimize cost)
        objective_weights_modified = torch.cat([
            torch_opt_config.objective_weights, 
            torch.tensor([-1.0], **tkwargs)  # Minimize cost
        ])
        
        # Update outcome constraints if they exist
        if torch_opt_config.outcome_constraints is not None:
            A, b = torch_opt_config.outcome_constraints
            outcome_constraints_modified = (
                torch.cat([A, torch.zeros(A.shape[0], 1, **tkwargs)], dim=1),
                b,
            )
        else:
            outcome_constraints_modified = None
            
        # Update objective thresholds if they exist
        if torch_opt_config.objective_thresholds is not None:
            objective_thresholds_modified = torch.cat([
                torch_opt_config.objective_thresholds,
                torch.tensor([1.0], **tkwargs),  # Threshold for cost
            ])
        else:
            objective_thresholds_modified = None
            
        return TorchOptConfig(
            objective_weights=objective_weights_modified,
            outcome_constraints=outcome_constraints_modified, 
            objective_thresholds=objective_thresholds_modified,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            pending_observations=torch_opt_config.pending_observations,
            model_gen_options=torch_opt_config.model_gen_options,
            rounding_func=torch_opt_config.rounding_func,
            opt_config_metrics=torch_opt_config.opt_config_metrics,
            is_moo=True,  # Mixed optimization is always multi-objective
        )
    
    def optimize(self, *args, **kwargs):
        """Delegate optimization to the underlying acquisition function."""
        return self.acquisition.optimize(*args, **kwargs)


def analytical_cost_function_torch(X: torch.Tensor) -> torch.Tensor:
    """Analytical cost function in torch tensor format: f(x,y) = x² + y²"""
    return (X**2).sum(dim=-1, keepdim=True)


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
    
    Note: This only evaluates the BLACK-BOX functions. The analytical function
    is handled by the DeterministicModel and doesn't need evaluation data.
    """
    x = parameterization["x"]
    y = parameterization["y"]
    
    return {
        "performance": (black_box_performance(x, y), 0.05),  # Black-box with uncertainty  
        "constraint": (constraint_function(x, y), 0.02),     # Black-box with uncertainty
    }


def demonstrate_proper_deterministic_integration():
    """
    Demonstrate proper DeterministicModel integration with Ax following SEBO pattern.
    """
    print("Proper DeterministicModel Integration with Ax")
    print("=" * 50)
    
    try:
        # Core imports
        from ax.core.search_space import SearchSpaceDigest
        from ax.generators.torch.botorch_modular.surrogate import Surrogate
        from ax.generators.torch_base import TorchOptConfig
        from botorch.models import SingleTaskGP
        from botorch.utils.datasets import SupervisedDataset
        from botorch.models.model import ModelList
        from botorch.models.deterministic import GenericDeterministicModel
        
        # Create some initial training data for black-box functions
        X_init = torch.tensor([[0.2, 0.3], [0.7, 0.4], [0.1, 0.8], [0.9, 0.2]])
        
        # Evaluate black-box functions
        Y_performance = torch.tensor([[black_box_performance(x[0].item(), x[1].item())] for x in X_init])
        Y_constraint = torch.tensor([[constraint_function(x[0].item(), x[1].item())] for x in X_init])
        
        # Create datasets
        performance_data = SupervisedDataset(
            X=X_init, Y=Y_performance, 
            feature_names=["x", "y"], outcome_names=["performance"]
        )
        constraint_data = SupervisedDataset(
            X=X_init, Y=Y_constraint,
            feature_names=["x", "y"], outcome_names=["constraint"] 
        )
        
        # Create search space digest
        search_space_digest = SearchSpaceDigest(
            feature_names=["x", "y"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        
        # Create surrogate for black-box functions
        surrogate = Surrogate()
        surrogate.fit(
            datasets=[performance_data, constraint_data],
            search_space_digest=search_space_digest,
        )
        
        print("Original surrogate:")
        print(f"  Model type: {type(surrogate.model).__name__}")
        if hasattr(surrogate.model, 'models'):
            print(f"  Sub-models: {[type(m).__name__ for m in surrogate.model.models]}")
        
        # Create DeterministicModel for analytical cost function
        det_model = GenericDeterministicModel(f=analytical_cost_function_torch)
        
        # Test the deterministic model
        test_point = torch.tensor([[0.3, 0.4]])
        direct_result = analytical_cost_function_torch(test_point)
        model_result = det_model(test_point)
        
        print(f"\nDeterministicModel test:")
        print(f"  Direct function: {direct_result.item():.6f}")
        print(f"  Model result:    {model_result.item():.6f}")
        print(f"  Match: {torch.allclose(direct_result, model_result)}")
        
        # Create mixed ModelList (this is the key integration!)
        mixed_model = ModelList(
            surrogate.model.models[0],  # Performance GP
            surrogate.model.models[1],  # Constraint GP  
            det_model                   # Analytical cost (deterministic)
        )
        
        print(f"\nMixed ModelList created:")
        print(f"  Total models: {len(mixed_model.models)}")
        for i, model in enumerate(mixed_model.models):
            print(f"  Model {i}: {type(model).__name__}")
            
        # Test mixed model evaluation
        print(f"\nTesting mixed model evaluation:")
        test_points = torch.tensor([[0.2, 0.3], [0.6, 0.7]])
        
        for i, point in enumerate(test_points):
            point_tensor = point.unsqueeze(0)
            
            # Model 0: Performance (probabilistic)
            perf_posterior = mixed_model.models[0].posterior(point_tensor)
            perf_mean = perf_posterior.mean.item()
            perf_std = perf_posterior.variance.sqrt().item()
            
            # Model 1: Constraint (probabilistic) 
            const_posterior = mixed_model.models[1].posterior(point_tensor)
            const_mean = const_posterior.mean.item()
            const_std = const_posterior.variance.sqrt().item()
            
            # Model 2: Cost (deterministic)
            cost_value = mixed_model.models[2](point_tensor).item()
            
            print(f"  Point ({point[0]:.1f}, {point[1]:.1f}):")
            print(f"    Performance: {perf_mean:.4f} ± {perf_std:.4f}")
            print(f"    Constraint:  {const_mean:.4f} ± {const_std:.4f}")
            print(f"    Cost:        {cost_value:.6f} (exact)")
            
        print(f"\nKey Insights:")
        print("✓ DeterministicModel provides exact analytical evaluations")
        print("✓ No approximation error for analytical functions")
        print("✓ Seamlessly combined with GP models in ModelList")
        print("✓ Ready for use in acquisition functions and optimization")
        
        # Show how this would be used in a custom acquisition function
        print(f"\nCustom Acquisition Integration:")
        print("• Create custom acquisition class following SEBO pattern")
        print("• Add DeterministicModel to surrogate via ModelList")
        print("• Update TorchOptConfig for multi-objective optimization")
        print("• Use in GenerationStrategy for full Ax integration")
        
        return True
        
    except ImportError as e:
        print(f"Could not import required components: {e}")
        return False
    except Exception as e:
        print(f"Error in demonstration: {e}")
        return False


def demonstrate_custom_acquisition():
    """Demonstrate the custom acquisition function with DeterministicModel."""
    print("\nCustom Acquisition with DeterministicModel")
    print("=" * 45)
    
    try:
        # This would be used in a full Ax workflow with custom generation strategy
        print("Conceptual integration with Ax:")
        print("1. Create custom acquisition class (like MixedOptimizationAcquisition)")
        print("2. Integrate with GenerationStrategy")
        print("3. Use in Experiment for full optimization workflow")
        print("4. Analytical functions computed exactly via DeterministicModel")
        print("5. Black-box functions modeled with GP uncertainty")
        
        print(f"\nExample usage in GenerationStrategy:")
        print("""
        from ax.modelbridge.registry import Models
        from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
        
        # Custom generation step using DeterministicModel
        generation_strategy = GenerationStrategy([
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=20,
                model_kwargs={
                    "acquisition_class": MixedOptimizationAcquisition,
                    "acquisition_options": {
                        "analytical_function": analytical_cost_function_torch
                    }
                }
            )
        ])
        """)
        
        return True
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        return False


def main():
    """Main demonstration function."""
    
    print("Ax + DeterministicModel Integration Tutorial")
    print("This demonstrates proper DeterministicModel usage with Ax")
    print("Following the SEBO pattern for mixed analytical/black-box optimization.\n")
    
    # Demonstrate proper DeterministicModel integration
    success = demonstrate_proper_deterministic_integration()
    
    if success:
        # Show custom acquisition pattern
        demonstrate_custom_acquisition()
        
        print(f"\n" + "="*60)
        print("Summary: How SEBO Uses DeterministicModel")
        print("="*60)
        print("1. Creates GenericDeterministicModel for analytical functions")
        print("2. Modifies Surrogate by adding deterministic training data")
        print("3. Combines models using ModelList (GP + DeterministicModel)")
        print("4. Updates TorchOptConfig for multi-objective optimization")
        print("5. Integrates through custom Acquisition class")
        print("\nThis pattern allows exact analytical evaluations within Ax workflows!")
        
    else:
        print("Could not run demonstration due to missing dependencies.")
        
    print(f"\nFor complete working example, see:")
    print("• SEBO implementation: ax/generators/torch/botorch_modular/sebo.py") 
    print("• SEBO tests: ax/generators/torch/tests/test_sebo.py")
    print("• This demonstrates the proper Ax + DeterministicModel integration pattern")


if __name__ == "__main__":
    main()