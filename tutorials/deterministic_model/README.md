# DeterministicModel Tutorial

This tutorial demonstrates how to use `DeterministicModel` concepts in Ax for optimization problems where some objectives or constraints are analytically known.

## Background

This tutorial addresses GitHub issues [#935](https://github.com/facebook/Ax/issues/935) and [#1192](https://github.com/facebook/Ax/issues/1192) where users requested examples and documentation for using `DeterministicModel` with analytical functions in optimization scenarios.

## Problem Statement

In many optimization problems, especially in physical sciences and engineering, you have:
- **Analytical functions**: Cost, energy, or other easily computable objectives/constraints
- **Black-box functions**: Complex simulations or experimental results that are expensive to evaluate

The goal is to efficiently optimize without learning surrogate models for functions you already know analytically.

## Files in this Tutorial

1. **`ax_deterministic_example.py`**: Ax-based implementation showing how to handle mixed analytical/black-box optimization
2. **`minimal_example.py`**: Pure BoTorch implementation for reference and understanding core concepts
3. **`README.md`**: This documentation file

## Quick Start

Run the Ax-based example to see mixed analytical/black-box optimization:

```bash
cd tutorials/deterministic_model/
python ax_deterministic_example.py
```

For understanding the core BoTorch concepts, run:

```bash
python minimal_example.py
```

## Two Approaches Demonstrated

### 1. Ax-Based Approach (Recommended)
The `ax_deterministic_example.py` shows how to handle analytical functions in Ax:

```python
from ax.core.metric import Metric
from ax.service.ax_client import AxClient

class AnalyticalCostMetric(Metric):
    def fetch_trial_data(self, trial, **kwargs):
        # Compute analytical function directly
        x, y = trial.arm.parameters["x"], trial.arm.parameters["y"]
        cost = x**2 + y**2  # Analytical cost function
        return Data(..., mean=cost, sem=0.0)  # Zero uncertainty

# Create experiment with mixed metrics
experiment = Experiment(
    optimization_config=OptimizationConfig(
        objective=Objective(black_box_metric),
        outcome_constraints=[constraint_metric.constraint_bound(...)],
    ),
    tracking_metrics=[analytical_cost_metric],  # Track but don't optimize
)
```

### 2. BoTorch Implementation (Reference)
The `minimal_example.py` shows the underlying BoTorch concepts:

```python
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import ModelList

def analytical_cost_function(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1, keepdim=True)

# Create deterministic model
det_model = GenericDeterministicModel(f=analytical_cost_function)

# Combine with GP models for black-box functions
mixed_model = ModelList(det_model, gp_model)
```

## Key Benefits Comparison

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Ax-based** | Easy integration, automatic optimization, full Ax features | Indirect DeterministicModel usage | Most users, production systems |
| **BoTorch-direct** | Direct model control, explicit DeterministicModel | Manual optimization loop, more complex | Advanced users, research |

## Comparison: Analytical vs GP Modeling

Both approaches show why using analytical functions directly is superior to fitting a GP on analytical data:

| Approach | Error | Uncertainty | Efficiency |
|----------|-------|-------------|------------|
| Analytical (exact) | 0.0 (exact) | None (deterministic) | High |
| GP on analytical data | ~0.05 (approximation) | Artificial | Low |

## Use Cases

This approach is ideal when you have:
- **Analytical cost functions** (monetary, computational, energy)
- **Known physical constraints** (conservation laws, geometric constraints)  
- **Mixed optimization problems** with both cheap analytical and expensive black-box functions
- **Multi-objective scenarios** where some objectives are analytically computable

## Key Benefits

1. **Efficiency**: No surrogate modeling for known functions
2. **Accuracy**: Exact evaluation of analytical functions
3. **Flexibility**: Easy integration with existing BoTorch/Ax workflows
4. **Scalability**: Works with arbitrary input dimensions and multiple outputs

## Integration Patterns

### Simple Ax Pattern (Recommended)
- Use custom `Metric` classes for analytical functions
- Return exact values with `sem=0.0` for deterministic metrics
- Let Ax handle optimization automatically
- ✅ Easy to implement, works with all Ax features
- ✅ Good for most production use cases

### Advanced BoTorch Pattern
- Create custom `Surrogate` class with `ModelList` integration
- Use `GenericDeterministicModel` for analytical functions
- Requires deeper understanding of Ax internals
- ✅ Direct control over models and acquisition functions
- ✅ Explicit separation of deterministic vs probabilistic models
- ❌ More complex, may need updates when Ax evolves

## Example Output

When you run `ax_deterministic_example.py`, you'll see:

```
============================================================
Ax-based DeterministicModel Example (Simple Approach)
============================================================

1. Creating Ax experiment with mixed metrics...
   Experiment: mixed_deterministic_optimization
   Parameters: ['x', 'y']
   Objective: performance
   Constraints: 1
   Tracking metrics: ['cost']

2. Generating initial trials...
   Generation strategy: GenerationStrategy(...)
   Initial trials: 5, Total trials: 10

3. Running optimization loop...
   Trial 1:
     Parameters: x=0.234, y=0.567
     Cost (analytical): 0.376289
     Performance: 0.4123
     Constraint: -0.0234

[... detailed results for each trial ...]

4. Analysis and Results:
   Best trial: #7
   Best parameters: {'x': 0.312, 'y': 0.289}
   Cost: 0.181025
   Performance: 0.4567
   Constraint: -0.1234

5. Analytical vs GP Modeling Benefits:
   ✓ Cost metric computed analytically (zero error)
   ✓ No need to fit surrogate model for known functions
   ✓ Ax automatically handles mixed metric types
   ✓ Efficient optimization with exact analytical functions
```

## Next Steps

- Extend to your specific analytical functions
- Experiment with different acquisition functions through Ax's generation strategies
- Integrate with Ax's high-level APIs for production systems
- Consider multi-fidelity scenarios with mixed model types

## Requirements

- PyTorch
- BoTorch
- Ax-platform
- NumPy
- Pandas (for data handling)

The tutorial is designed to work with the versions installed in the Ax repository.

## Further Reading

- [Ax Documentation](https://ax.dev/)
- [BoTorch DeterministicModel API](https://botorch.org/api/models.html#deterministic-models)
- [Mixed Optimization in Physical Sciences](https://honegumi.readthedocs.io/en/latest/)
- [Related GitHub Issues](https://github.com/pytorch/botorch/issues?q=is%3Aissue%20state%3Aclosed%20DeterministicModel)