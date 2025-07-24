# DeterministicModel Tutorial

This tutorial demonstrates how to use `DeterministicModel` in BoTorch for optimization problems where some objectives or constraints are analytically known.

## Background

This tutorial addresses GitHub issues [#935](https://github.com/facebook/Ax/issues/935) and [#1192](https://github.com/facebook/Ax/issues/1192) where users requested examples and documentation for using `DeterministicModel` with analytical functions in optimization scenarios.

## Problem Statement

In many optimization problems, especially in physical sciences and engineering, you have:
- **Analytical functions**: Cost, energy, or other easily computable objectives/constraints
- **Black-box functions**: Complex simulations or experimental results that are expensive to evaluate

The goal is to efficiently optimize without learning surrogate models for functions you already know analytically.

## Files in this Tutorial

1. **`deterministic_model_example.ipynb`**: Comprehensive Jupyter notebook tutorial with visualizations
2. **`minimal_example.py`**: Standalone Python script demonstrating core concepts
3. **`README.md`**: This documentation file

## Quick Start

Run the minimal example to see DeterministicModel in action:

```bash
cd tutorials/deterministic_model/
python minimal_example.py
```

## Key Concepts Demonstrated

### 1. Basic DeterministicModel Usage
```python
from botorch.models.deterministic import GenericDeterministicModel

def analytical_cost_function(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1, keepdim=True)

# Create deterministic model
det_model = GenericDeterministicModel(f=analytical_cost_function)

# Use it like any other BoTorch model
result = det_model(torch.tensor([[0.4, 0.5]]))
```

### 2. Mixed Models with ModelList
```python
from botorch.models.model import ModelList
from botorch.models import SingleTaskGP

# Combine deterministic and probabilistic models
mixed_model = ModelList(
    det_model,           # Analytical function
    gp_model            # Black-box function modeled with GP
)

# Access individual models
cost = mixed_model.models[0](x)                    # Deterministic
performance = mixed_model.models[1].posterior(x)   # Probabilistic
```

### 3. Comparison: DeterministicModel vs GP

The tutorial shows why using DeterministicModel is superior to fitting a GP on analytical functions:

| Approach | Error | Uncertainty | Efficiency |
|----------|-------|-------------|------------|
| DeterministicModel | 0.0 (exact) | None (deterministic) | High |
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

## Integration with Ax

While this tutorial focuses on BoTorch components, these patterns can be integrated into Ax through:
- Custom `Surrogate` classes
- Modular BoTorch interface configurations  
- Custom `ModelBridge` implementations

## Example Output

When you run `minimal_example.py`, you'll see:

```
============================================================
DeterministicModel Minimal Working Example
============================================================

1. Creating DeterministicModel for analytical cost function...
   Direct function call: 0.410000
   DeterministicModel call: 0.410000
   Results match: True

2. Generating initial data for black-box functions...
   Generated 8 initial points
   
3. Creating GP models for black-box functions...

4. Creating mixed ModelList...
   Mixed model created with 3 sub-models:
     Model 0: GenericDeterministicModel
     Model 1: SingleTaskGP
     Model 2: SingleTaskGP

5. Testing mixed model evaluation...
   [Detailed results table]

6. Comparing DeterministicModel vs GP for analytical function...
   [Comparison showing zero error for DeterministicModel]

7. Summary and Key Benefits:
   ✓ DeterministicModel provides exact predictions
   ✓ No approximation error
   ✓ Seamless integration with probabilistic models
   ✓ Efficient for known analytical functions
```

## Next Steps

- Extend to your specific analytical functions
- Experiment with different acquisition functions  
- Integrate with Ax's high-level APIs
- Consider multi-fidelity scenarios with mixed model types

## Requirements

- PyTorch
- BoTorch
- NumPy

The tutorial is designed to work with the versions installed in the Ax repository.