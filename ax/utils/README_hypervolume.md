# Standalone Hypervolume Calculation

This directory contains a standalone implementation of hypervolume calculation extracted from the Ax optimization framework. The implementation is designed to be self-contained and easily copy-pastable.

## Files

- `hypervolume.py` - Main implementation with all hypervolume functions
- `tests/test_hypervolume.py` - Comprehensive test suite
- `../examples/standalone_hypervolume_usage.py` - Usage examples

## Quick Start

```python
import numpy as np
from ax.utils.hypervolume import compute_hypervolume

# Your multi-objective optimization results
points = np.array([
    [2.0, 3.0],  # Point 1: [objective1, objective2]
    [3.0, 2.0],  # Point 2: [objective1, objective2]
    [1.5, 1.5],  # Point 3: [objective1, objective2] (will be dominated)
])

# Reference point (typically worse than any point you're interested in)
reference_point = np.array([0.0, 0.0])

# Compute hypervolume
hv = compute_hypervolume(points, reference_point)
print(f"Hypervolume: {hv}")  # Output: 8.0
```

## Key Functions

### `compute_hypervolume(points, reference_point, feasible_mask=None)`
Computes the hypervolume of a set of points relative to a reference point.

### `compute_hypervolume_trace(points, reference_point, feasible_mask=None, cumulative=True)`
Computes hypervolume trace over a sequence of points (useful for tracking optimization progress).

### `infer_reference_point(points, feasible_mask=None, offset_fraction=0.1)`
Automatically infers a reasonable reference point from data.

## Dependencies

- `numpy` only (no heavy dependencies like pymoo or botorch)

## Copy-Paste Usage

To use this independently of Ax:

1. Copy the file `ax/utils/hypervolume.py` 
2. Import and use the functions as shown above
3. Only numpy is required as a dependency

## Algorithm Details

- **1D**: Simple maximum range calculation
- **2D**: Efficient sweep line algorithm  
- **3D+**: Inclusion-exclusion principle with Pareto filtering

The implementation handles:
- Multiple dimensions (tested up to 5D, works for higher)
- Feasible/infeasible points
- Dominated points (automatically filtered to Pareto frontier)
- Various reference point scenarios

## Validation

The implementation has been validated against:
- Known mathematical cases with expected results
- Existing Ax framework test cases
- Various dimensional scenarios (1D through 5D)
- Edge cases (empty sets, dominated points, infeasible points)

## Performance

- Efficient for small to moderate numbers of points (<1000)
- Suitable for typical multi-objective optimization scenarios
- For very large point sets or high dimensions (>6), specialized algorithms like WFG may be more efficient

## Example Applications

- Multi-objective hyperparameter optimization
- Pareto frontier quality assessment
- Optimization progress tracking
- Multi-criteria decision making