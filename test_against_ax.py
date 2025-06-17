#!/usr/bin/env python3
"""
Test against Ax examples to validate our standalone implementation.
"""

import numpy as np
import pandas as pd
from standalone_hypervolume import compute_hypervolume, compute_hypervolume_trace

def test_ax_example():
    """Test against the example from Ax test cases."""
    print("=== Testing against Ax example ===")
    
    # From ax/service/tests/test_best_point_utils.py:
    # df_wide = pd.DataFrame.from_records([
    #     {"m1": 1.0, "m2": 1.0, "feasible": True},      # Point 0: ref becomes (1,1)
    #     {"m1": 2.0, "m2": 3.0, "feasible": True},      # Point 1
    #     {"m1": 4.0, "m2": 4.0, "feasible": False},     # Point 2: infeasible  
    #     {"m1": 3.0, "m2": 2.0, "feasible": True},      # Point 3
    # ])
    # Expected cumulative HV: [0.0, 2.0, 2.0, 3.0]
    # Expected individual HV: [0.0, 2.0, 0.0, 2.0]
    
    # For maximize objectives, reference point should be the minimum values
    # Point 0: (1,1) becomes reference, so HV = 0
    # Point 1: (2,3) with ref (1,1) -> HV = (2-1)*(3-1) = 1*2 = 2  
    # Point 2: (4,4) infeasible, so HV stays 2
    # Point 3: (3,2) with ref (1,1), combined with (2,3)
    #   - Point (2,3): contributes (2-1)*(3-1) = 2
    #   - Point (3,2): contributes (3-1)*(2-1) = 2  
    #   - Overlap: (2-1)*(2-1) = 1
    #   - Total: 2 + 2 - 1 = 3
    
    points = np.array([
        [1.0, 1.0],  # This becomes the reference point  
        [2.0, 3.0],
        [4.0, 4.0],  # Will be excluded due to feasibility
        [3.0, 2.0]
    ])
    
    feasible = np.array([True, True, False, True])
    reference_point = np.array([1.0, 1.0])  # Inferred from minimum values
    
    # Test cumulative hypervolume
    cumulative_hvs = []
    for i in range(len(points)):
        current_points = points[:i+1]
        current_feasible = feasible[:i+1]
        hv = compute_hypervolume(current_points, reference_point, current_feasible)
        cumulative_hvs.append(hv)
    
    print(f"Computed cumulative HVs: {cumulative_hvs}")
    print(f"Expected cumulative HVs: [0.0, 2.0, 2.0, 3.0]")
    
    # Test individual hypervolume
    individual_hvs = []
    for i in range(len(points)):
        if not feasible[i]:
            individual_hvs.append(0.0)
        else:
            point = points[i:i+1]
            hv = compute_hypervolume(point, reference_point)
            individual_hvs.append(hv)
    
    print(f"Computed individual HVs: {individual_hvs}")
    print(f"Expected individual HVs: [0.0, 2.0, 0.0, 2.0]")
    

def test_with_different_reference_points():
    """Test how reference point affects computation."""
    print("\n=== Testing different reference points ===")
    
    points = np.array([[2.0, 3.0], [3.0, 2.0]])
    
    # Test with (0,0) reference
    ref1 = np.array([0.0, 0.0])
    hv1 = compute_hypervolume(points, ref1)
    print(f"Reference (0,0): HV = {hv1}")
    
    # Test with (1,1) reference  
    ref2 = np.array([1.0, 1.0])
    hv2 = compute_hypervolume(points, ref2)
    print(f"Reference (1,1): HV = {hv2}")
    
    # Test with (1.5, 1.5) reference
    ref3 = np.array([1.5, 1.5])
    hv3 = compute_hypervolume(points, ref3)
    print(f"Reference (1.5,1.5): HV = {hv3}")


def test_higher_dimensions():
    """Test 4D and 5D cases."""
    print("\n=== Testing higher dimensions ===")
    
    # 4D test
    points_4d = np.array([
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 3.0, 1.0, 1.0],
        [3.0, 1.0, 1.0, 1.0]
    ])
    ref_4d = np.array([0.0, 0.0, 0.0, 0.0])
    hv_4d = compute_hypervolume(points_4d, ref_4d)
    print(f"4D hypervolume: {hv_4d}")
    
    # 5D test
    points_5d = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0, 1.0, 1.0]
    ])
    ref_5d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    hv_5d = compute_hypervolume(points_5d, ref_5d)
    print(f"5D hypervolume: {hv_5d}")


if __name__ == "__main__":
    test_ax_example()
    test_with_different_reference_points()
    test_higher_dimensions()