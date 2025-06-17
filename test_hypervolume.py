#!/usr/bin/env python3
"""
Test cases for the standalone hypervolume implementation.
"""

import numpy as np
from standalone_hypervolume import compute_hypervolume, compute_hypervolume_trace


def test_simple_2d_cases():
    """Test simple 2D cases with known expected values."""
    print("=== Testing Simple 2D Cases ===")
    
    # Single point case
    points = np.array([[2.0, 3.0]])
    ref = np.array([0.0, 0.0])
    hv = compute_hypervolume(points, ref)
    expected = 2.0 * 3.0  # Should be 6.0
    print(f"Single point (2,3) with ref (0,0): {hv}, expected: {expected}")
    
    # Two non-dominated points
    points = np.array([[1.0, 3.0], [3.0, 1.0]])
    ref = np.array([0.0, 0.0])
    hv = compute_hypervolume(points, ref)
    # Should be area of two rectangles minus overlap
    # Rectangle 1: 1*3 = 3, Rectangle 2: 3*1 = 3, Overlap: 1*1 = 1
    # Total: 3 + 3 - 1 = 5
    expected = 5.0
    print(f"Two points (1,3) and (3,1) with ref (0,0): {hv}, expected: {expected}")
    
    # Three points forming L-shape
    points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    ref = np.array([0.0, 0.0])
    hv = compute_hypervolume(points, ref)
    print(f"Three points L-shape: {hv}")
    
    # Test with different reference point
    points = np.array([[2.0, 3.0], [3.0, 2.0]])
    ref = np.array([1.0, 1.0])
    hv = compute_hypervolume(points, ref)
    # Point 1 contributes (2-1)*(3-1) = 1*2 = 2
    # Point 2 contributes (3-1)*(2-1) = 2*1 = 2  
    # Overlap: (2-1)*(2-1) = 1*1 = 1
    # Total: 2 + 2 - 1 = 3
    expected = 3.0
    print(f"Two points with ref (1,1): {hv}, expected: {expected}")


def test_dominated_points():
    """Test cases with dominated points."""
    print("\n=== Testing Dominated Points ===")
    
    # One point dominates another
    points = np.array([[1.0, 1.0], [2.0, 2.0]])
    ref = np.array([0.0, 0.0])
    hv = compute_hypervolume(points, ref)
    # Only the dominating point (2,2) should contribute
    expected = 2.0 * 2.0  # 4.0
    print(f"Dominated case (1,1) and (2,2): {hv}, expected: {expected}")


def test_infeasible_points():
    """Test with feasible/infeasible points."""
    print("\n=== Testing Feasible/Infeasible Points ===")
    
    points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    ref = np.array([0.0, 0.0])
    feasible = np.array([True, False, True])  # Middle point is infeasible
    
    hv = compute_hypervolume(points, ref, feasible)
    print(f"With infeasible middle point: {hv}")
    
    # Compare with all feasible
    hv_all = compute_hypervolume(points, ref)
    print(f"All feasible: {hv_all}")


def test_3d_simple():
    """Test simple 3D case."""
    print("\n=== Testing 3D Cases ===")
    
    # Single point
    points = np.array([[2.0, 2.0, 2.0]])
    ref = np.array([0.0, 0.0, 0.0])
    hv = compute_hypervolume(points, ref)
    expected = 2.0 * 2.0 * 2.0  # 8.0
    print(f"Single 3D point (2,2,2): {hv}, expected: {expected}")
    
    # Two points
    points = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    ref = np.array([0.0, 0.0, 0.0])
    hv = compute_hypervolume(points, ref)
    print(f"Two 3D points: {hv}")


def test_hypervolume_trace():
    """Test hypervolume trace functionality."""
    print("\n=== Testing Hypervolume Trace ===")
    
    points = np.array([
        [1.0, 1.0],
        [2.0, 3.0], 
        [1.5, 1.5],  # This should be dominated
        [3.0, 2.0]
    ])
    ref = np.array([0.0, 0.0])
    
    # Cumulative trace
    hv_trace_cum = compute_hypervolume_trace(points, ref, cumulative=True)
    print(f"Cumulative HV trace: {hv_trace_cum}")
    
    # Individual trace  
    hv_trace_ind = compute_hypervolume_trace(points, ref, cumulative=False)
    print(f"Individual HV trace: {hv_trace_ind}")


if __name__ == "__main__":
    test_simple_2d_cases()
    test_dominated_points()
    test_infeasible_points()
    test_3d_simple()
    test_hypervolume_trace()