#!/usr/bin/env python3
"""
Manual verification of 3D hypervolume calculation.
"""

import numpy as np
from standalone_hypervolume import compute_hypervolume, _find_non_dominated_indices

# Test case: Two 3D points
points = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
ref = np.array([0.0, 0.0, 0.0])

print("Points:", points)
print("Reference:", ref)

# Check if points are non-dominated
non_dom_idx = _find_non_dominated_indices(points)
print("Non-dominated indices:", non_dom_idx)
print("Non-dominated points:", points[non_dom_idx])

# Manual calculation:
# Point 1: (2,1,1) contributes volume 2*1*1 = 2
# Point 2: (1,2,1) contributes volume 1*2*1 = 2  
# Overlap: min(2,1) * min(1,2) * min(1,1) - 0 = 1*1*1 = 1
# Total: 2 + 2 - 1 = 3

p1_volume = 2.0 * 1.0 * 1.0  # 2
p2_volume = 1.0 * 2.0 * 1.0  # 2
overlap = 1.0 * 1.0 * 1.0    # 1 (from 0,0,0 to 1,1,1)
expected = p1_volume + p2_volume - overlap

print(f"Point 1 volume: {p1_volume}")
print(f"Point 2 volume: {p2_volume}")
print(f"Overlap volume: {overlap}")
print(f"Expected total: {expected}")

hv = compute_hypervolume(points, ref)
print(f"Computed hypervolume: {hv}")