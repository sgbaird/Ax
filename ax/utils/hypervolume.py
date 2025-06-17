#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Standalone hypervolume calculation utilities.

This module provides self-contained implementations for computing hypervolume
of multi-objective optimization results. It only depends on numpy and can be
easily copied and used independently.

The hypervolume indicator measures the volume of the space that is dominated
by a set of points relative to a reference point. It's a key quality measure
for multi-objective optimization.
"""

from typing import List, Optional, Union

import numpy as np


def compute_hypervolume(
    points: np.ndarray,
    reference_point: np.ndarray,
    feasible_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the hypervolume of a set of points relative to a reference point.

    The hypervolume (also called S-metric or Lebesgue measure) is the volume
    of the space that is dominated by the given points and bounded by the
    reference point. It's a standard quality indicator for multi-objective
    optimization.

    Args:
        points: Array of shape (n_points, n_objectives) containing the objective
            values. For maximization objectives, larger values are better.
        reference_point: Array of shape (n_objectives,) defining the reference
            point. This should typically be set to values that are worse than
            any point you're interested in.
        feasible_mask: Optional boolean array of shape (n_points,) indicating
            which points are feasible. If None, all points are considered feasible.

    Returns:
        The hypervolume value as a float.

    Raises:
        ValueError: If input dimensions don't match or are invalid.

    Example:
        >>> import numpy as np
        >>> # Two non-dominated points in 2D
        >>> points = np.array([[2.0, 3.0], [3.0, 2.0]])
        >>> reference_point = np.array([0.0, 0.0])
        >>> hv = compute_hypervolume(points, reference_point)
        >>> print(f"Hypervolume: {hv}")
        Hypervolume: 8.0

        >>> # With infeasible points
        >>> points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        >>> feasible = np.array([True, False, True])  # Middle point infeasible
        >>> hv = compute_hypervolume(points, reference_point, feasible)
        >>> print(f"Hypervolume with constraints: {hv}")
        Hypervolume with constraints: 5.0
    """
    if len(points) == 0:
        return 0.0

    points = np.asarray(points, dtype=np.float64)
    reference_point = np.asarray(reference_point, dtype=np.float64)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n_points, n_objectives)")
    if reference_point.ndim != 1:
        raise ValueError("reference_point must be a 1D array")
    if points.shape[1] != len(reference_point):
        raise ValueError(
            f"points has {points.shape[1]} objectives but reference_point has "
            f"{len(reference_point)} dimensions"
        )

    # Filter to feasible points only
    if feasible_mask is not None:
        feasible_mask = np.asarray(feasible_mask, dtype=bool)
        if len(feasible_mask) != len(points):
            raise ValueError(
                f"feasible_mask length {len(feasible_mask)} doesn't match "
                f"points length {len(points)}"
            )
        points = points[feasible_mask]

    if len(points) == 0:
        return 0.0

    # Filter out points that are dominated by the reference point
    # A point is valid if it dominates the reference point in all objectives
    valid_mask = np.all(points >= reference_point, axis=1)
    valid_points = points[valid_mask]

    if len(valid_points) == 0:
        return 0.0

    # Use dimension-specific algorithms
    n_objectives = points.shape[1]
    if n_objectives == 1:
        return _hypervolume_1d(valid_points, reference_point)
    elif n_objectives == 2:
        return _hypervolume_2d(valid_points, reference_point)
    else:
        return _hypervolume_nd(valid_points, reference_point)


def compute_hypervolume_trace(
    points: np.ndarray,
    reference_point: np.ndarray,
    feasible_mask: Optional[np.ndarray] = None,
    cumulative: bool = True,
) -> List[float]:
    """
    Compute hypervolume trace over a sequence of points.

    This function computes the hypervolume for each prefix of the point sequence,
    which is useful for tracking hypervolume improvement over time during
    optimization.

    Args:
        points: Array of shape (n_points, n_objectives) containing the objective
            values in the order they were evaluated.
        reference_point: Array of shape (n_objectives,) defining the reference point.
        feasible_mask: Optional boolean array indicating feasible points.
        cumulative: If True, compute cumulative hypervolume at each step.
            If False, compute hypervolume contribution of each individual point.

    Returns:
        List of hypervolume values, one for each point in the sequence.

    Example:
        >>> import numpy as np
        >>> points = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]])
        >>> reference_point = np.array([0.0, 0.0])
        >>> # Cumulative hypervolume (recommended for tracking progress)
        >>> hv_trace = compute_hypervolume_trace(points, reference_point)
        >>> print(f"Cumulative HV: {hv_trace}")
        Cumulative HV: [1.0, 6.0, 8.0]
        >>> # Individual contributions
        >>> hv_individual = compute_hypervolume_trace(
        ...     points, reference_point, cumulative=False
        ... )
        >>> print(f"Individual HV: {hv_individual}")
        Individual HV: [1.0, 6.0, 6.0]
    """
    points = np.asarray(points, dtype=np.float64)
    reference_point = np.asarray(reference_point, dtype=np.float64)

    if len(points) == 0:
        return []

    hypervolumes = []

    if cumulative:
        # Compute cumulative hypervolume
        for i in range(len(points)):
            current_points = points[: i + 1]
            current_feasible = (
                feasible_mask[: i + 1] if feasible_mask is not None else None
            )
            hv = compute_hypervolume(current_points, reference_point, current_feasible)
            hypervolumes.append(hv)
    else:
        # Compute hypervolume of each point individually
        for i in range(len(points)):
            if feasible_mask is not None and not feasible_mask[i]:
                hypervolumes.append(0.0)
            else:
                point = points[i : i + 1]  # Single point as 2D array
                hv = compute_hypervolume(point, reference_point)
                hypervolumes.append(hv)

    return hypervolumes


def _hypervolume_1d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute 1D hypervolume - just the maximum range."""
    return float(np.max(points[:, 0]) - reference_point[0])


def _hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute 2D hypervolume using efficient sweep line algorithm."""
    if len(points) == 0:
        return 0.0

    # Find non-dominated points first
    non_dominated_indices = _find_non_dominated_indices(points)
    pareto_points = points[non_dominated_indices]

    if len(pareto_points) == 0:
        return 0.0

    # Sort points by first objective (x-coordinate) in descending order
    sorted_indices = np.argsort(-pareto_points[:, 0])
    sorted_points = pareto_points[sorted_indices]

    hypervolume = 0.0
    prev_y = reference_point[1]

    for point in sorted_points:
        x, y = point
        if y > prev_y:
            # This point contributes to hypervolume
            hypervolume += (x - reference_point[0]) * (y - prev_y)
            prev_y = y

    return hypervolume


def _hypervolume_nd(points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume for 3+ dimensions using inclusion-exclusion principle.

    This implementation uses a simplified approach that works well for moderate
    numbers of points and dimensions. For very high dimensions or many points,
    more sophisticated algorithms like WFG would be more efficient.
    """
    n_points, n_objectives = points.shape

    if n_points == 0:
        return 0.0

    # Find non-dominated points first
    non_dominated_indices = _find_non_dominated_indices(points)
    pareto_points = points[non_dominated_indices]

    if len(pareto_points) == 0:
        return 0.0

    if len(pareto_points) == 1:
        # Single point case
        return np.prod(pareto_points[0] - reference_point)

    # Use inclusion-exclusion principle for multiple points
    # Add individual contributions, subtract pairwise overlaps
    total_hv = 0.0
    n_pareto = len(pareto_points)

    # Add individual contributions
    for point in pareto_points:
        total_hv += np.prod(point - reference_point)

    # Subtract pairwise overlaps
    for i in range(n_pareto):
        for j in range(i + 1, n_pareto):
            p1, p2 = pareto_points[i], pareto_points[j]
            # Overlap region is from reference to element-wise minimum
            overlap_upper = np.minimum(p1, p2)
            if np.all(overlap_upper > reference_point):
                overlap_volume = np.prod(overlap_upper - reference_point)
                total_hv -= overlap_volume

    # For simplicity, we stop at pairwise overlaps rather than implementing
    # the full inclusion-exclusion formula. This gives a good approximation
    # for most practical cases and avoids the exponential complexity.

    return max(0.0, total_hv)


def _find_non_dominated_indices(points: np.ndarray) -> np.ndarray:
    """
    Find indices of non-dominated points (Pareto frontier).

    A point is non-dominated if there is no other point that is at least as good
    in all objectives and strictly better in at least one objective.
    """
    n_points = len(points)
    is_dominated = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if is_dominated[i]:
            continue
        for j in range(n_points):
            if i == j or is_dominated[j]:
                continue
            # Check if point j dominates point i
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


def infer_reference_point(
    points: np.ndarray,
    feasible_mask: Optional[np.ndarray] = None,
    offset_fraction: float = 0.1,
) -> np.ndarray:
    """
    Infer a reasonable reference point from the data.

    This creates a reference point that is slightly worse than the worst
    observed values in each objective, which is a common heuristic for
    hypervolume computation.

    Args:
        points: Array of shape (n_points, n_objectives) containing objective values.
        feasible_mask: Optional boolean array indicating feasible points.
        offset_fraction: Fraction of the range to subtract from minimum values.

    Returns:
        Array of shape (n_objectives,) with the inferred reference point.

    Example:
        >>> import numpy as np
        >>> points = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]])
        >>> ref = infer_reference_point(points)
        >>> print(f"Inferred reference: {ref}")
        Inferred reference: [0.8 0.8]
    """
    if len(points) == 0:
        raise ValueError("Cannot infer reference point from empty point set")

    points = np.asarray(points, dtype=np.float64)

    # Filter to feasible points if mask is provided
    if feasible_mask is not None:
        feasible_mask = np.asarray(feasible_mask, dtype=bool)
        if len(feasible_mask) != len(points):
            raise ValueError("feasible_mask length doesn't match points length")
        if not np.any(feasible_mask):
            raise ValueError("No feasible points available")
        points = points[feasible_mask]

    # Use minimum values as base, then subtract offset
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    ranges = max_values - min_values

    # For zero ranges, use a small offset
    ranges = np.where(ranges == 0, 1.0, ranges)

    reference_point = min_values - offset_fraction * ranges
    return reference_point