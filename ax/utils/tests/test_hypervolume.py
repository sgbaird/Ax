#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.utils.common.testutils import TestCase
from ax.utils.hypervolume import (
    compute_hypervolume,
    compute_hypervolume_trace,
    infer_reference_point,
)


class TestHypervolume(TestCase):
    def test_compute_hypervolume_1d(self) -> None:
        """Test 1D hypervolume computation."""
        # Single point
        points = np.array([[2.0]])
        ref = np.array([0.0])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 2.0)

        # Multiple points
        points = np.array([[1.0], [3.0], [2.0]])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 3.0)  # Max value minus reference

    def test_compute_hypervolume_2d(self) -> None:
        """Test 2D hypervolume computation with known cases."""
        # Single point
        points = np.array([[2.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 6.0)

        # Two non-dominated points
        points = np.array([[1.0, 3.0], [3.0, 1.0]])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 5.0)

        # Two dominated points (one dominates the other)
        points = np.array([[1.0, 1.0], [2.0, 2.0]])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 4.0)

        # Three points forming L-shape
        points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 6.0)

        # With different reference point
        points = np.array([[2.0, 3.0], [3.0, 2.0]])
        ref = np.array([1.0, 1.0])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 3.0)

    def test_compute_hypervolume_3d(self) -> None:
        """Test 3D hypervolume computation."""
        # Single point
        points = np.array([[2.0, 2.0, 2.0]])
        ref = np.array([0.0, 0.0, 0.0])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 8.0)

        # Two non-dominated points
        points = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 3.0)  # 2 + 2 - 1 overlap

    def test_compute_hypervolume_with_feasibility(self) -> None:
        """Test hypervolume computation with feasible/infeasible points."""
        points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        ref = np.array([0.0, 0.0])
        feasible = np.array([True, False, True])

        hv = compute_hypervolume(points, ref, feasible)
        self.assertEqual(hv, 5.0)  # Should exclude the middle point

        # Compare with all feasible
        hv_all = compute_hypervolume(points, ref)
        self.assertEqual(hv_all, 6.0)

    def test_compute_hypervolume_edge_cases(self) -> None:
        """Test edge cases for hypervolume computation."""
        ref = np.array([0.0, 0.0])

        # Empty points
        points = np.array([]).reshape(0, 2)
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 0.0)

        # All points dominated by reference
        points = np.array([[-1.0, -1.0], [-2.0, -2.0]])
        hv = compute_hypervolume(points, ref)
        self.assertEqual(hv, 0.0)

        # All points infeasible
        points = np.array([[1.0, 1.0], [2.0, 2.0]])
        feasible = np.array([False, False])
        hv = compute_hypervolume(points, ref, feasible)
        self.assertEqual(hv, 0.0)

    def test_compute_hypervolume_trace_cumulative(self) -> None:
        """Test cumulative hypervolume trace."""
        points = np.array([
            [1.0, 1.0],
            [2.0, 3.0],
            [1.5, 1.5],  # This is dominated
            [3.0, 2.0],
        ])
        ref = np.array([0.0, 0.0])

        hv_trace = compute_hypervolume_trace(points, ref, cumulative=True)

        # Expected: [1.0, 6.0, 6.0, 8.0]
        # Point 0: (1,1) -> HV = 1
        # Point 1: (2,3) added -> HV = 6  
        # Point 2: (1.5,1.5) dominated -> HV = 6
        # Point 3: (3,2) added -> HV = 8
        expected = [1.0, 6.0, 6.0, 8.0]
        self.assertEqual(len(hv_trace), len(expected))
        for i, (actual, exp) in enumerate(zip(hv_trace, expected)):
            self.assertAlmostEqual(actual, exp, places=6, msg=f"Index {i}")

    def test_compute_hypervolume_trace_individual(self) -> None:
        """Test individual hypervolume trace."""
        points = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]])
        ref = np.array([0.0, 0.0])

        hv_trace = compute_hypervolume_trace(points, ref, cumulative=False)

        # Each point's individual contribution
        expected = [1.0, 6.0, 6.0]
        self.assertEqual(len(hv_trace), len(expected))
        for i, (actual, exp) in enumerate(zip(hv_trace, expected)):
            self.assertAlmostEqual(actual, exp, places=6, msg=f"Index {i}")

    def test_compute_hypervolume_trace_with_feasibility(self) -> None:
        """Test hypervolume trace with feasible/infeasible points."""
        points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        ref = np.array([0.0, 0.0])
        feasible = np.array([True, False, True])

        # Cumulative trace
        hv_trace_cum = compute_hypervolume_trace(
            points, ref, feasible, cumulative=True
        )
        expected_cum = [3.0, 3.0, 5.0]  # Middle point doesn't contribute
        self.assertEqual(len(hv_trace_cum), len(expected_cum))

        # Individual trace
        hv_trace_ind = compute_hypervolume_trace(
            points, ref, feasible, cumulative=False
        )
        expected_ind = [3.0, 0.0, 3.0]  # Middle point is infeasible
        self.assertEqual(len(hv_trace_ind), len(expected_ind))

    def test_infer_reference_point(self) -> None:
        """Test reference point inference."""
        points = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]])

        # Default offset
        ref = infer_reference_point(points)
        expected = np.array([0.8, 0.8])  # min - 0.1 * range
        np.testing.assert_array_almost_equal(ref, expected, decimal=6)

        # Custom offset
        ref = infer_reference_point(points, offset_fraction=0.2)
        expected = np.array([0.6, 0.6])  # min - 0.2 * range
        np.testing.assert_array_almost_equal(ref, expected, decimal=6)

        # With feasibility
        feasible = np.array([True, False, True])
        ref = infer_reference_point(points, feasible)
        # Only points [1,3] and [3,2] -> min=[1,2], range=[2,1]
        expected = np.array([0.8, 1.9])  # [1-0.1*2, 2-0.1*1]
        np.testing.assert_array_almost_equal(ref, expected, decimal=6)

    def test_infer_reference_point_edge_cases(self) -> None:
        """Test edge cases for reference point inference."""
        # Empty points
        with self.assertRaises(ValueError):
            infer_reference_point(np.array([]).reshape(0, 2))

        # All infeasible
        points = np.array([[1.0, 1.0]])
        feasible = np.array([False])
        with self.assertRaises(ValueError):
            infer_reference_point(points, feasible)

        # Zero range in one dimension
        points = np.array([[1.0, 2.0], [1.0, 3.0]])
        ref = infer_reference_point(points)
        expected = np.array([0.9, 1.9])  # [1-0.1*1, 2-0.1*1]
        np.testing.assert_array_almost_equal(ref, expected, decimal=6)

    def test_input_validation(self) -> None:
        """Test input validation for hypervolume functions."""
        # Wrong dimensions
        with self.assertRaises(ValueError):
            compute_hypervolume(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

        with self.assertRaises(ValueError):
            compute_hypervolume(np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]))

        # Mismatched dimensions
        with self.assertRaises(ValueError):
            compute_hypervolume(np.array([[1.0, 2.0]]), np.array([0.0]))

        # Wrong feasibility mask length
        with self.assertRaises(ValueError):
            compute_hypervolume(
                np.array([[1.0, 2.0]]), np.array([0.0, 0.0]), np.array([True, False])
            )

    def test_consistency_with_ax_examples(self) -> None:
        """Test consistency with examples from Ax test cases."""
        # From ax/service/tests/test_best_point_utils.py
        points = np.array([
            [1.0, 1.0],  # Point 0: becomes reference
            [2.0, 3.0],  # Point 1
            [4.0, 4.0],  # Point 2: will be excluded
            [3.0, 2.0],  # Point 3
        ])
        feasible = np.array([True, True, False, True])
        reference_point = np.array([1.0, 1.0])

        # Test cumulative hypervolume
        cumulative_hvs = []
        for i in range(len(points)):
            current_points = points[: i + 1]
            current_feasible = feasible[: i + 1]
            hv = compute_hypervolume(current_points, reference_point, current_feasible)
            cumulative_hvs.append(hv)

        expected_cumulative = [0.0, 2.0, 2.0, 3.0]
        self.assertEqual(len(cumulative_hvs), len(expected_cumulative))
        for i, (actual, exp) in enumerate(zip(cumulative_hvs, expected_cumulative)):
            self.assertAlmostEqual(actual, exp, places=6, msg=f"Cumulative index {i}")

        # Test individual hypervolume
        individual_hvs = []
        for i in range(len(points)):
            if not feasible[i]:
                individual_hvs.append(0.0)
            else:
                point = points[i : i + 1]
                hv = compute_hypervolume(point, reference_point)
                individual_hvs.append(hv)

        expected_individual = [0.0, 2.0, 0.0, 2.0]
        self.assertEqual(len(individual_hvs), len(expected_individual))
        for i, (actual, exp) in enumerate(zip(individual_hvs, expected_individual)):
            self.assertAlmostEqual(actual, exp, places=6, msg=f"Individual index {i}")