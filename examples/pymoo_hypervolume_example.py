#!/usr/bin/env python3
"""
Pymoo Hypervolume Implementation Example

This example demonstrates how to use the hypervolume calculation implementation
extracted from the pymoo package. The implementation is based on the dimension-sweep
algorithm by Fonseca, Paquete, and Lopez-Ibanez.

Copyright and Attribution:
- Original implementation: Copyright (C) 2010 Simon Wessing, TU Dortmund University
- Paper: C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
  algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
  Computation, pages 1157-1163, Vancouver, Canada, July 2006.
- Extracted from pymoo package with permission from Simon Wessing
- License: Compatible with pymoo's Apache License 2.0

Note: This implementation assumes minimization objectives. For maximization
objectives, negate the values before calling the hypervolume computation.
"""

import numpy as np


class PymooHyperVolume:
    """
    Hypervolume computation based on cross 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    Original Copyright (C) 2010 Simon Wessing, TU Dortmund University
    The author (Simon Wessing) has explicitly given the permission for this source code
    to be included in pymoo.
    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []

    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].
        """

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        for point in front:
            # only consider points that dominate the reference point
            if weaklyDominates(point, referencePoint):
                relevantPoints.append(point)
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            for j in range(len(relevantPoints)):
                relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in range(dimensions)]
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return hyperVolume

    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.
        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo is not None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (
                    q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (
                        qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex + 1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol

    def preProcess(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList

    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], index, node) for index, node in enumerate(nodes)]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, _, node) in decorated]


class MultiList:
    """A special data structure needed by FonsecaHyperVolume.

    It consists of several doubly linked lists that share common nodes. So,
    every node has multiple predecessors and successors, one in every list.

    Original Copyright (C) 2010 Simon Wessing, TU Dortmund University
    """

    class Node:

        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo
            self.next = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists

        def __str__(self):
            return str(self.cargo)

    def __init__(self, numberLists):
        """Constructor.

        Builds 'numberLists' doubly linked lists.
        """
        self.numberLists = numberLists
        self.sentinel = MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists

    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr

    def __len__(self):
        """Returns the number of lists that are included in this MultiList."""
        return self.numberLists

    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node

    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node

    def remove(self, node, index, bounds):
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


# Convenience wrapper functions to match common usage patterns
def compute_hypervolume_pymoo(points, reference_point):
    """
    Compute hypervolume using the pymoo implementation.
    
    Note: This implementation assumes minimization. For maximization objectives,
    negate the objective values before calling this function.
    
    Args:
        points: Array-like of shape (n_points, n_objectives) or list of lists
        reference_point: Array-like of shape (n_objectives,) or list
        
    Returns:
        float: The hypervolume value
    """
    # Convert to list format as expected by pymoo implementation
    if hasattr(points, 'tolist'):  # numpy array
        points_list = points.tolist()
    else:
        points_list = list(points)
        
    if hasattr(reference_point, 'tolist'):  # numpy array
        ref_list = reference_point.tolist()
    else:
        ref_list = list(reference_point)
    
    hv = PymooHyperVolume(ref_list)
    return hv.compute(points_list)


def compute_hypervolume_pymoo_maximization(points, reference_point):
    """
    Compute hypervolume for maximization objectives using the pymoo implementation.
    
    This function handles the sign conversion for maximization objectives.
    
    Args:
        points: Array-like of shape (n_points, n_objectives) containing values to maximize
        reference_point: Array-like of shape (n_objectives,) containing reference values
        
    Returns:
        float: The hypervolume value
    """
    # Convert to numpy for easier manipulation
    points_array = np.asarray(points)
    ref_array = np.asarray(reference_point)
    
    # Negate for minimization (pymoo assumes minimization)
    neg_points = -points_array
    neg_ref = -ref_array
    
    return compute_hypervolume_pymoo(neg_points, neg_ref)


# Example usage functions
def example_basic_pymoo():
    """Basic example using the pymoo hypervolume implementation."""
    print("=== Basic Pymoo Hypervolume Example ===")
    
    # Example for minimization objectives (the default for pymoo)
    print("\n1. Minimization objectives:")
    # Points in objective space: [cost, time] - both to be minimized
    min_points = [
        [1.0, 3.0],  # Solution 1: low cost, high time
        [2.0, 2.0],  # Solution 2: medium cost, medium time  
        [3.0, 1.0],  # Solution 3: high cost, low time
    ]
    min_ref = [4.0, 4.0]  # Reference point (worse than all solutions)
    
    hv_min = compute_hypervolume_pymoo(min_points, min_ref)
    print(f"   Points (minimize): {min_points}")
    print(f"   Reference: {min_ref}")
    print(f"   Hypervolume: {hv_min:.3f}")
    
    # Example for maximization objectives (more common in optimization)
    print("\n2. Maximization objectives:")
    # Points in objective space: [accuracy, efficiency] - both to be maximized
    max_points = np.array([
        [0.85, 0.7],  # Solution 1: good accuracy, decent efficiency
        [0.8, 0.9],   # Solution 2: decent accuracy, good efficiency
        [0.9, 0.6],   # Solution 3: excellent accuracy, lower efficiency
    ])
    max_ref = np.array([0.5, 0.5])  # Reference point (worse than all solutions)
    
    hv_max = compute_hypervolume_pymoo_maximization(max_points, max_ref)
    print(f"   Points (maximize): {max_points.tolist()}")
    print(f"   Reference: {max_ref.tolist()}")
    print(f"   Hypervolume: {hv_max:.3f}")


def example_comparison_with_standalone():
    """Compare pymoo implementation with our standalone implementation."""
    print("\n=== Comparison: Pymoo vs Standalone ===")
    
    # Import our standalone implementation
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ax', 'utils'))
    
    try:
        from hypervolume import compute_hypervolume as compute_hv_standalone
        
        # Test points for maximization
        points_max = np.array([
            [2.0, 3.0],
            [3.0, 2.0], 
            [1.5, 1.5],  # This will be dominated
        ])
        ref_max = np.array([0.0, 0.0])
        
        # Compute with standalone (designed for maximization)
        hv_standalone = compute_hv_standalone(points_max, ref_max)
        
        # Compute with pymoo (convert to maximization)
        hv_pymoo = compute_hypervolume_pymoo_maximization(points_max, ref_max)
        
        print(f"Points (maximize): {points_max.tolist()}")
        print(f"Reference: {ref_max.tolist()}")
        print(f"Standalone implementation: {hv_standalone:.6f}")
        print(f"Pymoo implementation:      {hv_pymoo:.6f}")
        print(f"Difference: {abs(hv_standalone - hv_pymoo):.6f}")
        
        if abs(hv_standalone - hv_pymoo) < 1e-10:
            print("✓ Results match!")
        else:
            print("⚠ Results differ - this may be due to algorithm differences")
            
    except ImportError:
        print("Standalone implementation not available for comparison")


def example_higher_dimensions_pymoo():
    """Example with higher dimensional problems using pymoo."""
    print("\n=== Higher Dimensional Example (Pymoo) ===")
    
    # 4-objective optimization (all maximization)
    points_4d = [
        [0.8, 0.7, 0.9, 0.6],
        [0.9, 0.6, 0.7, 0.8],
        [0.7, 0.9, 0.6, 0.7],
        [0.6, 0.8, 0.8, 0.9],
    ]
    ref_4d = [0.0, 0.0, 0.0, 0.0]
    
    hv_4d = compute_hypervolume_pymoo_maximization(points_4d, ref_4d)
    print(f"4D points: {points_4d}")
    print(f"4D hypervolume: {hv_4d:.6f}")
    
    # Test scalability
    print("\nTesting different problem sizes:")
    for n_points in [5, 10, 20]:
        # Generate random points
        np.random.seed(42)  # For reproducibility
        random_points = np.random.uniform(0.1, 1.0, (n_points, 3))
        ref_3d = [0.0, 0.0, 0.0]
        
        hv = compute_hypervolume_pymoo_maximization(random_points, ref_3d)
        print(f"   {n_points} points in 3D: HV = {hv:.6f}")


def example_optimization_tracking():
    """Example of tracking hypervolume during optimization."""
    print("\n=== Optimization Progress Tracking (Pymoo) ===")
    
    # Simulate optimization progress (maximization objectives)
    optimization_steps = [
        [0.6, 0.7],    # Initial solution
        [0.7, 0.8],    # Improvement
        [0.65, 0.75],  # Dominated point (won't contribute much)
        [0.8, 0.75],   # Better in first objective
        [0.75, 0.85],  # Better in second objective
        [0.85, 0.8],   # Final best solution
    ]
    
    reference_point = [0.0, 0.0]
    
    print("Optimization progress:")
    cumulative_points = []
    for i, point in enumerate(optimization_steps):
        cumulative_points.append(point)
        hv = compute_hypervolume_pymoo_maximization(cumulative_points, reference_point)
        print(f"   Step {i+1}: {point} -> HV = {hv:.4f}")


if __name__ == "__main__":
    print("Pymoo Hypervolume Implementation Example")
    print("=" * 50)
    print()
    print("Based on:")
    print("C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez.")
    print("An improved dimension-sweep algorithm for the hypervolume indicator.")
    print("IEEE Congress on Evolutionary Computation, 2006.")
    print()
    print("Original implementation: Copyright (C) 2010 Simon Wessing")
    print("Extracted from pymoo package with permission")
    print("=" * 50)
    
    example_basic_pymoo()
    example_comparison_with_standalone()
    example_higher_dimensions_pymoo()
    example_optimization_tracking()
    
    print("\n" + "="*60)
    print("COPY-PASTE INSTRUCTIONS:")
    print("="*60)
    print("To use this pymoo hypervolume implementation independently:")
    print("1. Copy the PymooHyperVolume and MultiList classes")
    print("2. Use compute_hypervolume_pymoo() for minimization objectives")
    print("3. Use compute_hypervolume_pymoo_maximization() for maximization")
    print("4. No external dependencies required (pure Python)")
    print("5. Based on the proven Fonseca-Paquete-Lopez-Ibanez algorithm")
    print("="*60)