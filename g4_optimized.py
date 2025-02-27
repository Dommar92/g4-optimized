"""
Script for Packing Optimization and Visualization

This script provides functionality for optimizing the arrangement of boxes on a pallet.
It allows for box rotation, calculates offsets, and generates visualizations
of the arranged boxes on the pallet grid. The output includes:
- CSV files listing box positions and their orientations.
- Optional visualizations of the grid using Matplotlib.

Usage:
    python script.py <pallet_w> <pallet_d> <pallet_max_h> <box_w> <box_d> <box_h> [<show_visualization:1|0>]

Arguments:
    <pallet_w>, <pallet_d>, <pallet_max_h> : Dimensions of the pallet (width, depth, max height).
    <box_w>, <box_d>, <box_h>             : Dimensions of the box to be packed.
    [<show_visualization>]                : Optional. Set to 1 to show visualizations of the pallet arrangement.

Output:
    - CSV containing box positions, orientations, and dimensions.
    - Visualization of the pallet layout (optional).

Author:
    Dominic Marcinkowski
    Department of Computer Science
    University of Duisburg Essen
    Email: dominic.marcinkowski@uni-due.de

"""

import math
import os
from typing import List, Tuple
import csv

import matplotlib.pyplot as plt
import sys

os.makedirs('output', exist_ok=True)

# Constants for quadrants
BOTTOM_LEFT = 0
TOP_LEFT = 1
TOP_RIGHT = 2
BOTTOM_RIGHT = 3


# Helper Functions
def calculate_box_dimensions(is_rotated: bool, box_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Calculate the dimensions of a box based on its rotation status.

    This function computes and returns the dimensions of a box. If the box
    is rotated, its width and height are swapped; otherwise, the original
    dimensions are returned unchanged. The input dimensions are provided
    as a tuple of three values: width, height, and depth.

    :param is_rotated: A boolean flag indicating whether the box is
        rotated.
    :param box_size: A tuple representing the dimensions of the box in
        the format (width, height, depth).
    :return: A tuple representing the calculated dimensions of the box
        considering its rotation status.
    """
    return (box_size[1], box_size[0], box_size[2]) if is_rotated else box_size


def calculate_offsets(quadrant: int, base_offset: Tuple[float, float], box_dim: Tuple[float, float, float],
                      grid_size: Tuple[int, int], pallet_dim: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Calculates the offsets required for aligning grid-based boxes on a pallet given
    the specified quadrant and dimensions. The function determines the correct
    (absolute) positioning of boxes or elements in a multi-quadrant system based
    on the base offset, box dimensions, grid size, and overall pallet dimensions.

    :param quadrant: Quadrant number (e.g., BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT,
        BOTTOM_RIGHT) used to determine placement logic.
    :param base_offset: Tuple containing the initial base offset in the x- and
        z-axes, respectively.
    :param box_dim: Tuple containing the dimensions of a single box in the format
        (width, depth, height).
    :param grid_size: Tuple specifying the number of boxes in the x- and z-directions
        on the pallet grid.
    :param pallet_dim: Tuple specifying the overall dimensions of the pallet in the
        format (width, depth, height).
    :return: A tuple containing the calculated offsets (x, z) based on the specified
        configuration.
    """
    box_w, box_d, _ = box_dim
    count_x, count_z = grid_size
    offset_x, offset_z = base_offset

    if quadrant == BOTTOM_LEFT:
        pass
    elif quadrant == TOP_LEFT:
        offset_z += pallet_dim[1] - box_d * count_z
    elif quadrant == TOP_RIGHT:
        offset_x += pallet_dim[0] - box_w * count_x
        offset_z += pallet_dim[1] - box_d * count_z
    elif quadrant == BOTTOM_RIGHT:
        offset_x += pallet_dim[0] - box_w * count_x
    return offset_x, offset_z


def maximize_boxes_under_constraint(x: float, y: float, max_sum: float) -> Tuple[int, int]:
    """
    Determine the optimal quantities of two types of boxes that can be combined without exceeding a given maximum
    sum constraint, while maximizing the total number of boxes.

    This function calculates the number of two distinct items that can be selected such that their combined cost
    does not exceed a specified maximum sum. It iterates over possible quantities of the first item (`x`), calculates
    the corresponding maximum number of the second item (`y`), and identifies the combination that maximizes the
    total quantity of both items.

    :param x: The cost of the first type of box.
    :type x: float
    :param y: The cost of the second type of box.
    :type y: float
    :param max_sum: The maximum allowable combined cost of the selected boxes.
    :type max_sum: float
    :return: A tuple containing the optimal quantities of the first and second type of boxes.
    :rtype: Tuple[int, int]
    """
    max_a = math.floor(max_sum / x)
    max_result, best_a, best_b = 0, 0, 0
    for a in range(1, max_a + 1):
        b = math.floor((max_sum - a * x) / y)
        if b > 0 and (a + b > max_result):
            max_result, best_a, best_b = a + b, a, b
    return best_a, best_b


def get_box_center(col: int, row: int, offset: Tuple[float, float], box_dim: Tuple[float, float, float]) -> Tuple[
    float, float]:
    """
    Calculate the center coordinates of a specific box in a grid based on its
    column and row indices, offsets for positioning, and the dimensions of the
    box. This function assumes a regular grid arrangement where each box is
    of uniform size. The center is computed using the offset, grid position,
    and dimensions provided.

    :param col: The column index of the box in the grid.
    :type col: int
    :param row: The row index of the box in the grid.
    :type row: int
    :param offset: Positional offsets in the x and z directions, represented
                   as a tuple (offset_x, offset_z).
    :type offset: Tuple[float, float]
    :param box_dim: Dimensions of the box in the grid as a tuple
                    (box_width, box_depth, box_height).
    :type box_dim: Tuple[float, float, float]
    :return: The x and z center coordinates of the specified box in the grid.
    :rtype: Tuple[float, float]
    """
    offset_x, offset_z = offset
    box_w, box_d, _ = box_dim
    center_x = offset_x + col * box_w + box_w / 2
    center_z = offset_z + row * box_d + box_d / 2
    return center_x, center_z


def calculate_center_of_mass(grids: List[Tuple[bool, int, int]], box_dim: Tuple[float, float, float],
                             pallet_dim: Tuple[float, float, float], offsets: List[Tuple[float, float]]) -> Tuple[
    float, float]:
    """
    Calculate the center of mass of boxes arranged on a pallet based on their
    positional arrangement and dimensions. The function accounts for boxes that
    may be rotated and multiple offset positions within grids on the pallet. Each
    grid is processed for its contributing center of mass using the respective
    offsets and box counts in specified directions.

    :param grids: The arrangement of boxes defined as a list of tuples, where each
                  tuple contains:
                  - bool: Rotation status of the boxes.
                  - int: Number of boxes along the x-axis (horizontal count).
                  - int: Number of boxes along the z-axis (depth count).
    :param box_dim: The dimensions of a single box defined as a tuple
                    (width, depth, height).
    :param pallet_dim: The dimensions of the pallet defined as a tuple
                       (width, depth, height).
    :param offsets: A list of tuples representing offsets applied to each grid,
                    where each tuple contains the offset values (x, z) for the grid
                    position on the pallet.
    :return: A tuple containing the calculated center of mass coordinates
             (center_x, center_z) on the pallet. Defaults to (0, 0) if no boxes
             are present on the pallet.
    """
    total_x, total_z, total_boxes = 0.0, 0.0, 0
    pallet_width, pallet_depth, _ = pallet_dim

    for i, (is_rotated, count_x, count_z) in enumerate(grids):
        box_width, box_depth, _ = calculate_box_dimensions(is_rotated, box_dim)
        offset = offsets[i]
        o_x, o_z = offset
        # Starting position
        start_x, start_z = o_x, o_z  # bottom left
        if i == 1:  # top left
            start_x, start_z = o_x, pallet_depth - box_depth * count_z + o_z
        elif i == 2:  # top right
            start_x, start_z = pallet_width - box_width * count_x + o_x, pallet_depth - box_depth * count_z + o_z
        elif i == 3:  # bottom right
            start_x, start_z = pallet_width - box_width * count_x + o_x, o_z

        for row in range(count_z):
            for col in range(count_x):
                # Calculate each box's center
                box_center_x = start_x + col * box_width + box_width / 2
                box_center_z = start_z + row * box_depth + box_depth / 2

                total_x += box_center_x
                total_z += box_center_z
                total_boxes += 1

    # Compute average center of mass
    if total_boxes > 0:
        center_x = total_x / total_boxes
        center_z = total_z / total_boxes
        return center_x, center_z

    return 0, 0


def save_boxes_to_csv(filename: str, grid: List, box_dim: Tuple[float, float, float],
                      pallet_dim: Tuple[float, float, float]) -> None:
    """
    Save box placement information into a CSV file. This function processes the provided grid
    to map positions of all boxes based on their dimensions, orientation, and calculated offsets
    within the pallet dimensions. The resulting organized data is then saved to a CSV file,
    facilitating further analysis or usage in other systems.

    :param filename: The name of the CSV file to be created or overwritten.
    :type filename: str
    :param grid: A data structure containing the number of boxes, grid orientation, counts, and
        offsets. This is often generated by fitting algorithms to fit boxes into the pallet dimensions.
    :type grid: List
    :param box_dim: A tuple containing the dimensions of the box as width, depth, and height.
    :type box_dim: Tuple[float, float, float]
    :param pallet_dim: A tuple containing the dimensions of the pallet as width, depth, and height.
    :type pallet_dim: Tuple[float, float, float]
    :return: No return value. The output is saved directly as a CSV file.
    :rtype: None
    """
    box_count, grids, offsets, _, _ = grid
    pallet_width, pallet_depth, _ = pallet_dim
    boxes = []

    for i, (is_rotated, count_x, count_z) in enumerate(grids):
        offset_x, offset_z = calculate_offsets(i, offsets[i], calculate_box_dimensions(is_rotated, box_dim),
                                               (count_x, count_z), pallet_dim)

        for row in range(count_z):
            for col in range(count_x):
                center_x = offset_x + col * box_dim[0] + box_dim[0] / 2
                center_z = offset_z + row * box_dim[1] + box_dim[1] / 2
                boxes.append([center_x, center_z, int(is_rotated)])

    # Sort boxes by Z (row) and then X (column)
    boxes.sort(key=lambda b: (b[1], b[0]))

    # Write to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Box No', 'Center X (cm)', 'Center Z (cm)', 'Rotated (1=Yes, 0=No)'])

        for index, (x, z, rotated) in enumerate(boxes, start=1):
            writer.writerow([index, x, z, rotated])


def visualize(grid, box_dim, pallet_dim, usage_percentage=0.0):
    """
    Visualizes the grid configuration of boxes on a pallet, representing their placement,
    rotations, and center of mass. This function uses matplotlib to generate a 2D
    representation of the pallet, with each section color-coded and labeled with details
    such as area usage and deviation of the center of mass from the pallet's geometric center.

    :param grid: Contains information about the box arrangement including total box count,
        grid data, offsets, bounding rectangles, center of mass, and center deviation.
        It is structured as a tuple with detailed configurations required to plot the boxes
        accurately on the pallet.
    :type grid: tuple
    :param box_dim: A tuple specifying the width and depth dimensions of a single box in
        centimeters.
    :type box_dim: tuple[float, float]
    :param pallet_dim: A tuple specifying the width, depth, and height of the pallet in
        centimeters.
    :type pallet_dim: tuple[float, float, float]
    :param usage_percentage: The percentage of the pallet's area occupied by the boxes,
        used in the visualization title. Defaults to 0.0 if not provided.
    :type usage_percentage: float, optional
    :return: None. The function displays the visualization using matplotlib.
    :rtype: None
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    box_count, grids, offsets, bounding_rects, (center_of_mass, center_deviation) = grid

    # Extract config
    colors = ['red', 'blue', 'green', 'orange']

    # Draw the pallet
    pallet_width, pallet_depth, _ = pallet_dim
    ax.add_patch(plt.Rectangle((0, 0), pallet_width, pallet_depth, edgecolor='black', fill=False, linewidth=2))

    for i in range(len(grids)):
        grid = grids[i]
        is_rotated = grid[0]
        count_x = grid[1]
        count_z = grid[2]
        offset = offsets[i]
        o_x = offset[0]
        o_z = offset[1]

        # Determine orientation
        box_width, box_depth = (box_dim[1], box_dim[0]) if is_rotated else (box_dim[0], box_dim[1])

        # Starting position
        start_x, start_y = o_x, o_z  # bottom left
        if i == 1:  # top left
            start_x, start_y = o_x, pallet_depth - box_depth * count_z + o_z
        elif i == 2:  # top right
            start_x, start_y = pallet_width - box_width * count_x + o_x, pallet_depth - box_depth * count_z + o_z
        elif i == 3:  # bottom right
            start_x, start_y = pallet_width - box_width * count_x + o_x, o_z

        # Draw boxes
        for row in range(count_z):
            for col in range(count_x):
                x = start_x + col * box_width
                y = start_y + row * box_depth
                ax.add_patch(plt.Rectangle(
                    (x, y), box_width, box_depth,
                    edgecolor='black', facecolor=colors[i], alpha=0.6
                ))

    # Plot center of mass
    center_x, center_z = center_of_mass
    deviation_x, deviation_z = center_deviation
    ax.plot(center_x, center_z, 'ro', label=f'Center of Mass ({center_x:.2f}, {center_z:.2f})')
    ax.legend()

    # Set axes properties
    plt.title(
        f"{box_count} boxes {box_dim[0]} x {box_dim[1]} cm in a pallet {pallet_width} x {pallet_depth} - Area usage: {usage_percentage:.2f}%\n"
        f"Deviation from pallet center: Δx={deviation_x:.2f} cm, Δz={deviation_z:.2f} cm")
    plt.xlim(0, pallet_width)
    plt.ylim(0, pallet_depth)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel("Width (cm)")
    plt.ylabel("Height (cm)")
    plt.grid(visible=True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()


def find_grids(box_dim: tuple[int, int, int], pallet_dim: tuple[int, int, int]) -> list[
    tuple[int, list[list[int]], list[tuple[int, int]], list[tuple[int, int, int, int]]]]:
    """
    Finds all possible configurations of grids that fit within a pallet with the given dimensions while
    accounting for box rotations and constraints. The method evaluates each potential rotation scheme
    and calculates the number of boxes that can fit in the pallet for a given set of rotations. It also
    ensures that no overlaps occur between rectangles representing box placements.

    :param box_dim: A tuple of dimensions (width, depth, height) of a box for which grids are calculated.
    :param pallet_dim: A tuple of dimensions (width, depth, height) of the pallet where boxes will be arranged.
    :return: A list of tuples, each containing the following:
        - The total number of boxes that can fit into the pallet.
        - A list of grids specifying box arrangements for each quadrant.
        - A list of offsets for the box placements.
        - A list of rectangular coordinates for each quadrant derived from box and pallet dimensions.
    :rtype: list[tuple[int, list[list[int]], list[tuple[int, int]], list[tuple[int, int, int, int]]]]
    """
    b_w, b_d, _ = box_dim
    p_w, p_d, _ = pallet_dim

    rotations = [
        [0, 0, 0, 0],
    ]
    if not box_dim[0] == box_dim[1]:
        rotations = [
            [0, 0, 0, 0],
            [0, 0, 1, 1],  # eq to 1,1,0,0
            [0, 1, 1, 0],  # eq to 1,0,1,0
            [0, 1, 0, 1],  # eq to 1,0,0,1
            [1, 1, 1, 1],
            [1, 0, 0, 0]  # eq to 0,1,0,0; 0,0,1,0; 0,0,0,1
        ]

    results = []

    for rotation in rotations:
        c0_w, c0_d = (b_w, b_d) if not rotation[0] else (b_d, b_w)
        c1_w, c1_d = (b_w, b_d) if not rotation[1] else (b_d, b_w)
        c2_w, c2_d = (b_w, b_d) if not rotation[2] else (b_d, b_w)
        c3_w, c3_d = (b_w, b_d) if not rotation[3] else (b_d, b_w)

        c0_x, c3_x = maximize_boxes_under_constraint(c0_w, c3_w, p_w)
        c0_z, c1_z = maximize_boxes_under_constraint(c0_d, c1_d, p_d)
        c1_x, c2_x = maximize_boxes_under_constraint(c1_w, c2_w, p_w)
        c2_z, c3_z = maximize_boxes_under_constraint(c2_d, c3_d, p_d)
        grids = [
            [rotation[0], c0_x, c0_z],
            [rotation[1], c1_x, c1_z],
            [rotation[2], c2_x, c2_z],
            [rotation[3], c3_x, c3_z]
        ]
        boxes = c0_x * c0_z + c1_x * c1_z + c2_x * c2_z + c3_x * c3_z

        # Calculate rectangle positions and check for overlaps
        rects = [
            (0, 0, c0_x * c0_w, c0_z * c0_d),
            (0, pallet_dim[1] - c1_z * c1_d, c1_x * c1_w, c1_z * c1_d),
            (pallet_dim[0] - c2_x * c2_w, pallet_dim[1] - c2_z * c2_d, c2_x * c2_w, c2_z * c2_d),
            (pallet_dim[0] - c3_x * c3_w, 0, c3_x * c3_w, c3_z * c3_d)
        ]
        overlap_found = False

        # Check overlaps between opposite corners
        if not (rects[0][0] + rects[0][2] <= rects[2][0] or rects[2][0] + rects[2][2] <= rects[0][0] or
                rects[0][1] + rects[0][3] <= rects[2][1] or rects[2][1] + rects[2][3] <= rects[0][1]):
            overlap_found = True
        if not (rects[1][0] + rects[1][2] <= rects[3][0] or rects[3][0] + rects[3][2] <= rects[1][0] or
                rects[1][1] + rects[1][3] <= rects[3][1] or rects[3][1] + rects[3][3] <= rects[1][1]):
            overlap_found = True
        if not overlap_found:
            offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]
            results.append((boxes, grids, offsets, rects))
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def pack_grid(grid, box_dim, pallet_dim):
    """
    Packs a grid of boxes into a pallet, calculating the offsets for grid placement,
    bounding rectangles, and center of mass deviations for optimal alignment.

    This function takes the grid configuration, box dimensions, and pallet dimensions,
    and determines how to arrange the grids on the pallet while factoring in collisions
    and alignment considerations. The function returns the adjusted grid offsets, bounding
    rectangles for each grid, and center of mass deviation from the pallet's center.

    :param grid: A tuple containing the number of boxes, grid configuration, and additional
        grid parameters. The grid configuration defines the rotation, count in x and z
        dimensions for each grid segment placed on the pallet.
    :type grid: Tuple[int, List[Tuple[bool, int, int]], Any, Any]
    :param box_dim: Dimensions of a single box in the format (width, depth, height).
    :type box_dim: Tuple[float, float, float]
    :param pallet_dim: Dimensions of the pallet in the format (width, depth, height).
    :type pallet_dim: Tuple[float, float, float]
    :return: Returns a list containing the following:
        - Total number of boxes.
        - Updated grid configuration with added offsets.
        - Calculated offsets (x, z) for each grid placement.
        - Bounding rectangles for all grid segments.
        - Center of mass coordinates and deviation from the pallet's center.
    :rtype: List[Any]
    """
    # returns grids with additionally offset-array for each corner
    box_count, grids, _, _ = grid
    pal_w, pal_d, _ = pallet_dim
    b_w, b_d, _ = box_dim

    rotations = [g[0] for g in grids]

    bounding_rects = []
    for i, (is_rotated, count_x, count_z) in enumerate(grids):
        box_w, box_d = (b_d, b_w) if is_rotated else (b_w, b_d)
        rect_width = count_x * box_w
        rect_depth = count_z * box_d

        # Determine the origin based on corner positions
        if i == 0:  # Bottom left
            origin_x, origin_z = 0, 0
        elif i == 1:  # Top left
            origin_x, origin_z = 0, pal_d - rect_depth
        elif i == 2:  # Top right
            origin_x, origin_z = pal_w - rect_width, pal_d - rect_depth
        elif i == 3:  # Bottom right
            origin_x, origin_z = pal_w - rect_width, 0
        else:
            origin_x, origin_z = 0, 0  # Default, should not occur

        bounding_rects.append((origin_x, origin_z, rect_width, rect_depth))

    offsets: List[Tuple[float, float]] = []
    for i, rect in enumerate(bounding_rects):
        origin_x, origin_z, rect_width, rect_depth = rect
        min_offset_x, min_offset_z = pal_w, pal_d  # Initialize with max pallet dimensions

        for j, other_rect in enumerate(bounding_rects):
            if i == j:
                continue
            other_x, other_z, other_width, other_depth = other_rect

            # Check for collisions in the x-direction
            if origin_z < other_z + other_depth and origin_z + rect_depth > other_z:  # Overlapping z-axis
                if origin_x + rect_width <= other_x:  # Can push right
                    min_offset_x = min(min_offset_x, other_x - (origin_x + rect_width))
                elif other_x + other_width <= origin_x:  # Can push left
                    min_offset_x = min(min_offset_x, origin_x - (other_x + other_width))

            # Check for collisions in the z-direction
            if origin_x < other_x + other_width and origin_x + rect_width > other_x:  # Overlapping x-axis
                if origin_z + rect_depth <= other_z:  # Can push up
                    min_offset_z = min(min_offset_z, other_z - (origin_z + rect_depth))
                elif other_z + other_depth <= origin_z:  # Can push down
                    min_offset_z = min(min_offset_z, origin_z - (other_z + other_depth))


        # Apply the calculated offsets (positive or negative movement)
        offset_x = -min_offset_x if i in (2, 3) else min_offset_x  # Right corners move left when needed
        offset_z = -min_offset_z if i in (1, 2) else min_offset_z  # Top corners move down when needed

        if i > 0:
            prevOffset = offsets[i-1]
            p_o_x = prevOffset[0]
            p_o_z = prevOffset[1]
            if i%2 == 0:
                if abs(offset_x) > abs(p_o_x):
                    offset_x += p_o_x
            else:
                if abs(offset_z) > abs(p_o_z):
                    offset_z += p_o_z

        offsets.append((offset_x, offset_z))

    # Adjust offsets with opposing directions and same absolute value
    for i in range(len(offsets)):
        for j in range(i + 1, len(offsets)):
            x_i, z_i = offsets[i]
            x_j, z_j = offsets[j]

            # Check opposing directional offsets with matching absolute values
            if x_i != 0 and x_i == -x_j and abs(x_i) == abs(x_j):
                offsets[i] = (x_i / 2, z_i)
                offsets[j] = (x_j / 2, z_j)
            if z_i != 0 and z_i == -z_j and abs(z_i) == abs(z_j):
                offsets[i] = (x_i, z_i / 2)
                offsets[j] = (x_j, z_j / 2)
                
    # collision check
    for i, rect in enumerate(bounding_rects):
        x, z, width, depth = rect
        o_x, o_z = offsets[i]
        for j, other_rect in enumerate(bounding_rects):
            if i == j:
                continue
            other_x, other_z, other_width, other_depth = other_rect
            other_o_x, other_o_z = offsets[j]

            # Apply offsets to check adjusted bounding box positions
            adjusted_x = x + o_x
            adjusted_z = z + o_z
            adjusted_other_x = other_x + other_o_x
            adjusted_other_z = other_z + other_o_z

            # Check for overlap
            if not (adjusted_x + width <= adjusted_other_x or
                    adjusted_other_x + other_width <= adjusted_x or
                    adjusted_z + depth <= adjusted_other_z or
                    adjusted_other_z + other_depth <= adjusted_z):
                print(f"Collision detected between box {i} and box {j}")

    # Compute center of mass
    center_x, center_z = calculate_center_of_mass(grids, box_dim, pallet_dim, offsets)
    # Calculate pallet's center point
    pallet_center_x, pallet_center_z = pallet_dim[0] / 2, pallet_dim[1] / 2
    # Calculate deviations
    deviation_x = center_x - pallet_center_x
    deviation_z = center_z - pallet_center_z
    center_of_mass = (center_x, center_z)
    center_deviation = (deviation_x, deviation_z)
    return [box_count, grids, offsets, bounding_rects, (center_of_mass, center_deviation)]


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "Usage: script.py <pallet_w> <pallet_d> <pallet_max_h> <box_w> <box_d> <box_h> <optional:show_visualization:1>")
        exit(0)
    else:
        pallet_w, pallet_d, pallet_h = map(float, sys.argv[1:4])
        box_w, box_d, box_h = map(float, sys.argv[4:7])
        show_visualization = int(sys.argv[7]) if len(sys.argv) > 7 else 0
        if show_visualization:
            print("### will show vizualization")
        boxes = [(max(box_w, box_d), min(box_w, box_d), box_h)]
        pallet_size = (pallet_w, pallet_d, pallet_h)

    for box_size in boxes:
        print(f"### packing box size {box_size}")
        print(f"### pallet size {pallet_size}")
        print('finding grids...')
        p_size = [*pallet_size]
        grids = find_grids(box_size, pallet_size)
        maximum_box_count = grids[0][0]

        # discarding grids that does not fulfill the maximum number of boxes (-2).
        grids = [grid for grid in grids if grid[0] >= maximum_box_count - 2]
        print('packing grids...')
        grids_with_offsets = [pack_grid(grid, box_size, pallet_size) for grid in grids]
        # [box_count, grids, offsets, bounding_rects, (center_of_mass, center_deviation)]
        grids_with_offsets.sort(key=lambda x: (-x[0], math.sqrt(pow(x[4][1][0], 2) + pow(x[4][1][1], 2))))
        for grid in grids_with_offsets:
            # Calculate the area usage
            pallet_area = pallet_size[0] * pallet_size[1]
            bounding_area_sum = sum(w * h for _, _, w, h in grid[3])
            area_usage_percentage = round((bounding_area_sum / pallet_area) * 100, 1)
            if show_visualization:
                visualize(grid, box_size, pallet_size, usage_percentage=area_usage_percentage)
            output_file = f"output/boxes_{grid[0]}_{area_usage_percentage}_pal{int(pallet_w)}x{int(pallet_d)}x{int(pallet_h)}_box{int(box_w)}x{int(box_d)}x{int(box_h)}.csv"
            save_boxes_to_csv(output_file, grid, box_size, pallet_size)
            print(f">> {output_file}")
