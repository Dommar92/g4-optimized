# Pallet Box Packing Tool
This project provides a tool for optimizing the layout of boxes on a pallet while maximizing area usage and considering specific constraints. Additionally, it offers features for visualization and exporting results to CSV.
## Features
- **Box Dimensions Calculation**: Specify and calculate dimensions for boxes and pallets.
- **Optimization Algorithm**: Efficiently organizes boxes on a pallet to maximize usage of available space while meeting constraints.
- **Center of Mass Calculation**: Determines the center of mass for balanced packing.
- **Visualization**: Provides a visualization of the pallet and box arrangements.
- **Export Results**: Save optimized packing data to a CSV file.

## Requirements
Install the required dependencies before running the project. You can install the packages with:
``` bash
pip install -r requirements.txt
```
### Required Packages:
- `matplotlib`: Used for visualization.

### Attributes
Some of the key attributes used in calculations and optimizations:
- **Pallet dimensions**: `pallet_w`, `pallet_d`, `pallet_h`.
- **Box dimensions**: `box_w`, `box_d`, `box_h`.
- **Visualization-related attributes**: `show_visualization`.

## Visualization
The tool generates a graphical representation of the packing arrangement. This feature is particularly useful for analyzing and verifying the optimization results.

## Usage
``` bash
    python script.py <pallet_w> <pallet_d> <pallet_max_h> <box_w> <box_d> <box_h> [<show_visualization:1|0>]
```
Height of both pallet and box is not used atm. For calculating next layer of boxes input the surface of the last result as pallet size.