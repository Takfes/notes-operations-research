import time
from pathlib import Path

import numpy as np
import pandas as pd
from functions import (
    DATA_DIR,
    footprint_function,
    generate_output,
    list_files_in_dir,
    load_footprints_from_disk,
    parse_input_data,
    write_solution,
)

"""
vrp_16_3_1    Vehicle Routing Problem 1
vrp_26_8_1    Vehicle Routing Problem 2
vrp_51_5_1    Vehicle Routing Problem 3
vrp_101_10_1  Vehicle Routing Problem 4
vrp_200_16_1  Vehicle Routing Problem 5
vrp_421_41_1  Vehicle Routing Problem 6
"""

# * DEFINE CONSTANTS
list_files_in_dir(full_path=False)
file_name = "vrp_16_3_1"

# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_customers = data["n_customers"]
n_vehicles = data["n_vehicles"]
vehicle_capacity = data["vehicle_capacity"]
dataset = data["dataset"]
print(f"{file_name}: {n_customers=:,} | {n_vehicles=} | {vehicle_capacity=:,}")

# footprint operations
footprints = load_footprints_from_disk()
footprint = footprint_function(data)
assert footprints[str(footprint)] == file_name
print(f"Footprint: {footprints[str(footprint)]} matches {file_name=}")


# # * WRITE SOLUTION
# output_data = generate_output(obj_value, solution, optimized_indicator=1)
# print(f"Solution {file_name}:")
# print(output_data)
# write_solution(output_data, file_name)
