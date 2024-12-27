import time
from pathlib import Path

import numpy as np
import pandas as pd
from functions import (
    DATA_DIR,
    calculate_data_footprint,
    generate_output,
    load_footprints_from_disk,
    parse_input_data,
    write_solution,
)

# * DEFINE CONSTANTS
# list_files_in_dir(full_path=False)
file_name = "XXX"

# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
info1 = data["info1"]
info2 = data["info2"]
dataset = data["dataset"]
print(f"{file_name}: {info2=:,} | {info1=} | {info1 * info2:,}")

# footprint operations
footprints = load_footprints_from_disk()
footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Footprint: {footprints[str(footprint)]} matches {file_name=}")


# # * WRITE SOLUTION
# output_data = generate_output(obj_value, solution, optimized_indicator=1)
# print(f"Solution {file_name}:")
# print(output_data)
# write_solution(output_data, file_name)
