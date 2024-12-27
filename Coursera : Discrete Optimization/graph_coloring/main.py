import time
from pathlib import Path

import numpy as np
import pandas as pd
from functions import (
    DATA_DIR,
    DSatur,
    calculate_data_footprint,
    generate_output,
    load_footprints_from_disk,
    parse_input_data,
    write_solution,
)

"""
# ==============================================================
# Ingest test data
# ==============================================================
"""

"""
gc_50_3    Coloring Problem 1
gc_70_7    Coloring Problem 2
gc_100_5   Coloring Problem 3
gc_250_9   Coloring Problem 4
gc_500_1   Coloring Problem 5
gc_1000_5  Coloring Problem 6
"""

# * DEFINE CONSTANTS
# list_files_in_dir(full_path=False)
file_name = "gc_1000_5"

# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_nodes = data["n_nodes"]
n_edges = data["n_edges"]
dataset = data["dataset"]
print(f"{file_name}: {n_edges=:,} | {n_nodes=} | {n_nodes * n_edges:,}")

# footprint operations
footprints = load_footprints_from_disk()
footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Footprint: {footprints[str(footprint)]} matches {file_name=}")

# * SOLVE PROBLEM
edges = [tuple(x) for x in dataset.values.tolist()]
start = time.perf_counter()
dsat2 = DSatur(n_nodes, edges)
dsat2.solve(verbose=False)
solution = [x.color for x in dsat2.nodes]
solution = [x - 1 for x in solution]  # 1-indexed to 0-indexed
end = time.perf_counter()
obj_value = dsat2.cost
print(f"DSatur executed in {end - start:.4f} seconds")

# # * WRITE SOLUTION
output_data = generate_output(obj_value, solution, optimized_indicator=0)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)
