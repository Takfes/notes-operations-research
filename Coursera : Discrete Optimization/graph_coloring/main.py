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
    or_cp_iter_sat,
    or_cp_obj_opt,
    parse_input_data,
    validate_solution,
    write_solution,
)

"""
# ==============================================================
# Ingest test data
# ==============================================================
"""

"""
gc_50_3    Coloring Problem 1 --> mine: 7   7/10 | coursera next target: 6
gc_70_7    Coloring Problem 2 --> mine: 21  3/10 | coursera next target: 20
gc_100_5   Coloring Problem 3 --> mine: 18  7/10 | coursera next target: 16
gc_250_9   Coloring Problem 4 --> mine: 92  7/10 | coursera next target: 78
gc_500_1   Coloring Problem 5 --> mine: 16  10/10 | coursera next target: 16
gc_1000_5  Coloring Problem 6 --> mine: 116 7/10 | coursera next target: 100
"""

# * DEFINE CONSTANTS
# list_files_in_dir(full_path=False)
file_name = "gc_250_9"

# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_nodes = data["n_nodes"]
n_edges = data["n_edges"]
dataset = data["dataset"]
edges = [tuple(x) for x in dataset.values.tolist()]
print(f"{file_name}: {n_edges=:,} | {n_nodes=} | {n_nodes * n_edges:,}")

# footprint operations
footprints = load_footprints_from_disk()
footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Footprint: {footprints[str(footprint)]} matches {file_name=}")

"""
# ==============================================================
# SOLVE PROBLEM WITH OR-TOOLS
# ==============================================================
"""
# ! use this for problems 1,2,3
time_limit = 60
enable_logging = False

# objective optimization
start = time.perf_counter()
solution_dict = or_cp_obj_opt(
    n_nodes, edges, time_limit=time_limit, enable_logging=enable_logging
)
end = time.perf_counter()
obj_value = len(set(solution_dict.values()))
validate_solution(edges, solution_dict)
print(f"or_cp_obj_opt execution time {end - start:.4f} seconds")
print(f"or_cp_obj_opt objective value: {obj_value}")

# constraint satisfaction
start = time.perf_counter()
solution_dict = or_cp_iter_sat(
    n_nodes,
    edges,
    start_colors=92,
    ascending=False,
    time_limit=None,
    enable_logging=False,
)
end = time.perf_counter()
obj_value = len(set(solution_dict.values()))
validate_solution(edges, solution_dict)
print(f"or_cp_iter_sat execution time {end - start:.4f} seconds")
print(f"or_cp_iter_sat objective value: {obj_value}")

# * WRITE SOLUTION
solution = list(solution_dict.values())
output_data = generate_output(obj_value, solution, optimized_indicator=0)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)

"""
# ==============================================================
# SOLVE WITH DSATUR HEURISTIC
# ==============================================================
"""
# ! use this for problems 4,5,6
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
