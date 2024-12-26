import time
from pathlib import Path

import numpy as np
import pandas as pd
from functions import (
    DATA_DIR,
    calculate_data_footprint,
    generate_output,
    get_optimal_value,
    get_selected_items,
    knapsack,
    list_files_in_dir,
    load_footprints_from_disk,
    parse_input_data,
    vknapsack,
    write_solution,
)

# """
# # ==============================================================
# # Ingest mock data
# # ==============================================================
# """

# # * MOCK CASES
# # value - weight
# input_data = """4 7
# 16 2
# 19 3
# 23 4
# 28 5
# """
# # value - weight
# input_data = """3 9
# 5 4
# 6 5
# 3 2
# """
# # https://medium.com/@yeap0022/solving-knapsack-problem-1-0-using-dynamic-programming-in-python-536070efccc9
# # value - weight
# input_data = """3 9
# 2 2
# 4 4
# 7 5
# """

# data = parse_input_data(input_data, input_type="string")
# n_items = data["n_items"]
# capacity = data["capacity"]
# items = data["items"]
# print(f"{capacity=:,} | {n_items=}")


"""
# ==============================================================
# Ingest test data
# ==============================================================
"""

"""
ks_30_0     Knapsack Problem 1  capacity=100,000 | n_items=30 | 3,000,000
ks_50_0     Knapsack Problem 2  capacity=341,045 | n_items=50 | 17,052,250
ks_200_0    Knapsack Problem 3  capacity=100,000 | n_items=200 | 20,000,000
ks_400_0    Knapsack Problem 4  capacity=9,486,367 | n_items=400 | 3,794,546,800
ks_1000_0   Knapsack Problem 5  capacity=100,000 | n_items=1000 | 100,000,000
ks_10000_0  Knapsack Problem 6  capacity=1,000,000 | n_items=10000 | 10,000,000,000
"""

# * DEFINE CONSTANTS
# list_files_in_dir(full_path=False)
file_name = "ks_1000_0"

# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_items = data["n_items"]
capacity = data["capacity"]
items = data["items"]
print(f"{capacity=:,} | {n_items=} | {n_items * capacity:,}")

# footprint operations
footprints = load_footprints_from_disk()
footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Footprint: {footprints[str(footprint)]} matches {file_name=}")

"""
# ==============================================================
# Solve the problem
# ==============================================================
"""

start = time.perf_counter()
# k = knapsack(items, capacity)
k = vknapsack(items, capacity)
end = time.perf_counter()
print(f"Time taken: {end - start:.4f} seconds")

# extract optimal value
optimal_value = get_optimal_value(k)
print("Optimal value:", optimal_value)

# extract selected items
solution = get_selected_items(items, capacity, k, solution_as_binary=False)
solution_binary = get_selected_items(
    items, capacity, k, solution_as_binary=True
)

# quick validation
assert len(solution_binary) == n_items
min_weight = items.weight.min().item()
total_weight = items.loc[solution, :].weight.sum().item()
leftover = capacity - items.loc[solution, :].weight.sum().item()
print("Capacity:", capacity)
print("Total weight:", total_weight)
print("Leftover capacity:", leftover)
print("Minimum weight:", min_weight)

# * WRITE SOLUTION
output_data = generate_output(
    optimal_value, solution_binary, optimized_indicator=1
)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)
