import time
from pathlib import Path

import numpy as np
import pandas as pd
from functions import (
    DATA_DIR,
    calculate_data_footprint,
    calculate_selected_items_binary_mask,
    calculate_total_value_from_knapsack,
    calculate_total_value_from_solution,
    generate_output,
    get_selected_items_from_knapsack,
    greedy_knapsack,
    greedy_knapsack_dataset,
    greedy_knapsack_stochastic,
    knapsack,
    list_files_in_dir,
    load_footprints_from_disk,
    parse_input_data,
    vknapsack,
    write_solution,
)

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
file_name = "ks_10000_0"

# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_items = data["n_items"]
capacity = data["capacity"]
items = data["items"]
print(f"{file_name}: {capacity=:,} | {n_items=} | {n_items * capacity:,}")

# footprint operations
footprints = load_footprints_from_disk()
footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Footprint: {footprints[str(footprint)]} matches {file_name=}")


"""
# ==============================================================
# Heuristic - reduce the search space - removing dominated items
# ==============================================================
"""
# ! use this is for ks_10000_0 - Knapsack Problem 6

n_buckets = 1000  # split items into n_buckets with similar weights
top_items_per_bucket = 2  # select top density items from each bucket

# discard dominated items + force some smaller items in the dataset
items_heuristic = greedy_knapsack_dataset(
    items, n_buckets=n_buckets, top_items_per_bucket=top_items_per_bucket
)

print(f"heuristic items dataset size: {items_heuristic.shape[0]}")

solution = greedy_knapsack_stochastic(
    items_heuristic, capacity, n_iter=10_000, acceptance_rate=0.5
)
solution_binary = calculate_selected_items_binary_mask(n_items, solution)
total_value = calculate_total_value_from_solution(items_heuristic, solution)

# quick validation
assert len(solution_binary) == n_items
min_weight = items.weight.min().item()
total_weight = items.loc[solution, :].weight.sum().item()
leftover = capacity - items.loc[solution, :].weight.sum().item()
print("Capacity:", capacity)
print("Total weight:", total_weight)
print("Leftover capacity:", leftover)
print("Minimum weight:", min_weight)
print("Total Value:", total_value)

# ! best solution : 1099888, achieved through :
# ! greedy_knapsack_dataset n_buckets = 1000, top_items_per_bucket = 2
# ! achieved through : greedy_knapsack_stochastic n_iter = 10_000, acceptance_rate = 0.5

# * WRITE SOLUTION
output_data = generate_output(
    total_value, solution_binary, optimized_indicator=0
)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)

"""
# ==============================================================
# Solve the problem with Greedy Knapsack
# ==============================================================
"""
# ! use this is for ks_400_0 - Knapsack Problem 4

# test greedy_knapsack
solution = greedy_knapsack(items, capacity)
solution_binary = calculate_selected_items_binary_mask(n_items, solution)
total_value = calculate_total_value_from_solution(items, solution)

# test greedy_knapsack_stochastic
solution = greedy_knapsack_stochastic(
    items, capacity, n_iter=10_000, acceptance_rate=0.75
)
solution_binary = calculate_selected_items_binary_mask(n_items, solution)
total_value = calculate_total_value_from_solution(items, solution)

# quick validation
assert len(solution_binary) == n_items
min_weight = items.weight.min().item()
total_weight = items.loc[solution, :].weight.sum().item()
leftover = capacity - items.loc[solution, :].weight.sum().item()
print("Capacity:", capacity)
print("Total weight:", total_weight)
print("Leftover capacity:", leftover)
print("Minimum weight:", min_weight)
print("Total Value:", total_value)

# * WRITE SOLUTION
output_data = generate_output(
    total_value, solution_binary, optimized_indicator=0
)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)


"""
# ==============================================================
# Solve the problem with Dynamic Programming
# ==============================================================
"""
# ! use this is for ks_30_0, ks_50_0, ks_200_0, ks_1000_0

start = time.perf_counter()
# k = knapsack(items, capacity)
k = vknapsack(items, capacity)
end = time.perf_counter()
print(f"Time taken: {end - start:.4f} seconds")

# extract selected items
solution = get_selected_items_from_knapsack(items, capacity, k)
solution_binary = calculate_selected_items_binary_mask(n_items, solution)

# extract optimal value
total_value = calculate_total_value_from_knapsack(k)
total_value = calculate_total_value_from_solution(items, solution)
print("Total value:", total_value)

# quick validation
assert len(solution_binary) == n_items
min_weight = items.weight.min().item()
total_weight = items.loc[solution, :].weight.sum().item()
leftover = capacity - items.loc[solution, :].weight.sum().item()
print("Capacity:", capacity)
print("Total weight:", total_weight)
print("Leftover capacity:", leftover)
print("Minimum weight:", min_weight)
print("Total Value:", total_value)

# * WRITE SOLUTION
output_data = generate_output(
    total_value, solution_binary, optimized_indicator=1
)
print(f"Solution {file_name}:")
print(output_data)
write_solution(output_data, file_name)
