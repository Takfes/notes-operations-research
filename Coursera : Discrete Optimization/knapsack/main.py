import time
from pathlib import Path

from functions import (
    DATA_DIR,
    list_files_in_dir,
    parse_input_data,
    write_solution,
)

"""
ks_30_0     Knapsack Problem 1
ks_50_0     Knapsack Problem 2
ks_200_0    Knapsack Problem 3
ks_400_0    Knapsack Problem 4
ks_1000_0   Knapsack Problem 5
ks_10000_0  Knapsack Problem 6
"""

# * DEFINE CONSTANTS
list_files_in_dir(full_path=False)
file_name = "ks_30_0"


# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_items = data["n_items"]
capacity = data["capacity"]
items = data["items"]


# * WRITE SOLUTION
# assert len(sequence[:-1]) == n_locations, "Mismatch in sequence length"
# output_data = generate_output(sequence, dm)
print(f"Solution {file_name}:")
# print(output_data)
# write_solution(output_data, file_name)
