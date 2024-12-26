import time
from pathlib import Path

import numpy as np
import pandas as pd
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
file_name = "ks_4_0"


# * LOAD DATA
input_data = Path(DATA_DIR) / file_name
data = parse_input_data(input_data, input_type="file")
n_items = data["n_items"]
capacity = data["capacity"]
items = data["items"]


# * TEST CASES
# value - weight
input_data = """4 7
16 2
19 3
23 4
28 5
"""
# value - weight
input_data = """3 9
5 4
6 5
3 2
"""

data = parse_input_data(input_data, input_type="string")
n_items = data["n_items"]
capacity = data["capacity"]
items = data["items"]

k = pd.DataFrame(np.empty((capacity + 1, n_items + 1), dtype=int))

for j in range(1, k.shape[1]):
    value = items.loc[j - 1, "value"].item()
    weight = items.loc[j - 1, "weight"].item()
    for i in range(k.shape[0]):
        if weight <= i:
            # the moment the new item fits, I have two options out of which i choose the best:
            # 1. I don't take the new item and do what i did before the new item was introduced
            # i.e. `k.loc[i, j - 1]` - simply put check value on the left of the current cell
            # 2. I take the new item (e.g.`value`) and see what i could do (or even better had been doing) with the remaining weight `k.loc[i - weight, j - 1]`, where :
            # `i - weight` is the remaining capacity after taking the new item
            # `j - 1` is what i had been doing before the new item was introduced - remember the new item i have already picked under option 2. so by checking on the left i am checking the best option there is, as my problem has now been reduced to a subproblem i have already solved.
            # simply put, sum value of the new item + value one column to the left (previous move) few rows up, representing the remaining capacity after selecting the item
            k.loc[i, j] = max(value + k.loc[i - weight, j - 1], k.loc[i, j - 1])
        else:
            # while new item does not fit (weight <= i), worst i can do is
            # exactly what i did before the new item was introduced k.loc[i, j - 1]
            k.loc[i, j] = k.loc[i, j - 1]

# extract optimal value
optimal_value = k.iloc[-1, -1].item()

# trace back solution
solution_items = []
i = capacity
for j in range(n_items, 0, -1):
    if k.loc[i, j] != k.loc[i, j - 1]:
        # j-1 is for the index to match the items dataframe - j = 4 means the 4th item in the dataframe, i.e. the 3rd index
        solution_items.append(j - 1)
        i -= items.loc[j - 1, "weight"].item()
solution_items.reverse()
print("Selected items:", solution_items)


# * WRITE SOLUTION
# assert len(sequence[:-1]) == n_locations, "Mismatch in sequence length"
# output_data = generate_output(sequence, dm)
print(f"Solution {file_name}:")
# print(output_data)
# write_solution(output_data, file_name)
