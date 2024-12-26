import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Data Paths
ROOT_DIR = Path(__file__).resolve().parent
FOOTPRINTS = ROOT_DIR / "footprints.json"
SOLUTION_DIR = ROOT_DIR / "sols"
DATA_DIR = ROOT_DIR / "data"

"""
# ==============================================================
# file management functions
# ==============================================================
"""


def list_files_in_dir(dir_path=DATA_DIR, full_path=True):
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    if not full_path:
        files = [os.path.basename(file) for file in files]
    return sorted(files, key=lambda s: (len(s), s))


def parse_input_data(input_data, input_type="string"):
    """Parse a knapsack problem from either a file or directly from a string."""

    if input_type == "string":
        lines = input_data.strip().split("\n")
    elif input_type == "file":
        with open(input_data) as file:
            lines = file.readlines()
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Parse problem parameters
    n_items, capacity = map(int, lines[0].split())

    # Parse facilities data
    items_columns = ["value", "weight"]
    items_lines = [x for x in lines if x != "\n"][1:]
    assert len(items_lines) == n_items, (
        f"Expected {n_items} items, got {len(items_lines)}"
    )
    items = pd.DataFrame(
        [x.split() for x in items_lines],
        columns=items_columns,
        dtype=float,
    )

    return {
        "n_items": n_items,
        "capacity": capacity,
        "items": items,
    }


def write_solution(contents, file_name, file_path=SOLUTION_DIR):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), "w") as file:
        file.write(contents)


def calculate_data_footprint(data):
    return data["n_items"] + data["capacity"] + data["items"].sum().sum().item()


def save_footprints_to_disk(dictionary, file_path=FOOTPRINTS):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


def load_footprints_from_disk(file_path=FOOTPRINTS):
    with open(file_path) as file:
        return json.load(file)


def load_solution_from_disk(file_name, file_path=SOLUTION_DIR):
    with open(os.path.join(file_path, file_name)) as file:
        return file.read()


def generate_output(optimal_value, solution_binary, optimized_indicator=0):
    obj = optimal_value
    answer = f"{obj} {optimized_indicator}\n"
    answer += " ".join(map(str, solution_binary))
    return answer


# def generate_dummy_output(sequence, distance_matrix):
#     obj = 999999999.99
#     answer = f"{obj} 0\n"
#     answer += " ".join(map(str, distance_matrix.index.tolist()))
#     return answer


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    return wrapper


"""
# ==============================================================
# Knapsack functions
# ==============================================================
"""

# TODO : MIP pyomo highs sovler
# TODO : BnB sorting by density worth
# TODO : DP


@timeit
def knapsack(items, capacity):
    # Initalize the knapsack table
    k = pd.DataFrame(np.empty((capacity + 1, items.shape[0] + 1), dtype=int))

    # part of the table that will be empty
    smallest_item = int(items.weight.min().item())

    # # knapsack table size
    # table_size = k.shape[0] * k.shape[1]
    # table_size_empty = smallest_item * k.shape[1]
    # table_size_empty / table_size

    for j in tqdm(range(1, k.shape[1])):
        value = items.loc[j - 1, "value"].item()
        weight = items.loc[j - 1, "weight"].item()
        for i in range(smallest_item, k.shape[0]):
            # for i in range(k.shape[0]):
            if weight <= i:
                # the moment the new item fits, I have two options out of which i choose the best:
                # 1. I don't take the new item and do what i did before the new item was introduced
                # i.e. `k.loc[i, j - 1]` - simply put check value on the left of the current cell
                # 2. I take the new item (e.g.`value`) and see what i could do (or even better had been doing) with the remaining weight `k.loc[i - weight, j - 1]`, where :
                # `i - weight` is the remaining capacity after taking the new item
                # `j - 1` is what i had been doing before the new item was introduced - remember the new item i have already picked under option 2. so by checking on the left i am checking the best option there is, as my problem has now been reduced to a subproblem i have already solved.
                # simply put, sum value of the new item + value one column to the left (previous move) few rows up, representing the remaining capacity after selecting the item
                k.loc[i, j] = max(
                    value + k.loc[i - weight, j - 1], k.loc[i, j - 1]
                )
            else:
                # while new item does not fit (weight <= i), worst i can do is
                # exactly what i did before the new item was introduced k.loc[i, j - 1]
                k.loc[i, j] = k.loc[i, j - 1]

    return k


@timeit
def vknapsack(items, capacity):
    n_items = len(items)
    # Use numpy array instead of pandas DataFrame for faster access
    k = np.zeros((capacity + 1, n_items + 1), dtype=np.int32)

    # Pre-compute these values to avoid repeated .loc access and .item() calls
    weights = items["weight"].astype(int).to_numpy()
    values = items["value"].astype(int).to_numpy()
    smallest_item = int(np.min(weights))

    # Use numpy's vectorized operations where possible
    for j in tqdm(range(1, n_items + 1)):
        # Get current item's properties - array indexing is faster than .loc
        value = values[j - 1].item()
        weight = weights[j - 1].item()

        # Vectorize the weight comparison
        valid_weights = np.arange(smallest_item, capacity + 1) >= weight
        remaining_capacity = np.arange(smallest_item, capacity + 1) - weight

        # Where weight doesn't fit, copy previous column
        k[smallest_item:, j] = k[smallest_item:, j - 1]

        # Where weight fits, compute max value
        mask = valid_weights
        k[smallest_item:, j][mask] = np.maximum(
            value + k[remaining_capacity[mask], j - 1],
            k[smallest_item:, j - 1][mask],
        )

    return k


def get_optimal_value(knapsack_table):
    if isinstance(knapsack_table, np.ndarray):
        k = pd.DataFrame(knapsack_table)
    elif isinstance(knapsack_table, pd.DataFrame):
        k = knapsack_table.copy()
    return k.iloc[-1, -1].item()


def get_selected_items(
    items, capacity, knapsack_table, solution_as_binary=False
):
    if isinstance(knapsack_table, np.ndarray):
        k = pd.DataFrame(knapsack_table)
    elif isinstance(knapsack_table, pd.DataFrame):
        k = knapsack_table.copy()

    solution_items = []
    i = capacity
    n_items = len(items)

    for j in range(n_items, 0, -1):
        if k.loc[i, j] != k.loc[i, j - 1]:
            # j-1 is for the index to match the items dataframe - j = 4 means the 4th item in the dataframe, i.e. the 3rd index
            solution_items.append(j - 1)
            i -= items.loc[j - 1, "weight"].item()

    if solution_as_binary:
        # turn the solution into binary mask - pick or not pick
        ic = items.copy()
        ic["selected"] = 0
        ic.loc[solution_items, "selected"] = 1
        solution_items_binary = ic["selected"].tolist()
        print("Selected items binary mask:", solution_items_binary)
        return solution_items_binary
    else:
        solution_items.reverse()
        print("Selected items:", solution_items)
        return solution_items
