import json
import os
import random
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


def generate_dummy_output(sequence, distance_matrix):
    pass


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

# TODO : MIP pyomo highs sovler - warm start based on heuristics
# TODO : BnB sorting by density worth
# TODO : Genetic Algorithms
# TODO : implemented heuristics could be enhanced based on Tabu search logic


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


def calculate_total_value_from_knapsack(knapsack_table):
    if isinstance(knapsack_table, np.ndarray):
        k = pd.DataFrame(knapsack_table)
    elif isinstance(knapsack_table, pd.DataFrame):
        k = knapsack_table.copy()
    return k.iloc[-1, -1].item()


def get_selected_items_from_knapsack(items, capacity, knapsack_table):
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
    solution_items.reverse()
    print("Selected items:", solution_items)
    return solution_items


def calculate_selected_items_binary_mask(n_items, solution_items):
    # turn the solution into binary mask - pick or not pick
    solution_items_binary = [
        0 if i not in solution_items else 1 for i in range(n_items)
    ]
    print("Selected items binary mask:", solution_items_binary)
    return solution_items_binary


def calculate_total_value_from_solution(items, solution):
    return int(items.loc[solution, :].value.sum().item())


def greedy_knapsack(items, capacity):
    # sort items by density
    ic = items.copy()
    ic["density"] = ic.value / ic.weight
    ic = ic.sort_values(by="density", ascending=False)

    solution = []
    total_weight = 0
    free_space = capacity - total_weight

    for idx, row in ic.iterrows():
        if row.weight <= free_space:
            solution.append(idx)
            total_weight += row.weight
            free_space = capacity - total_weight
        else:
            continue

    return solution


@timeit
def greedy_knapsack_stochastic(
    items, capacity, n_iter=100, acceptance_rate=0.5
):
    # sort items by density
    ic = items.copy()
    ic["density"] = ic.value / ic.weight
    ic = ic.sort_values(by="density", ascending=False)

    # track the best deterministic solution
    best_solution = greedy_knapsack(items, capacity)
    best_solution_value = calculate_total_value_from_solution(
        items, best_solution
    )

    # run the greedy algorithm n_iter times infusing randomness
    while n_iter > 0:
        solution = []
        total_weight = 0
        free_space = capacity - total_weight

        for idx, row in ic.iterrows():
            if row.weight <= free_space and np.random.rand() > acceptance_rate:
                solution.append(idx)
                total_weight += row.weight
                free_space = capacity - total_weight
            else:
                continue

        current_solution_value = calculate_total_value_from_solution(
            items, solution
        )

        if current_solution_value > best_solution_value:
            best_solution = solution
            best_solution_value = current_solution_value

        n_iter -= 1

    return best_solution


def greedy_knapsack_dataset(items, n_buckets, top_items_per_bucket):
    """this is based on the observation that there are items that are dominated by others in the dataset. that is several items have almost the same weight, however, one has a higher value. in such a case, the lower value item is dominated by the higher value item. so we can remove the lower value item from the dataset and still get the same value. this is a heuristic to reduce the search space. the idea is to remove dominated items from the dataset. the heuristic is based on the density of the items. the density is the value divided by the weight. the heuristic is as follows:"""
    ic = items.copy()
    # calculate density
    ic["density"] = ic.value / ic.weight
    ic = ic.sort_values(by="weight", ascending=False)
    ic = ic.rename_axis("item_id").reset_index()
    # create weight buckets
    ic["bucket"] = pd.qcut(ic.index, n_buckets, labels=False)
    # rank top items in each bucket
    ic["top_in_bucket"] = (
        ic.groupby("bucket")["density"]
        .transform(lambda x: x.rank(ascending=False) <= top_items_per_bucket)
        .astype(int)
    )
    # select top items per bucket
    items_dom = (
        ic.query("top_in_bucket == 1")
        .set_index("item_id")[["value", "weight"]]
        .copy()
    )
    # select small items (irrespective of their value) that were excluded
    items_dom_min_weight = items_dom.weight.min().item()
    items_small = items.loc[items.weight < items_dom_min_weight, :]
    items_heuristic = pd.concat([items_dom, items_small], axis=0).sort_values(
        by="weight"
    )
    return items_heuristic
