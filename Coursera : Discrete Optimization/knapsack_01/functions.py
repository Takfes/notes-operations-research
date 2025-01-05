import heapq
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
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
def knapsack_mip_pyomo(data):
    # initialize model
    model = pyo.ConcreteModel()
    # define sets
    model.I = pyo.Set(initialize=range(data["n_items"]))
    # define parameters
    model.V = pyo.Param(model.I, initialize=data["items"].value.to_dict())
    model.W = pyo.Param(model.I, initialize=data["items"].weight.to_dict())
    model.C = pyo.Param(initialize=data["capacity"])
    # define variables
    model.x = pyo.Var(model.I, within=pyo.Binary)

    # define constraints
    def weight_constraint_rule(model):
        return sum(model.W[i] * model.x[i] for i in model.I) <= model.C

    model.capacity_constraint = pyo.Constraint(rule=weight_constraint_rule)

    # define objective
    def obj(model):
        return sum(model.V[i] * model.x[i] for i in model.I)

    model.obj = pyo.Objective(rule=obj, sense=pyo.maximize)
    # solve
    solver = pyo.SolverFactory("appsi_highs")
    solver.solve(model)
    # extract solution
    solution_dict = {
        k: np.isclose(v, 1, atol=1e-6).item() * 1
        for k, v in model.x.extract_values().items()
    }
    return model.obj(), solution_dict


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


class Node:
    """
    Intuition around what the Node class represents:
    - The nodes in the priority queue represent "states" in the decision tree, each corresponding to a specific set of decisions about which items to include or exclude so far.
    - "What-if" Scenarios: Each node explores a different hypothetical reality where certain items are included, and others are not.
    - Accumulated Memory: The node remembers the cumulative value, weight, and which items are included in the solution up to this point.
    - Potential Future Outcomes: Each node has an upper bound (calculated using linear relaxation), which estimates the maximum value that can be achieved if we proceed optimally from that point.
    """

    def __init__(self, level, value, weight, bound, included):
        self.level = level  # Level in the decision tree (index of current item)
        self.value = value  # Total value so far
        self.weight = weight  # Total weight so far
        self.bound = bound  # Upper bound (linear relaxation)
        self.included = (
            included  # Items included in this branch (binary decision)
        )

    def __lt__(self, other):
        # For max-heap behavior in heapq (use negative bound)
        return (
            self.bound > other.bound
        )  # ! this is reversed to make it a max heap


@timeit
def knapsack_branch_and_bound(values, weights, capacity):
    # Number of items
    n = len(values)

    # Sort items by value-to-weight ratio
    items = sorted(
        zip(values, weights), key=lambda x: x[0] / x[1], reverse=True
    )

    # Helper to calculate upper bound (linear relaxation)
    def calculate_bound(node):
        if node.weight >= capacity:
            return 0  # Infeasible
        bound = node.value
        total_weight = node.weight
        level = node.level + 1

        # Add items fractionally
        while level < n and total_weight + items[level][1] <= capacity:
            bound += items[level][0]
            total_weight += items[level][1]
            level += 1

        # Add fractional part of the next item if possible
        if level < n:
            bound += (capacity - total_weight) * (
                items[level][0] / items[level][1]
            )

        return bound

    # Priority queue for B&B (max-heap using negative bound)
    pq = []
    root = Node(level=-1, value=0, weight=0, bound=0, included=[])
    root.bound = calculate_bound(root)
    heapq.heappush(pq, root)

    # Track the best solution
    best_value = 0
    best_items = []

    # Branch and Bound algorithm
    while pq:
        # ! what's the purpose of the priority queue?
        # heapq ensures that we always process the node with the highest upper bound (most promising potential value) first.
        # By processing high-bound nodes first, we establish a best solution early. Subsequent nodes with a bound lower than this best solution can be discarded (pruned) immediately.
        # Thus, pruning happens faster because the queue helps us focus on promising nodes early.
        current = heapq.heappop(pq)  # Node with highest bound

        # ! this is where the pruning happens
        # If this node cannot improve the best value, skip it
        # if bound <= best_value, it means the current node cannot improve upon the best solution already found.
        # "Can this path ever surpass my current best?" if not, then there is no point in exploring this path further.
        if current.bound <= best_value:
            continue

        # ! what is the level?
        # this is used as an "index" linking the current node to the items list.
        # it's like a pointer to the current item in the items list - where this node was originated from.
        # every time we pick a node from the priority queue, we increment ITS level by one.
        # remember : the items list is sorted by value-to-weight ratio.
        # thus, it's like knowing where we were left off, and consider items from that level onwards
        level = current.level + 1

        # Generate left (include) and right (exclude) child nodes
        # Left child: Include current item
        if level < n:  # There's an actual item to include
            item_value, item_weight = items[level]
            if current.weight + item_weight <= capacity:
                left = Node(
                    level=level,
                    value=current.value + item_value,
                    weight=current.weight + item_weight,
                    bound=0,
                    included=current.included + [1],
                )
                left.bound = calculate_bound(left)

                # Update best solution if found
                if left.value > best_value:
                    best_value = left.value
                    best_items = left.included

                heapq.heappush(pq, left)

        # Right child: Exclude current item
        right = Node(
            level=level,
            value=current.value,
            weight=current.weight,
            bound=0,
            included=current.included + [0],
        )
        right.bound = calculate_bound(right)

        heapq.heappush(pq, right)

    # Map best_items (sorted order) back to original indices
    original_indices = [
        i
        for i, _ in sorted(
            enumerate(zip(values, weights)),
            key=lambda x: x[1][0] / x[1][1],
            reverse=True,
        )
    ]

    best_solution = [0] * n
    for sorted_idx, bit in enumerate(best_items):
        if bit == 1:
            original_idx = original_indices[sorted_idx]
            best_solution[original_idx] = 1

    return best_value, best_solution
