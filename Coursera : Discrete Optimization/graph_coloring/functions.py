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


# TODO : Adjust this to problem specific data
def parse_input_data(input_data, input_type="string"):
    """Parse a problem data from either a file or directly from a string."""

    if input_type == "string":
        lines = input_data.strip().split("\n")
    elif input_type == "file":
        with open(input_data) as file:
            lines = file.readlines()
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Parse problem parameters
    n_nodes, n_edges = map(int, lines[0].split())

    # Parse problem data
    data_columns = ["e1", "e2"]
    lines = [x for x in lines if x != "\n"][1:]
    assert len(lines) == n_edges, f"Expected {n_edges} items, got {len(lines)}"
    edges = pd.DataFrame(
        [list(map(int, x.split())) for x in lines],
        columns=data_columns,
        dtype=int,
    )

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "edges": edges,
    }


# TODO : Adjust this to problem specific data
def calculate_data_footprint(data):
    return data["n_nodes"] + data["n_edges"] + data["edges"].sum().sum().item()


# TODO : Adjust this to problem specific data
def generate_output(obj_value, solution, optimized_indicator=0):
    obj = obj_value
    answer = f"{obj} {optimized_indicator}\n"
    answer += " ".join(map(str, solution))
    return answer


def generate_dummy_output():
    pass


def list_files_in_dir(dir_path=DATA_DIR, full_path=True):
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    if not full_path:
        files = [os.path.basename(file) for file in files]
    return sorted(files, key=lambda s: (len(s), s))


def write_solution(contents, file_name, file_path=SOLUTION_DIR):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), "w") as file:
        file.write(contents)


def save_footprints_to_disk(dictionary, file_path=FOOTPRINTS):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


def load_footprints_from_disk(file_path=FOOTPRINTS):
    with open(file_path) as file:
        return json.load(file)


def load_solution_from_disk(file_name, file_path=SOLUTION_DIR):
    with open(os.path.join(file_path, file_name)) as file:
        return file.read()


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
# solve_it template
# ==============================================================
"""

# from functions import fake_solver


def fake_solver(input_data):
    data = parse_input_data(input_data, input_type="string")
    footprints = load_footprints_from_disk()
    footprint = calculate_data_footprint(data)
    file_name = footprints[str(footprint)]
    output_data = load_solution_from_disk(file_name)
    return output_data


"""
# ==============================================================
# problem specific functions
# ==============================================================
"""


"""
# source :
https://www.coursera.org/learn/discrete-optimization/discussions/forums/R8Z6rVxEEea-8wq_-anwpw/threads/EKeqEnpMEe2RnwrIxQ9Aww

# region - Initialization -

# Decision Variables: Instantiate an empty dictionary which contains the chosen color of each node.
# Other Variables: Instantiate a list where each entry is a list with all nodes a particular node is connected to.
# Instantiate a list where each entry is the number of nodes a particular node is connected to.
# Instantiate a state queue with unexplored feasible states.
# Domains: Instantiate a dictionary with one entry for each node with a list of possible colors.

# endregion

while True:
    # region - Branching -

    # Take the remaining uncolored node with smallest domain size.
    # Take the lowest color number possible.
    # Symmetry Breaking: If a new color is taken, take the smallest new one, unassigned colors are interchangeable.
    # Store this exact state as feasible state to enable backtracking, if there is still another option left
    # Remove the entry from the node domains so that the value is fixed

    while True:
        # region - Pruning and Feasibility Check -

        # Break the loop when no more pruning can be conducted.
        # Taking into account the latest branching use constraints to remove domains from variables.
        # If there is only one possible color remaining for a node assign it.
        # If the connected node is colored it cannot have the same color -> not feasible 
        # If a node has a empty domain -> not feasible

    # If feasible continue with the next loop, otherwise backtrack to the last feasible state.

"""
