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


# def generate_output(sequence, distance_matrix):
#     obj = calculate_sequence_length_from_distance_matrix(
#         distance_matrix=distance_matrix, sequence=sequence
#     )
#     answer = f"{obj} 0\n"
#     answer += " ".join(map(str, sequence[:-1]))
#     return answer


def generate_dummy_output(sequence, distance_matrix):
    obj = 999999999.99
    answer = f"{obj} 0\n"
    answer += " ".join(map(str, distance_matrix.index.tolist()))
    return answer


"""
# ==============================================================
# Knapsack functions
# ==============================================================
"""

# TODO : MIP pyomo highs sovler
# TODO : BnB sorting by density worth
# TODO : DP
