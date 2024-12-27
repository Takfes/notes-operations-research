import json
import os
import time
from pathlib import Path

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
    info1, info2 = map(int, lines[0].split())

    # Parse problem data
    data_columns = ["colname1", "colname2"]
    lines = [x for x in lines if x != "\n"][1:]
    assert len(lines) == info1, f"Expected {info1} items, got {len(lines)}"
    dataset = pd.DataFrame(
        [list(map(int, x.split())) for x in lines],
        columns=data_columns,
        dtype=int,
    )

    return {
        "info1": info1,
        "info2": info2,
        "dataset": dataset,
    }


# TODO : Adjust this to problem specific data
def calculate_data_footprint(data):
    return data["info1"] + data["info2"] + data["dataset"].sum().sum().item()


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
