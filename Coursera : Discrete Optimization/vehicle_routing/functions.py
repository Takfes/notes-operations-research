import hashlib
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
    n_customers, n_vehicles, vehicle_capacity = map(int, lines[0].split())

    # Parse problem data
    data_columns = ["demand", "x", "y"]
    tail_lines = [x for x in lines if x != "\n" and x != " "][1:]
    assert len(tail_lines) == n_customers, (
        f"Expected {n_customers} items, got {len(tail_lines)}"
    )
    dataset = pd.DataFrame(
        [list(map(float, x.split())) for x in tail_lines],
        columns=data_columns,
        dtype=float,
    )

    assert dataset.isnull().sum().sum() == 0, "Dataset contains NaN values"

    return {
        "n_customers": n_customers,
        "n_vehicles": n_vehicles,
        "vehicle_capacity": vehicle_capacity,
        "dataset": dataset,
    }


def calculate_data_footprint(data):
    return (
        data["n_customers"]
        + data["n_vehicles"]
        + data["vehicle_capacity"]
        + data["dataset"].sum().sum().item()
    )


# TODO : Alternative to `calculate_data_footprint`
def calculate_dataframe_hash(data) -> str:
    # Convert DataFrame to a string representation
    df_str = (
        str(data["n_customers"])
        + str(data["n_vehicles"])
        + str(data["vehicle_capacity"])
    )
    df_str += data["dataset"].to_string()
    # Calculate the hash using SHA-256
    hash_object = hashlib.sha256(df_str.encode())
    return hash_object.hexdigest()


# TODO : select calculate_dataframe_hash or calculate_data_footprint
footprint_function = calculate_dataframe_hash


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
