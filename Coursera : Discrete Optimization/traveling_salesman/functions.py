import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data Paths
ROOT_DIR = Path(__file__).resolve().parent
FOOTPRINTS = ROOT_DIR / "footprints.json"
SOLUTION_DIR = ROOT_DIR / "sols"
DATA_DIR = ROOT_DIR / "data"


def list_files_in_dir(dir_path=DATA_DIR, full_path=True):
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    if not full_path:
        files = [os.path.basename(file) for file in files]
    return sorted(files, key=lambda s: (len(s), s))


def parse_input_data(input_data, input_type="string"):
    """Parse a traveling salesman problem from either a file or directly from a string."""

    if input_type == "string":
        lines = input_data.split("\n")[:-1]
    elif input_type == "file":
        with open(input_data) as file:
            lines = file.readlines()
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Parse problem parameters
    n_locations = int(lines[0].split()[0])

    # Parse facilities data
    location_columns = ["x", "y"]
    location_lines = lines[1:]
    assert len(location_lines) == n_locations, (
        f"Expected {n_locations} facilities, got {len(location_lines)}"
    )
    locations = (
        pd.DataFrame(
            [x.split() for x in location_lines],
            columns=location_columns,
            dtype=float,
        )
        # .rename_axis("location_id")
        # .reset_index()
    )

    return {
        "n_locations": n_locations,
        "locations": locations,
    }


def write_solution(contents, file_name, file_path=SOLUTION_DIR):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), "w") as file:
        file.write(contents)


def euclidean_distance(point1x, point1y, point2x, point2y):
    return math.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)


# TODO : Make direction of the trip more visible
def plot_tsp_trip(locations, sequence=None, marker_size=100, title="TSP Trip"):
    if sequence is None:
        sequence = locations.index.tolist()
    else:
        assert len(sequence) == len(locations), (
            "Sequence length must match number of locations"
        )

    # Calculate total trip distance
    total_distance = sum(
        euclidean_distance(
            locations.iloc[sequence[i]]["x"],
            locations.iloc[sequence[i]]["y"],
            locations.iloc[sequence[i + 1]]["x"],
            locations.iloc[sequence[i + 1]]["y"],
        )
        for i in range(len(sequence) - 1)
    )
    total_distance += euclidean_distance(
        locations.iloc[sequence[-1]]["x"],
        locations.iloc[sequence[-1]]["y"],
        locations.iloc[sequence[0]]["x"],
        locations.iloc[sequence[0]]["y"],
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(locations["x"], locations["y"], s=marker_size, c="blue")

    for i, (x, y) in enumerate(zip(locations["x"], locations["y"])):
        plt.annotate(
            i, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    for i in range(len(sequence) - 1):
        start = locations.iloc[sequence[i]]
        end = locations.iloc[sequence[i + 1]]
        plt.plot([start["x"], end["x"]], [start["y"], end["y"]], "k-")

    # Closing the loop
    start = locations.iloc[sequence[-1]]
    end = locations.iloc[sequence[0]]
    plt.plot([start["x"], end["x"]], [start["y"], end["y"]], "k-")

    plt.title(
        f"{title} - {len(locations)} points - Total trip size: {total_distance:.2f}"
    )
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.show()


def calculate_distance_matrix(locations):
    coords = locations[["x", "y"]].values
    dist_matrix = np.sqrt(
        ((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2)
    )
    return pd.DataFrame(dist_matrix)


def calculate_data_footprint(data):
    return data.sum().sum().item()


def save_footprints_to_disk(dictionary, file_path=FOOTPRINTS):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


def load_footprints_from_disk(file_path=FOOTPRINTS):
    with open(file_path) as file:
        return json.load(file)


def load_solution_from_disk(file_name, file_path=SOLUTION_DIR):
    with open(os.path.join(file_path, file_name)) as file:
        return file.read()


# TODO : Implement this function
def generate_output():
    pass
