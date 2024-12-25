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


"""
# ==============================================================
# TSP functions
# ==============================================================
"""


def euclidean_distance(point1x, point1y, point2x, point2y):
    return math.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)


def calculate_trip_sequence_from_distance_matrix(
    distance_matrix, sequence, close_loop=False
):
    # Calculate total trip distance using the distance matrix
    total_distance = sum(
        distance_matrix.iloc[sequence[i], sequence[i + 1]]
        for i in range(len(sequence) - 1)
    )
    # Closing the loop
    if close_loop:
        total_distance += distance_matrix.iloc[sequence[-1], sequence[0]]
    return total_distance


def calculate_trip_sequence_from_locations(
    locations, sequence, close_loop=False
):
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
    # Closing the loop
    if close_loop:
        total_distance += euclidean_distance(
            locations.iloc[sequence[-1]]["x"],
            locations.iloc[sequence[-1]]["y"],
            locations.iloc[sequence[0]]["x"],
            locations.iloc[sequence[0]]["y"],
        )
    return total_distance


def plot_tsp_trip(
    locations,
    sequence=None,
    marker_size=100,
    title="TSP Trip",
    close_loop=False,
):
    if sequence is None:
        sequence = locations.index.tolist()
    else:
        assert len(sequence) <= len(locations) + 1, (
            "Sequence length should be less than or equal to the number of locations + 1 (if the trip is closed)"
        )

    # Calculate total trip distance
    total_distance = calculate_trip_sequence_from_locations(
        locations=locations, sequence=sequence, close_loop=close_loop
    )
    plt.figure(figsize=(10, 6))
    plt.scatter(locations["x"], locations["y"], s=marker_size, c="blue")

    for i, (x, y) in enumerate(zip(locations["x"], locations["y"])):
        plt.annotate(
            i, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Plotting the trip
    for i in range(len(sequence) - 1):
        start = locations.iloc[sequence[i]]
        end = locations.iloc[sequence[i + 1]]
        plt.plot([start["x"], end["x"]], [start["y"], end["y"]], "k-")

    # Closing the loop if sequence is not empty
    if close_loop and sequence:
        start = locations.iloc[sequence[-1]]
        end = locations.iloc[sequence[0]]
        plt.plot([start["x"], end["x"]], [start["y"], end["y"]], "k-")

    # Plotting the plot title
    plt.title(
        f"{title} : {min(len(sequence), len(locations))} / {len(locations)} points | Total trip size: {total_distance:.2f}"
    )
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis("off")
    # plt.gca().xaxis.set_ticks_position("none")
    # plt.gca().yaxis.set_ticks_position("none")
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["left"].set_visible(False)
    # plt.gca().spines["bottom"].set_visible(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().set_axis_off()
    # plt.grid(False)
    plt.show()


def calculate_distance_matrix(locations):
    coords = locations[["x", "y"]].values
    dist_matrix = np.sqrt(
        ((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2)
    )
    return pd.DataFrame(dist_matrix)


def tsp_nearest_point_insertion(
    distance_matrix, close_loop=False, start_point=0
):
    sequence = [start_point]
    distances = []
    while len(sequence) < distance_matrix.shape[0]:
        # last element
        current_point = sequence[-1]
        # closest next point starting from the last sequence
        next_point = (
            distance_matrix.iloc[
                current_point, ~distance_matrix.index.isin(sequence)
            ]
            .idxmin()
            .item()
        )
        # add the next point to the sequence list
        sequence.append(next_point)
        # add the distance between the current and next point to the distances list
        distances.append(distance_matrix.iloc[current_point, next_point].item())

    if close_loop:
        sequence.append(sequence[0])
        distances.append(
            distance_matrix.iloc[sequence[-2], sequence[-1]].item()
        )

    assert len(sequence) == len(distances) + 1, (
        "Sequence and distances mismatch"
    )
    return sequence, distances


# TODO : Greedy insertion heuristic
def tsp_greedy_subtrip_insertion(
    distance_matrix, start_point=0, subtrip_size=3, neighborhood_size=5
):
    # start by defining an initial subtrip
    sequence = [start_point]
    while len(sequence) < subtrip_size:
        # last element
        current_point = sequence[-1]
        # closest next point starting from the last sequence
        next_point = (
            distance_matrix.iloc[
                current_point, ~distance_matrix.index.isin(sequence)
            ]
            .idxmin()
            .item()
        )
        # add the next point to the sequence list
        sequence.append(next_point)

    # close loop to finish the subtrip
    sequence.append(sequence[0])

    # define neighborhood of the subtrip
    # find distance of all points to the subtrip excluding the subtrip itself
    # find the closest X points to the subtrip
    neighborhood = (
        distance_matrix.iloc[
            list(set(sequence)), ~distance_matrix.index.isin(sequence)
        ]
        .melt(ignore_index=False)
        .reset_index()
        .rename(
            columns={"index": "from", "variable": "to", "value": "distance"}
        )
        .nsmallest(neighborhood_size, "distance")
    )

    # track the candidates to insert into the subtrip
    neighborhood_candidates = neighborhood.to.unique().tolist()

    # candidate = neighborhood_candidates[0]

    # find the best candidate to insert into the subtrip
    new_subtrip_size = distance_matrix.sum().sum()
    for candidate in neighborhood_candidates:
        for posx in range(1, len(sequence)):
            copy_sequence = sequence.copy()
            print(f"Inserting {candidate=} at position {posx}")
            copy_sequence.insert(posx, candidate)
            print(copy_sequence)
            candidate_size = calculate_trip_sequence_from_distance_matrix(
                distance_matrix, copy_sequence, close_loop=True
            )
            print(candidate_size)
            if candidate_size < new_subtrip_size:
                new_subtrip_size = candidate_size
                new_sequence = copy_sequence
    # assert len(sequence) == len(distances) + 1, (
    #     "Sequence and distances mismatch"
    # )
    # return sequence, distances


# TODO : Greedy swap heuristic
# TODO : Greedy 2-opt heuristic
# TODO : Simulated Annealing
# TODO : Tabu Search
# TODO : or-tools TSP solver
