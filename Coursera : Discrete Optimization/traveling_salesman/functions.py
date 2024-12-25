import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
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


def generate_output(sequence, distance_matrix):
    obj = calculate_trip_sequence_from_distance_matrix(
        distance_matrix=distance_matrix, sequence=sequence
    )
    answer = f"{obj} 0\n"
    answer += " ".join(map(str, sequence[:-1]))
    return answer


"""
# ==============================================================
# TSP custom functions
# ==============================================================
"""


def euclidean_distance(point1x, point1y, point2x, point2y):
    return math.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)


def calculate_trip_sequence_from_distance_matrix(
    distance_matrix, sequence, close_loop=True
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
    distance_matrix, close_loop=True, start_point=0
):
    sequence = [start_point]
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

    if close_loop:
        sequence.append(sequence[0])

    return sequence


def tsp_greedy_subtrip_insertion(
    distance_matrix,
    start_point=0,
    subtrip_size=2,
    neighborhood_size=None,
):
    if neighborhood_size is None:
        neighborhood_size = distance_matrix.shape[0] // 5
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

    # repeat the process until the sequence includes all points, that is the size of the distance matrix + 1, since the loop is closed
    while len(sequence) <= distance_matrix.shape[0]:
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

        # find the best candidate to insert into the subtrip
        new_subtrip_size = distance_matrix.sum().sum()
        for candidate in neighborhood_candidates:
            for posx in range(1, len(sequence)):
                copy_sequence = sequence.copy()
                # print(f"Inserting {candidate=} at position {posx}")
                copy_sequence.insert(posx, candidate)
                # print(copy_sequence)
                candidate_size = calculate_trip_sequence_from_distance_matrix(
                    distance_matrix, copy_sequence, close_loop=True
                )
                # print(candidate_size)
                if candidate_size < new_subtrip_size:
                    new_subtrip_size = candidate_size
                    new_sequence = copy_sequence
        sequence = new_sequence

    assert len(sequence) == distance_matrix.shape[0] + 1
    assert len(sequence[:-1]) == len(set(sequence))
    assert sequence[0] == sequence[-1]

    return sequence


def tsp_two_opt_improvement(distance_matrix, sequence):
    # TODO : check whether the remaining process needs to be applied on the initial sequence or the best sequence
    best_sequence = sequence.copy()
    best_sequence_length = calculate_trip_sequence_from_distance_matrix(
        distance_matrix, sequence
    )
    for j in range(1, len(sequence) - 1):
        for i in range(1, j - 1):
            start = sequence[:i]
            middle = sequence[i:j]
            end = sequence[j:]
            new_sequence = start + list(reversed(middle)) + end
            new_sequence_length = calculate_trip_sequence_from_distance_matrix(
                distance_matrix, new_sequence
            )
            if new_sequence_length < best_sequence_length:
                best_sequence_length = new_sequence_length
                best_sequence = new_sequence

    return best_sequence


def tsp_swap_improvement(distance_matrix, sequence):
    # TODO : check whether the remaining process needs to be applied on the initial sequence or the best sequence
    best_sequence = sequence.copy()
    best_sequence_length = calculate_trip_sequence_from_distance_matrix(
        distance_matrix, sequence
    )
    for j in range(1, len(sequence) - 1):
        for i in range(1, j):
            new_sequence = sequence.copy()
            # swap the cities
            new_sequence[i], new_sequence[j] = (
                new_sequence[j],
                new_sequence[i],
            )
            new_sequence_length = calculate_trip_sequence_from_distance_matrix(
                distance_matrix, new_sequence
            )
            if new_sequence_length < best_sequence_length:
                best_sequence_length = new_sequence_length
                best_sequence = new_sequence

    return best_sequence


def start_point_optimizer(distance_matrix, constructive_heuristic):
    best_distance = distance_matrix.sum().sum()
    best_seq = None
    best_start_point = None
    for x in tqdm(distance_matrix.index.tolist()):
        temp_seq = constructive_heuristic(distance_matrix, start_point=x)
        temp_seq_length = calculate_trip_sequence_from_distance_matrix(
            distance_matrix, temp_seq
        )
        if temp_seq_length < best_distance:
            best_distance = temp_seq_length
            best_seq = temp_seq
            best_start_point = x
    print(
        f"Start Point Optimization : {best_distance=:.2f}, {best_start_point=:.2f} | Step function: {constructive_heuristic.__name__}"
    )
    return best_seq, best_distance, best_start_point


def hill_climb_corrective(
    distance_matrix, sequence, step_function, max_iter=100
):
    best_sequence = sequence.copy()
    best_sequence_length = calculate_trip_sequence_from_distance_matrix(
        distance_matrix, best_sequence
    )
    counter = 0

    while True:
        new_sequence = step_function(distance_matrix, best_sequence)
        new_sequence_length = calculate_trip_sequence_from_distance_matrix(
            distance_matrix, new_sequence
        )
        counter += 1
        if new_sequence_length < best_sequence_length and counter <= max_iter:
            best_sequence = new_sequence
            best_sequence_length = new_sequence_length
        else:
            print(
                f"Hill Climb : Best sequence length: {best_sequence_length:.2f} | Iterations: {counter} | Step function: {step_function.__name__}"
            )
            return best_sequence


# TODO : Simulated Annealing
# TODO : Tabu Search

"""
# ==============================================================
# TSP or tools constraint solver
# ==============================================================
"""


def tsp_or_constraint_solver(
    distance_matrix, search_strategy="guided", logging=False, time_limit=None
):
    """_summary_

    Args:
        distance_matrix (pd.DataFrame): _description_
        search_strategy (str, optional): guided or first. Defaults to 'guided'.
        logging (bool, optional): _description_. Defaults to False.
        time_limit (_type_, optional): _description_. Defaults to None.
    """

    def get_routes(solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    # Instantiate the data problem.
    matrix = [list(map(int, row)) for row in distance_matrix.values]
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    # Register a transit callback.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Setting the search strategy
    if search_strategy == "first":
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
    elif search_strategy == "guided":
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
    # Setting the time limit
    if time_limit is not None:
        search_parameters.time_limit.seconds = 30
    # Setting the logging
    search_parameters.log_search = logging
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    routes = get_routes(solution, routing, manager)
    sequence = routes[0]

    return sequence
