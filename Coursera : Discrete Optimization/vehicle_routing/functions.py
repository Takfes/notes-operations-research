import hashlib
import json
import os
import time
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import pdist, squareform

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


def generate_output(vrpdata, sequence, optimized_indicator=0):
    # get number of customers
    n_customers = vrpdata["n_customers"]
    # get number of vehicles
    n_vehicles = vrpdata["n_vehicles"]
    # calculate distance matrix
    dm = calculate_distance_matrix(vrpdata["dataset"])
    # instantiate output variables
    answer_iter = ""
    total_distance = 0
    seq_list = sequence.strip().split("\n")
    # check if the number of vehicles is correct
    assert n_vehicles == len(seq_list), (
        f"Output contains less vehicles than expected: {n_vehicles=}, output_sequences={len(seq_list)}"
    )
    visited_custs = []
    # iterate over the sequence list
    for sq in seq_list:
        vid, vseq = sq.split("):")
        vid = int(vid)
        answer_iter += vseq.strip() + "\n"
        vseq = list(map(int, vseq.strip().split(" ")))
        # validate that customers are visited only once
        new_custs = vseq[1:-1]
        assert len(set(new_custs).intersection(visited_custs)) == 0, (
            f"Customer visited more than once over the sequence: {vid}"
        )
        visited_custs.extend(new_custs)
        # calculate total distance
        for start, end in zip(vseq[:-1], vseq[1:]):
            total_distance += dm.loc[start, end].item()
    # check if all customers are visited
    assert len(visited_custs) + 1 == n_customers, (
        f"Expected {n_customers} customers to be visited, got {len(visited_custs) + 1}"
    )
    # generate the answer
    answer = (
        f"{total_distance:.2f} {optimized_indicator}\n{answer_iter.strip()}"
    )
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
    footprint = footprint_function(data)
    file_name = footprints[str(footprint)]
    output_data = load_solution_from_disk(file_name)
    return output_data


"""
# ==============================================================
# problem specific functions
# ==============================================================
"""


def calculate_distance_matrix(dataframe, chunk_size=1000):
    """
    Calculate euclidean distance matrix efficiently using chunks and scipy's pdist.

    Args:
        dataframe: DataFrame with 'x' and 'y' columns
        chunk_size: Size of chunks to process at once

    Returns:
        DataFrame containing the distance matrix
    """
    coords = dataframe[["x", "y"]].values
    n = len(coords)

    # For small matrices, use scipy's pdist directly
    if n <= chunk_size:
        distances = pdist(coords)
        return pd.DataFrame(squareform(distances))

    # Initialize output matrix
    result = np.zeros((n, n))

    # Calculate number of chunks needed
    n_chunks = ceil(n / chunk_size)

    # Process upper triangle in chunks
    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, n)

        for j in range(i, n_chunks):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n)

            # Calculate distances for this chunk
            chunk_distances = np.sqrt(
                (
                    (
                        coords[start_i:end_i, np.newaxis, :]
                        - coords[np.newaxis, start_j:end_j, :]
                    )
                    ** 2
                ).sum(axis=2)
            )

            # Store in result matrix
            result[start_i:end_i, start_j:end_j] = chunk_distances

            # Mirror the distances for lower triangle (skip diagonal chunks)
            if i != j:
                result[start_j:end_j, start_i:end_i] = chunk_distances.T

    return pd.DataFrame(result)


@timeit
def or_vrp_solver(
    vrpdata, depot_index=0, logging=False, time_limit=None, verbose=False
):
    # define solution printer
    def print_solution(data, manager, routing, solution):
        print(f"Objective: {solution.ObjectiveValue()}")
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data["demands"][node_index]
                plan_output += f" {node_index} Load({route_load}) -> "
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
            plan_output += f"Distance of the route: {route_distance}m\n"
            plan_output += f"Load of the route: {route_load}\n"
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print(f"Total distance of all routes: {total_distance}m")
        print(f"Total load of all routes: {total_load}")

    def generate_sequence(data, manager, routing, solution):
        plan_output = ""
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            plan_output += f"{vehicle_id}): "
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data["demands"][node_index]
                plan_output += f"{node_index} "
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            plan_output += f"{manager.IndexToNode(index)}\n"
        return plan_output

    # create data model
    def create_data_model(vrpdata, depot_index):
        # initiate data storage
        data = {}
        # distance matrix calculation
        dm = calculate_distance_matrix(vrpdata["dataset"])
        # translate the matrix for the or library
        matrix = [list(map(int, row)) for row in dm.values]
        # populate data
        data["distance_matrix"] = matrix
        data["demands"] = vrpdata["dataset"]["demand"].astype(int).tolist()
        data["vehicle_capacities"] = [vrpdata["vehicle_capacity"]] * vrpdata[
            "n_vehicles"
        ]
        data["num_vehicles"] = vrpdata["n_vehicles"]
        data["depot"] = depot_index
        return data

    # create a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    # create a demand callback.
    def demand_callback(from_index):
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    # instantiate the data model
    data = create_data_model(vrpdata, depot_index)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Register the distance callback.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc - here cost comes directly from the distance
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add distance dimensions.
    # This computes the cumulative distance traveled by each vehicle along its route.
    dimension_name = "Distance"
    vehicle_maximum_distance = np.sum(data["distance_matrix"]).item()
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        vehicle_maximum_distance,  # vehicle maximum travel distance # 3000
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # register the demand callback
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    # Add capcity dimensions.
    # This accumulates the weight of the load a vehicle is carrying over the route - pertitent to the capacity constraint
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Setting the time limit
    if time_limit is not None:
        search_parameters.time_limit.seconds = time_limit

    # Setting the logging
    search_parameters.log_search = logging

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        if verbose:
            print_solution(data, manager, routing, solution)
        sequence = generate_sequence(data, manager, routing, solution)
        return sequence
    else:
        print("No solution found !")
