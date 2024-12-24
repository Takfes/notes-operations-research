import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Data Paths
ROOT_DIR = Path(__file__).resolve().parent
FOOTPRINTS = ROOT_DIR / "footprints.json"
SOLUTION_DIR = ROOT_DIR / "sols"
DATA_DIR = ROOT_DIR / "data"


def list_files_in_dir(dir_path=DATA_DIR, full_path=True):
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    if not full_path:
        files = [os.path.basename(file) for file in files]
    return files


def parse_input_data(input_data, input_type="string"):
    """Parse a facility location problem from either a file or directly from a string."""

    if input_type == "string":
        lines = input_data.split("\n")[:-1]
    elif input_type == "file":
        with open(input_data) as file:
            lines = file.readlines()
    else:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Parse problem parameters
    n_facilities, n_customers = map(int, lines[0].split())

    # Parse facilities data
    facilities_columns = ["cost", "capacity", "fx", "fy"]
    facilities_lines = lines[1 : n_facilities + 1]
    assert len(facilities_lines) == n_facilities, (
        f"Expected {n_facilities} facilities, got {len(facilities_lines)}"
    )
    facilities = pd.DataFrame(
        [x.split() for x in facilities_lines],
        columns=facilities_columns,
        dtype=float,
    )

    # Parse customers data
    customerers_columns = ["demand", "cx", "cy"]
    customers_lines = lines[n_facilities + 1 :]
    assert len(customers_lines) == n_customers, (
        f"Expected {n_customers} customers, got {len(customers_lines)}"
    )
    customers = pd.DataFrame(
        [x.split() for x in customers_lines],
        columns=customerers_columns,
        dtype=float,
    )

    return {
        "n_facilities": n_facilities,
        "n_customers": n_customers,
        "facilities": facilities,
        "customers": customers,
    }


def write_solution(contents, file_name, file_path=SOLUTION_DIR):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), "w") as file:
        file.write(contents)


def euclidean_distance(point1x, point1y, point2x, point2y):
    return math.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)


def plot_facilities_and_customers(facilities, customers):
    plt.figure(figsize=(10, 8))

    # Plot facilities
    plt.scatter(
        facilities.fx, facilities.fy, c="red", marker="s", label="Facilities"
    )

    # Plot customers
    plt.scatter(
        customers.cx, customers.cy, c="blue", marker="o", label="Customers"
    )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Facility and Customer Locations")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_distance_matrix(facilities, customers):
    facility_coords = facilities[["fx", "fy"]].to_numpy()
    customer_coords = customers[["cx", "cy"]].to_numpy()

    distances = np.linalg.norm(
        facility_coords[:, np.newaxis, :] - customer_coords[np.newaxis, :, :],
        axis=2,
    )

    return pd.DataFrame(
        distances, index=facilities.index, columns=customers.index
    )


def calculate_data_footprint(data):
    data_id = (
        data["n_facilities"]
        + data["n_customers"]
        + data["facilities"].sum().sum().item()
        + data["customers"].sum().sum().item()
    )
    return data_id


def save_footprints_to_disk(dictionary, file_path=FOOTPRINTS):
    with open(file_path, "w") as file:
        json.dump(dictionary, file)


def load_footprints_from_disk(file_path=FOOTPRINTS):
    with open(file_path) as file:
        return json.load(file)


# ? DEBUGGING
# data = parse_input_data(Path(DATA_DIR) / Path("fl_50_6"), input_type="file")


def build_model(data):
    # -------------- PARSE DATA --------------
    n_facilities = data["n_facilities"]
    n_customers = data["n_customers"]
    facilities = data["facilities"]
    customers = data["customers"]

    # -------------- SETUP MODEL --------------
    model = pyo.ConcreteModel()

    # -------------- DEFINE SETS --------------
    # set_facilities = range(n_facilities)
    # set_customers = range(n_customers)
    set_facilities = model.set_facilities = pyo.Set(
        initialize=range(n_facilities)
    )
    set_customers = model.set_customers = pyo.Set(initialize=range(n_customers))
    # -------------- DEFINE VARIABLES --------------
    x = model.x = pyo.Var(set_facilities, domain=pyo.Binary)
    y = model.y = pyo.Var(set_customers, set_facilities, domain=pyo.Binary)

    # -------------- DEFINE CONSTRAINTS --------------
    # constraint: each facility must be open to serve customers
    model.facility_open = pyo.ConstraintList()
    for c in set_customers:
        for f in set_facilities:
            model.facility_open.add(y[c, f] <= x[f])

    # constraint: each customer must be assigned to exactly one facility
    model.customer_allocation = pyo.ConstraintList()
    for c in set_customers:
        model.customer_allocation.add(sum(y[c, f] for f in set_facilities) == 1)

    # constraint: each facility must sustain the demand of all customers assigned to it
    model.demand_satisfaction = pyo.ConstraintList()
    for f in set_facilities:
        facility_capacity = facilities.capacity.loc[f]
        facility_demand = sum(
            customers.demand.loc[c] * y[c, f] for c in set_customers
        )
        model.demand_satisfaction.add(facility_demand <= facility_capacity)

    # -------------- DEFINE OBJECTIVE FUNCTION --------------
    facility_cost_to_open = sum(
        facilities.cost.loc[f] * x[f] for f in set_facilities
    )
    facility_cost_to_serve = sum(
        euclidean_distance(
            facilities.fx.loc[f],
            facilities.fy.loc[f],
            customers.cx.loc[c],
            customers.cy.loc[c],
        )
        * y[c, f]
        for c in set_customers
        for f in set_facilities
    )
    model.obj = pyo.Objective(
        expr=facility_cost_to_open + facility_cost_to_serve, sense=pyo.minimize
    )
    return model


def generate_output(model, results):
    objective_value = pyo.value(model.obj)
    is_optimal_solution = int(
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    )

    output_data = f"{objective_value:.2f} {is_optimal_solution}"
    output_data += "\n"

    counter = 0
    for c in model.set_customers:
        for f in model.set_facilities:
            if pyo.value(model.y[c, f]) > 0.5:
                output_data += f"{f} "
                counter += 1

    output_data = output_data.strip()

    assert counter == len(model.set_customers), (
        f"Number of customers assigned: {counter} vs {model.set_customers}"
    )
    return output_data
