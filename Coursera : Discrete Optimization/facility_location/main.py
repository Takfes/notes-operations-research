import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

"""
# ==============================================================
# DEFINE FUNCTIONS
# ==============================================================
"""


def list_files_in_dir(dir_path, full_path=True):
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


def write_solution(contents, file_path, file_name):
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


def load_dict_from_disk(file_path):
    with open(file_path) as file:
        return json.load(file)


def calculate_data_footprint(data):
    data_id = (
        data["n_facilities"]
        + data["n_customers"]
        + data["facilities"].sum().sum().item()
        + data["customers"].sum().sum().item()
    )
    return data_id


"""
# ==============================================================
# MAIN APPLICATION
# ==============================================================
"""

# test_data = """3 4
# 100 100 1065.0 1065.0
# 100 100 1062.0 1062.0
# 100 500 0.0 0.0
# 50 1397.0 1397.0
# 50 1398.0 1398.0
# 75 1399.0 1399.0
# 75 586.0 586.0
# """


# * DEFINE CONSTANTS
EXPLORATION_ENABLED = False
TIME_LIMIT = 60 * 30  # None
SOLVER_TEE = False
SOLVER = "appsi_highs"  # "glpk" "ipopt" "appsi_highs" "cbc"
STORE_SOLUTION = False

# Where data is stored
FOOTPRINTS = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/footprints.json"
SOLUTION_DIR = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/sols"
DATA_DIR = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data"

# Get data ids
footprints = load_dict_from_disk(FOOTPRINTS)

# Get all data files
data_files = list_files_in_dir(DATA_DIR, full_path=False)
# Input file name
"""
"fl_25_2"         # Facility Location Problem 1
"fl_50_6"         # Facility Location Problem 2
"fl_100_7"        # Facility Location Problem 3
"fl_100_1"        # Facility Location Problem 4 # 1803.6453 seconds | Solver optimality gap: 8.00% | 23161799.54 0
"fl_200_7"        # Facility Location Problem 5 # 252.3118 seconds | Solver optimality gap: 0.01% | 4711458.83 1
"fl_500_7"        # Facility Location Problem 6 # 1276.3987 seconds | Solver optimality gap: 5.28% | 27642010.95 0
"fl_1000_2"       # Facility Location Problem 7
"fl_2000_2"       # Facility Location Problem 8
"""
# file_name = data_files[0]
file_name = "fl_500_7"
# Input file path
file_path = Path(Path(DATA_DIR) / Path(file_name))
print(f"File Name: {file_name}")

# * PARSE DATA
data = parse_input_data(
    file_path, input_type="file"
)  # parse_input_data(test_data, input_type="string")

n_facilities = data["n_facilities"]
n_customers = data["n_customers"]
facilities = data["facilities"]
customers = data["customers"]

print(
    f"Problem Size : {n_facilities=}, {n_customers=}, total={n_facilities * n_customers:,}"
)

# * DATA FOOTPRINT
footprint = calculate_data_footprint(data)
assert footprints[str(footprint)] == file_name
print(f"Data Footprint : {footprint}")


# * DATA EXPLORATION
if EXPLORATION_ENABLED:
    # Explore facility costs
    facility_cost_coefficient_of_variation = (
        facilities.cost.std() / facilities.cost.mean()
    ).item()
    print(
        f"Facility cost coefficient of variation: {facility_cost_coefficient_of_variation:.4f}"
    )
    facilities.cost.hist()
    plt.show()

    facilities.cost.describe()
    facilities.capacity.describe()

    # Explore customer demand
    customer_demand_coefficient_of_variation = (
        customers.demand.std() / customers.demand.mean()
    ).item()
    print(
        f"Customer demand coefficient of variation: {customer_demand_coefficient_of_variation:.4f}"
    )
    customers.demand.hist()
    plt.show()

    customers.demand.describe()
    customers.demand.sum()

    # Explore geographical distribution of facilities and customers
    plot_facilities_and_customers(facilities, customers)

    # Explore distances
    dm_wide = calculate_distance_matrix(
        facilities, customers
    )  # facilities x customers

    dm_wide.sum(axis=1).sort_values(ascending=True)

    dm = (
        dm_wide.T.reset_index()
        .melt(id_vars="index", var_name="facility", value_name="distance")
        .rename(columns={"index": "customer"})
    )

    distance_threshold = dm["distance"].mean()
    distance_threshold = (
        dm.groupby("customer")["distance"].describe()["25%"].max()
    )

    filtered_distances = (
        dm[dm.distance <= distance_threshold]
        .groupby("customer")["distance"]
        .count()
        .sort_values(ascending=True)
    )
    print(filtered_distances)

    # Calculate the average of the X lowest distance values per customer
    x_closest_facilities = int(n_facilities / 2)
    x_closest_facilities = 25
    average_lowest_distances = (
        dm.groupby("customer")["distance"]
        .apply(lambda x: x.nsmallest(x_closest_facilities).min())
        .reset_index(name="avg_5_lowest_distances")
        .sort_values("avg_5_lowest_distances", ascending=False)
    )

    print(average_lowest_distances)

# * SETUP MODEL
model = pyo.ConcreteModel()

# * DEFINE SETS
set_facilities = range(n_facilities)
set_customers = range(n_customers)

# * DEFINE VARIABLES
x = model.x = pyo.Var(set_facilities, domain=pyo.Binary)
y = model.y = pyo.Var(set_customers, set_facilities, domain=pyo.Binary)

# * DEFINE CONSTRAINTS
# * Constraint: each facility must be open to serve customers
model.facility_open = pyo.ConstraintList()
for c in set_customers:
    for f in set_facilities:
        model.facility_open.add(y[c, f] <= x[f])
# model.facility_open.pprint()

# * Constraint: each customer must be assigned to exactly one facility
model.customer_allocation = pyo.ConstraintList()
for c in set_customers:
    model.customer_allocation.add(sum(y[c, f] for f in set_facilities) == 1)
# model.customer_allocation.pprint()

# * Constraint: each facility must sustain the demand of all customers assigned to it
model.demand_satisfaction = pyo.ConstraintList()
for f in set_facilities:
    facility_capacity = facilities.capacity.loc[f]
    facility_demand = sum(
        customers.demand.loc[c] * y[c, f] for c in set_customers
    )
    model.demand_satisfaction.add(facility_demand <= facility_capacity)
# model.demand_satisfaction.pprint()

# * DEFINE OBJECTIVE FUNCTION
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

# * SETUP AND RUN THE SOLVER
solver = SolverFactory(SOLVER)
if TIME_LIMIT is not None:
    solver.options["time_limit"] = TIME_LIMIT

print(f"Solving model with {SOLVER=}...")
start = time.perf_counter()
results = solver.solve(model, tee=SOLVER_TEE)
end = time.perf_counter()
print(f"Execution time : {end - start:.4f} seconds")

# * OUTPUT RESULTS
# * Generic solver results
solver_status = results.solver.status.value
solver_termination_condition = results.solver.termination_condition.value
lower_bound = results.problem.lower_bound
upper_bound = results.problem.upper_bound
# lower_bound = results.get("Problem").get("Lower bound").value
# upper_bound = results.get("Problem").get("Upper bound").value
optimality_gap = 1 - (lower_bound / upper_bound)

print("> Solver Report:")
print(f"* Solver name : {SOLVER}")
print(f"* Solver execution time : {end - start:.4f} seconds")
print(f"* Solver status: {solver_status}")
print(f"* Solver termination condition: {solver_termination_condition}")
print(f"* Solver optimality gap: {optimality_gap:.2%}")


# * Required solver results
objective_value = pyo.value(model.obj)
is_optimal_solution = int(
    results.solver.termination_condition == pyo.TerminationCondition.optimal
)

output_data = f"{objective_value:.2f} {is_optimal_solution}"
output_data += "\n"

counter = 0
for c in set_customers:
    for f in set_facilities:
        if pyo.value(y[c, f]) > 0.5:
            output_data += f"{f} "
            counter += 1

output_data = output_data.strip()
print("> Solution:\n")
print(output_data)
assert counter == n_customers, (
    f"Number of customers assigned: {counter} vs {n_customers}"
)

if STORE_SOLUTION:
    write_solution(
        contents=output_data, file_path=SOLUTION_DIR, file_name=file_name
    )

# for f in set_facilities:
#     print(f"Facility {f} value: {pyo.value(x[f])}")

# for c in set_customers:
#     for f in set_facilities:
#         if pyo.value(y[c, f]) > 0.5:
#             print(
#                 f"Customer {c} assigned to facility {f} value: {pyo.value(y[c, f])}"
#             )
