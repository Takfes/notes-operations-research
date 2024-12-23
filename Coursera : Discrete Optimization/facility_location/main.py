import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def list_files_in_dir(dir_path):
    return [os.path.join(dir_path, file) for file in os.listdir(dir_path)]


def parse_data_from_file(file_path):
    """Parse a facility location problem from a file"""
    # Define column names for facilities and customers
    facilities_columns = ["cost", "capacity", "fx", "fy"]
    customerers_columns = ["demand", "cx", "cy"]

    with open(file_path) as file:
        lines = file.readlines()

    # Parse problem parameters
    n_facilities, n_customers = map(int, lines[0].split())

    # Parse facilities data
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


def calculate_distances(facilities, customers):
    facility_coords = facilities[["fx", "fy"]].to_numpy()
    customer_coords = customers[["cx", "cy"]].to_numpy()

    distances = np.linalg.norm(
        facility_coords[:, np.newaxis, :] - customer_coords[np.newaxis, :, :],
        axis=2,
    )

    return pd.DataFrame(
        distances, index=facilities.index, columns=customers.index
    ).T


data_dir = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data"

file_paths = list_files_in_dir(data_dir)

# file_path = file_paths[0]

# file_path = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data/fl_3_1"

# file_path = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data/fl_50_6"

# file_path = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data/fl_100_14"

file_path = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data/fl_200_8"

# file_path = "/Users/takis/Documents/sckool/notes-operations-research/Coursera : Discrete Optimization/facility_location/data/fl_4000_1"

print(f"file_path: {file_path}")

EXPLORATION_ENABLED = False

"""
# ==============================================================
# parse data
# ==============================================================
"""
data = parse_data_from_file(file_path)
n_facilities = data["n_facilities"]
n_customers = data["n_customers"]
facilities = data["facilities"]
customers = data["customers"]
print(
    f"Size of the problem : {n_facilities} facilities, {n_customers} customers"
)

"""
# ==============================================================
# data exploration
# ==============================================================
"""
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

    # Explore customer demand
    customer_demand_coefficient_of_variation = (
        customers.demand.std() / customers.demand.mean()
    ).item()
    print(
        f"Customer demand coefficient of variation: {customer_demand_coefficient_of_variation:.4f}"
    )
    customers.demand.hist()
    plt.show()

    # Explore geographical distribution of facilities and customers
    plot_facilities_and_customers(facilities, customers)

    # Explore distances
    dm_wide = calculate_distances(facilities, customers)
    dm = (
        dm_wide.reset_index()
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

"""
# ==============================================================
# create model
# ==============================================================
"""
SOLVER = "appsi_highs"  # "glpk" "ipopt" "appsi_highs" "cbc"
model = pyo.ConcreteModel()

"""
# ==============================================================
# define sets
# ==============================================================
"""
set_facilities = range(n_facilities)
set_customers = range(n_customers)

"""
# ==============================================================
# define variables
# ==============================================================
"""
x = model.x = pyo.Var(set_facilities, domain=pyo.Binary)
y = model.y = pyo.Var(set_customers, set_facilities, domain=pyo.Binary)

"""
# ==============================================================
# define constraints
# ==============================================================
"""

# * CONSTRAINT: each facility must be open to serve customers
model.facility_open = pyo.ConstraintList()
for c in set_customers:
    for f in set_facilities:
        model.facility_open.add(y[c, f] <= x[f])
# model.facility_open.pprint()

# * CONSTRAINT: each customer must be assigned to exactly one facility
model.customer_allocation = pyo.ConstraintList()
for c in set_customers:
    model.customer_allocation.add(sum(y[c, f] for f in set_facilities) == 1)
# model.customer_allocation.pprint()

# * CONSTRAINT: each facility must sustain the demand of all customers assigned to it
model.demand_satisfaction = pyo.ConstraintList()
for f in set_facilities:
    facility_capacity = facilities.capacity.loc[f]
    facility_demand = sum(
        customers.demand.loc[c] * y[c, f] for c in set_customers
    )
    model.demand_satisfaction.add(facility_demand <= facility_capacity)
# model.demand_satisfaction.pprint()

"""
# ==============================================================
# define objective function
# ==============================================================
"""
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


"""
# ==============================================================
# solve model
# ==============================================================
"""
print(f"Solving model with {SOLVER=}...")
opt = SolverFactory(SOLVER)
start = time.perf_counter()
results = opt.solve(model)
end = time.perf_counter()

"""
# ==============================================================
# output results
# ==============================================================
"""

# print(f"Solution took {results.solver.time:.4f} seconds")
print(f"Exeuction time : {end - start:.4f} seconds")

if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    objective_value = pyo.value(model.obj)
    optimization_status = int(
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    )

    answer = f"{objective_value:.2f} {optimization_status}"
    answer += "\n"

    counter = 0
    for c in set_customers:
        for f in set_facilities:
            if pyo.value(y[c, f]) > 0.5:
                answer += f"{f} "
                counter += 1

    answer = answer.strip()
    print("Solution:")
    print(answer)
    assert counter == n_customers, (
        f"Number of customers assigned: {counter} vs {n_customers}"
    )


# for f in set_facilities:
#     print(f"Facility {f} value: {pyo.value(x[f])}")

# for c in set_customers:
#     for f in set_facilities:
#         if pyo.value(y[c, f]) > 0.5:
#             print(
#                 f"Customer {c} assigned to facility {f} value: {pyo.value(y[c, f])}"
#             )
