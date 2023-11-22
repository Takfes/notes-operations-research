import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def generate_data():
    data = {}
    # demand per time period
    periods = {
        "Time Period": [
            "12 pm to 6 am",
            "6 am to 9 am",
            "9 am to 3 pm",
            "3 pm to 6 pm",
            "6 pm to 12 pm",
        ],
        "Demand (megawatts)": [15000, 30000, 25000, 40000, 27000],
        "Hours in the Period": [6, 3, 6, 3, 6],
        "Period Index": [1, 2, 3, 4, 5],
    }
    data["periods"] = pd.DataFrame(periods).set_index("Time Period")
    # generators
    generators = {
        "Type": [0, 1, 2],
        "Number available": [12, 10, 5],
        "Minimum output (MW)": [850, 1250, 1500],
        "Maximum output (MW)": [2000, 1750, 4000],
        "Cost per hour (when on)": [1000, 2600, 3000],
        "Cost per hour per MWh above minimum": [2.00, 1.30, 3.00],
        "Startup cost": [2000, 1000, 500],
    }
    data["generators"] = pd.DataFrame(generators).set_index("Type")
    return data


# extract data
data = generate_data()
periods = data["periods"]
generators = data["generators"]

# create model
model = pyo.ConcreteModel()
SOLVER = "glpk"

# define parameters
base_cost = generators["Cost per hour (when on)"].to_dict()
startup_cost = generators["Startup cost"].to_dict()
excess_cost = generators["Cost per hour per MWh above minimum"].to_dict()
minimum_output = generators["Minimum output (MW)"].to_dict()
hours_in_period = periods["Hours in the Period"].to_dict()

# define sets
G = generators.index.tolist()
T = periods.index.tolist()

# define variables
# How many generators of type g are on in time period t
model.on = pyo.Var(G, T, domain=pyo.NonNegativeIntegers)
# How many generators of type g must start in time period t
model.start = pyo.Var(G, T, domain=pyo.NonNegativeIntegers)
# What is the output of generator g in time period t
model.output = pyo.Var(G, T, domain=pyo.NonNegativeReals)

# define objective function
modeL_base_cost = [model.on[g, t] * base_cost[g] for g in G for t in T]
model_startup_cost = [model.start[g, t] * startup_cost[g] for g in G for t in T]
model_output_cost = [
    hours_in_period[t] * excess_cost[g] * (model.output[g, t] - minimum_output[g])
    for g in G
    for t in T
]
model.obj = pyo.Objective(
    expr=sum(modeL_base_cost) + sum(model_startup_cost) + sum(model_output_cost)
)
3
# define constraints
