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
    data["periods"] = pd.DataFrame(periods).set_index("Period Index")
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
generators = data["generators"]
periods = data["periods"]

# define parameters
RESERVE_CAPACITY = 1.15
n_gens = generators["Number available"].to_dict()
min_output = generators["Minimum output (MW)"].to_dict()
max_output = generators["Maximum output (MW)"].to_dict()
base_cost = generators["Cost per hour (when on)"].to_dict()
excess_cost = generators["Cost per hour per MWh above minimum"].to_dict()
startup_cost = generators["Startup cost"].to_dict()
demand_in_period = periods["Demand (megawatts)"].to_dict()
hours_in_period = periods["Hours in the Period"].to_dict()

SOLVER = "glpk"

"""
# ==============================================================
# Formulate model
# ==============================================================
"""

# create model
model = pyo.ConcreteModel()

# define sets
T = periods.index.tolist()
G = generators.index.tolist()

# define variables
# let x be the # of generators of type g ∈ G starting up on time period t ∈ T
model.s = pyo.Var(T, G, domain=pyo.NonNegativeIntegers)
# let w be the # of generators of type g ∈ G already working on time period t ∈ T
model.w = pyo.Var(T, G, domain=pyo.NonNegativeIntegers)
# let p be the total power generated by the generators of type g ∈ G on time period t ∈ T
model.p = pyo.Var(T, G, domain=pyo.NonNegativeReals)

# define constraints
# 1. number of generators
model.number_of_generators = pyo.ConstraintList()
for t in T:
    for g in G:
        model.number_of_generators.add(model.w[t, g] <= n_gens[g])

# 2. generators balance
model.generators_balance = pyo.ConstraintList()
for t in T[1:]:
    for g in G:
        model.generators_balance.add(model.w[t, g] <= model.w[t - 1, g] + model.s[t, g])

# 3. respect generator limits
model.generator_limits = pyo.ConstraintList()
for t in T:
    for g in G:
        model.generator_limits.add(
            model.p[t, g] <= max_output[g] * (model.s[t, g] + model.w[t, g])
        )
        model.generator_limits.add(
            model.p[t, g] >= min_output[g] * (model.s[t, g] + model.w[t, g])
        )

# 4. meet demand
model.demand = pyo.ConstraintList()
for t in T:
    model.demand.add(sum(model.p[t, g] for g in G) >= demand_in_period[t])

# 5. reserve capacity
model.reserve_capacity = pyo.ConstraintList()
for t in T:
    model.reserve_capacity.add(
        sum(model.p[t, g] for g in G) >= RESERVE_CAPACITY * demand_in_period[t]
    )

# define objective
model.obj = pyo.Objective(
    expr=sum(
        startup_cost[g] * model.s[t, g]
        + base_cost[g] * model.w[t, g]
        + excess_cost[g] * (model.p[t, g] - min_output[g] * model.w[t, g])
        for t in T
        for g in G
    ),
    sense=pyo.minimize,
)

"""
# ==============================================================
# Solve model
# ==============================================================
"""

opt = SolverFactory(SOLVER)
results = opt.solve(model)

if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    # The solution is optimal
    # Access and save the solution to a file (e.g., JSON)
    with open("src/C.gurobi/18.tariff-rates/solution.json", "w") as f:
        model.solutions.store_to(results)
        results.write(num=1, format="json", ostream=f)

    # Access other solution information as needed
    print(75 * "-")
    print(f"Objective ${pyo.value(model.obj):=.2f}")
    print(75 * "-")


def reporting(model_variable):
    tmp = {(t, g): pyo.value(model_variable[t, g]) for t in T for g in G}
    df = pd.DataFrame().from_dict(tmp, orient="index", columns=["value"]).reset_index()
    df[["T", "G"]] = pd.DataFrame(df["index"].tolist(), index=df.index)
    df.drop("index", axis=1, inplace=True)
    return df.pivot(index="G", columns="T", values="value")


reporting(model.w)
reporting(model.s)
reporting(model.p)

# TODO fix discrepancy with the solution - a constraing must be wrong
# TODO how to apply the same problem with the pyomo functions logic
# TODO how to make the sums more concise
# TODO enable script with util functions (solvers, reporting, etc)
