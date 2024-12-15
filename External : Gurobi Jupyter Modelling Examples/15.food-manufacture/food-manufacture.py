# Model Building in Mathematical Programming
# Food Manufacture
# https://www.gurobi.com/jupyter_models/factory-planning/
from itertools import product

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def generate_data():
    data = {}
    # Price per product and month
    price = {
        "VEG 1": [110, 130, 110, 120, 100, 90],
        "VEG 2": [120, 130, 140, 110, 120, 100],
        "OIL 1": [130, 110, 130, 120, 150, 140],
        "OIL 2": [110, 90, 100, 120, 110, 80],
        "OIL 3": [115, 115, 95, 125, 105, 135],
    }
    data["price"] = pd.DataFrame(
        price, index=["01-January", "02-February", "03-March", "04-April", "05-May", "06-June"]
    )
    # Hardness per raw product
    hardness = {
        "Oil Type": ["VEG 1", "VEG 2", "OIL 1", "OIL 2", "OIL 3"],
        "Value": [8.8, 6.1, 2.0, 4.2, 5.0],
    }
    data["hardness"] = pd.DataFrame(hardness)
    # return data
    return data


# extract data
data = generate_data()
price_df = data["price"]
hardness_df = data["hardness"]

# create model
SOLVER = "glpk"
model = pyo.ConcreteModel()

# define parameters
PRICE_PER_TON = 150
MONTHLY_CAPACITY_TONS_VEGETABLE_OIL = 200
MONTHLY_CAPACITY_TONS_NONVEGETABLE_OIL = 250
STORAGE_CAPACITY_TONS = 1000
STORAGE_COST_PER_TON = 5
STARTING_STORAGE_TONS = 500
ENDING_STORAGE_TONS = 500
HARDNESS_MIN = 3
HARDNESS_MAX = 6

price = price_df.stack().to_dict()
hardness = hardness_df.set_index("Oil Type")["Value"].to_dict()

# define sets
set_months_m = price_df.index.tolist()
set_products_p = price_df.columns.tolist()
set_vegetable_oils_v = hardness_df.loc[
    hardness_df["Oil Type"].str.startswith("VEG"), "Oil Type"
].tolist()
set_nonvegetable_oils_n = hardness_df.loc[
    ~hardness_df["Oil Type"].str.startswith("VEG"), "Oil Type"
].tolist()

# define variables
model.buy = pyo.Var(set_months_m, set_products_p, domain=pyo.NonNegativeReals)
model.make = pyo.Var(set_months_m, set_products_p, domain=pyo.NonNegativeReals)
model.store = pyo.Var(
    set_months_m, set_products_p, domain=pyo.NonNegativeReals, bounds=(0, STORAGE_CAPACITY_TONS)
)

# define objective function
production_revenue = sum(
    model.make[m, p] * PRICE_PER_TON for m in set_months_m for p in set_products_p
)
raw_material_cost = sum(
    model.buy[m, p] * price[m, p] for m in set_months_m for p in set_products_p
)
storage_cost = sum(
    model.store[m, p] * STORAGE_COST_PER_TON for m in set_months_m for p in set_products_p
)
model.obj = pyo.Objective(
    expr=production_revenue - raw_material_cost - storage_cost, sense=pyo.maximize
)

# define constraint : PRODUCTION CAPACITY
model.production_capacity = pyo.ConstraintList()
for m in set_months_m:
    vegetable_refinement = sum(model.make[m, v] for v in set_vegetable_oils_v)
    non_vegetable_refinement = sum(model.make[m, n] for n in set_nonvegetable_oils_n)
    model.production_capacity.add(vegetable_refinement <= MONTHLY_CAPACITY_TONS_VEGETABLE_OIL)
    model.production_capacity.add(
        non_vegetable_refinement <= MONTHLY_CAPACITY_TONS_NONVEGETABLE_OIL
    )

# define constraint : HARDNESS
model.hardness = pyo.ConstraintList()
for m in set_months_m:
    model.hardness.add(
        sum(model.make[m, p] * hardness[p] for p in set_products_p)
        >= sum(model.make[m, p] for p in set_products_p) * HARDNESS_MIN
    )
    model.hardness.add(
        sum(model.make[m, p] * hardness[p] for p in set_products_p)
        <= sum(model.make[m, p] for p in set_products_p) * HARDNESS_MAX
    )

# define constraint : STARTING STORAGE BALANCE
# ! make = buy + store_previous_month - store_current_month
model.balance = pyo.ConstraintList()
for m in set_months_m:
    for p in set_products_p:
        if m == set_months_m[0]:
            model.balance.add(
                model.make[m, p] == model.buy[m, p] + STARTING_STORAGE_TONS - model.store[m, p]
            )
        else:
            model.balance.add(
                model.make[m, p]
                == model.buy[m, p]
                + model.store[set_months_m[set_months_m.index(m) - 1], p]
                - model.store[m, p]
            )

# define constraint : ENDING STORAGE BALANCE
model.ending_storage = pyo.ConstraintList()
for p in set_products_p:  # ! last month
    model.ending_storage.add(model.store[set_months_m[-1], p] == ENDING_STORAGE_TONS)

# solve model
opt = SolverFactory(SOLVER)
results = opt.solve(model)

# output results
if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    # The solution is optimal
    # Access and save the solution to a file (e.g., JSON)
    with open("src/15.food-manufacture/solution.json", "w") as f:
        model.solutions.store_to(results)
        results.write(num=1, format="json", ostream=f)

    # Access other solution information as needed
    print(75 * "-")
    print(f"Profit ${pyo.value(model.obj):=.2f}")
    print(75 * "-")

    make_plan = (
        pd.DataFrame([(k[0], k[1], round(pyo.value(v), 2)) for k, v in model.make.items()])
        .rename(columns={0: "month", 1: "product", 2: "make"})
        .pivot(index="month", columns="product", values="make")
    )
    print("Production Plan:")
    print(make_plan)
    print(75 * "-")

    sell_plan = (
        pd.DataFrame([(k[0], k[1], round(pyo.value(v), 2)) for k, v in model.buy.items()])
        .rename(columns={0: "month", 1: "product", 2: "purchase"})
        .pivot(index="month", columns="product", values="purchase")
    )
    print("Purchase Plan:")
    print(sell_plan)
    print(75 * "-")

    store_plan = (
        pd.DataFrame([(k[0], k[1], round(pyo.value(v), 2)) for k, v in model.store.items()])
        .rename(columns={0: "month", 1: "product", 2: "store"})
        .pivot(index="month", columns="product", values="store")
    )
    print("Inventory Plan:")
    print(store_plan)
    print(75 * "-")

    hardness_plan = (
        make_plan.mul(hardness_df["Value"].values, axis=1).sum(axis=1).div(make_plan.sum(axis=1))
    ).round(2)

    print("Hardness Plan:")
    print(hardness_plan)
    print(75 * "-")

else:
    print("Solver did not find an optimal solution.")
