# Model Building in Mathematical Programming
# Factory Planning
# https://www.gurobi.com/jupyter_models/factory-planning/
from itertools import product

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# create data
def generate_data():
    data = {}
    # manufacturing dataset
    manufacturing = {
        "PROD 1": [10, 0.5, 0.1, 0.2, 0.05, None],
        "PROD 2": [6, 0.7, 0.2, None, 0.03, None],
        "PROD 3": [8, None, None, 0.8, None, 0.01],
        "PROD 4": [4, None, 0.3, None, 0.07, None],
        "PROD 5": [11, 0.3, None, None, 0.1, 0.05],
        "PROD 6": [9, 0.2, 0.6, None, None, None],
        "PROD 7": [3, 0.5, None, 0.6, 0.08, 0.05],
    }
    index = [
        "Contribution to profit",
        "Grinding",
        "Vertical drilling",
        "Horizontal drilling",
        "Boring",
        "Planing",
    ]
    data["manufacturing"] = pd.DataFrame(manufacturing, index=index)
    # maintenance dataset
    maintenance = {
        "Month": ["January", "February", "March", "April", "May", "June"],
        "Equipment": [
            "1 Grinder",
            "2 Horizontal drills",
            "1 Borer",
            "1 Vertical drill",
            "1 Grinder and 1 Vertical drill",
            "1 Planer and 1 Horizontal drill",
        ],
    }
    data["maintenance"] = pd.DataFrame(maintenance)
    # demand dataset
    demand = {
        1: [500, 600, 300, 200, 0, 500],
        2: [1000, 500, 600, 300, 100, 500],
        3: [300, 200, 0, 400, 500, 100],
        4: [300, 0, 0, 500, 100, 300],
        5: [800, 400, 500, 200, 1000, 1100],
        6: [200, 300, 400, 0, 300, 500],
        7: [100, 150, 100, 100, 0, 60],
    }
    index = ["January", "February", "March", "April", "May", "June"]
    data["demand"] = pd.DataFrame(demand, index=index)
    # machines dataset
    machines = {
        "Grinding": 4,
        "Vertical drilling": 2,
        "Horizontal drilling": 3,
        "Boring": 1,
        "Planing": 1,
    }
    data["machines"] = pd.DataFrame(machines, index=["Number of machines"]).T
    return data


data = generate_data()
manufacturing = data["manufacturing"]
maintenance = data["maintenance"]
demand = data["demand"]
machines = data["machines"]

# create model
SOLVER = "glpk"
model = pyo.ConcreteModel()

# create sets
set_time = range(1, demand.shape[0] + 1)
set_products = range(1, manufacturing.shape[1] + 1)
set_machines = machines.index.tolist()

# create parameters
STORE_COST = 0.5
STORE_CAPACITY = 100
FINAL_STOCK = 50
SHIFTS_A_DAY = 2
HOURS_A_SHIFT = 8
DAYS_A_MONTH = 24
HOURS_A_MONTH = SHIFTS_A_DAY * HOURS_A_SHIFT * DAYS_A_MONTH

# profit per product
profit = {k: v for k, v in zip(set_products, manufacturing.iloc[0, :].tolist())}
# number of machines per machine type
no_machines = {k: v for k, v in zip(set_machines, machines.iloc[:, 0].tolist())}
# hours needed per machine type per product
hours_per_machine_type_per_product = (
    manufacturing.rename(columns=lambda x: int(x.replace("PROD ", "")))
    .loc[manufacturing.index != "Contribution to profit", :]
    .fillna(0)
    .stack()
    .to_dict()
)
# maintenance per machine type per month
outage_per_machine_type_per_month = {k: 0 for k in list(product(no_machines.keys(), set_time))}
outage_per_machine_type_per_month[("Grinding", 1)] = -1
outage_per_machine_type_per_month[("Horizontal drilling", 2)] = -2
outage_per_machine_type_per_month[("Boring", 3)] = -1
outage_per_machine_type_per_month[("Vertical drilling", 4)] = -1
outage_per_machine_type_per_month[("Vertical drilling", 5)] = -1
outage_per_machine_type_per_month[("Grinding", 5)] = -1
outage_per_machine_type_per_month[("Horizontal drilling", 6)] = -1
outage_per_machine_type_per_month[("Planing", 6)] = -1
# operating machines per machine type per month
operating_machines_per_machine_type_per_month = {
    k: no_machines[k[0]] + v for k, v in outage_per_machine_type_per_month.items()
}
# demand per product per month
limits_per_product_per_month = demand.set_index(pd.Index(set_time)).unstack().to_dict()

# create variables
model.make = pyo.Var(set_products, set_time, domain=pyo.NonNegativeReals)
model.store = pyo.Var(
    set_products, set_time, bounds=(0, STORE_CAPACITY), domain=pyo.NonNegativeReals
)
model.sell = pyo.Var(set_products, set_time, domain=pyo.NonNegativeReals)
for (p, t), upper_bound in limits_per_product_per_month.items():
    model.sell[p, t].setub(upper_bound)
    model.sell[p, t].setlb(0)

# create objective
model.obj = pyo.Objective(
    expr=sum(model.make[p, t] * profit[p] for p in set_products for t in set_time)
    - sum(model.store[p, t] * STORE_COST for p in set_products for t in set_time),
    sense=pyo.maximize,
)

# create constraint : INITIAL BALANCE
model.cnstr_initial_balance = pyo.ConstraintList()
for p in set_products:  # ! just for first month
    model.cnstr_initial_balance.add(model.make[p, 1] == model.store[p, 1] + model.sell[p, 1])

# create constraint : BALANCE
model.cnstr_monthly_demand = pyo.ConstraintList()
for t in list(set_time)[1:]:  # ! skip first month
    for p in set_products:
        model.cnstr_monthly_demand.add(
            model.make[p, t] + model.store[p, t - 1] == model.sell[p, t] + model.store[p, t]
        )

# create constraint : INVENTORY TARGET
model.cnstr_final_stock = pyo.ConstraintList()
for p in set_products:  # ! just for last month
    model.cnstr_final_stock.add(model.store[p, demand.shape[0]] == FINAL_STOCK)

# create constraint : MACHINE CAPACITY
model.cnstr_monthly_machine_hours = pyo.ConstraintList()
for m in set_machines:
    for t in set_time:
        machine_hours_expr = sum(
            model.make[p, t] * hours_per_machine_type_per_product[m, p]
            for p in set_products
            if hours_per_machine_type_per_product[m, p] > 0
        )
        # model.cnstr_monthly_machine_hours.add(machine_hours_expr >= 0)
        model.cnstr_monthly_machine_hours.add(
            machine_hours_expr
            <= operating_machines_per_machine_type_per_month[m, t] * HOURS_A_MONTH
        )

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
    with open("src/16.factory-planning/solution.json", "w") as f:
        model.solutions.store_to(results)
        results.write(num=1, format="json", ostream=f)

    # Access other solution information as needed
    print(f"Profit ${pyo.value(model.obj):=.2f}")
    print(75 * "-")

    make_plan = (
        pd.DataFrame([(k[0], k[1], round(pyo.value(v), 2)) for k, v in model.make.items()])
        .rename(columns={0: "product", 1: "month", 2: "make"})
        .pivot(index="month", columns="product", values="make")
    )
    print("Production Plan:")
    print(make_plan)
    print(75 * "-")

    sell_plan = (
        pd.DataFrame([(k[0], k[1], round(pyo.value(v), 2)) for k, v in model.sell.items()])
        .rename(columns={0: "product", 1: "month", 2: "sell"})
        .pivot(index="month", columns="product", values="sell")
    )
    print("Sales Plan:")
    print(sell_plan)
    print(75 * "-")

    store_plan = (
        pd.DataFrame([(k[0], k[1], round(pyo.value(v), 2)) for k, v in model.store.items()])
        .rename(columns={0: "product", 1: "month", 2: "store"})
        .pivot(index="month", columns="product", values="store")
    )
    print("Inventory Plan:")
    print(store_plan)
    print(75 * "-")

else:
    print("Solver did not find an optimal solution.")
