import highspy
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ---------------------------------------------
# Read Data
# ---------------------------------------------

# dfr = pd.read_excel(
#     "scripts/monitoring/data/M5-Driver-Allocation.xlsx", sheet_name="Case1"
# )

# df = dfr.assign(
#     Driver=lambda x: x["To"].astype(str),
#     Driver_Worked_Hours=lambda x: round(x["Hours_worked_to"], 2),
#     Route=lambda x: x["From"].astype(str),
#     Route_Duration_Hours=lambda x: round(x["truck_duration_from"] / 60, 2),
# )[["Driver", "Driver_Worked_Hours", "Route", "Route_Duration_Hours"]]

df = pd.read_clipboard()

# Driver	Driver_Worked_Hours	Route	Route_Duration_Hours
# D15006	56.45	R15006	10.67
# D15006	56.45	R15009	4.75


# ---------------------------------------------
# Parse Data
# ---------------------------------------------

driver_hours = (
    df[["Driver", "Driver_Worked_Hours"]].set_index("Driver").to_dict()["Driver_Worked_Hours"]
)

route_hours = (
    df[["Route", "Route_Duration_Hours"]].set_index("Route").to_dict()["Route_Duration_Hours"]
)

eligibility = [(row["Route"], row["Driver"]) for _, row in df[["Route", "Driver"]].iterrows()]

# ---------------------------------------------
# Define Solver
# ---------------------------------------------

# http://www.osemosys.org/uploads/1/8/5/0/18504136/glpk_installation_guide_for_windows10_-_201702.pdf
# https://stackoverflow.com/questions/60888032/how-to-install-cbc-for-pyomo-locally-on-windows-machine
# https://stackoverflow.com/questions/59951763/install-ipopt-solver-to-use-with-pyomo-in-windows
# https://groups.google.com/g/pyomo-forum/c/tApZf3i2cso

# Show all available solvers
print(SolverFactory.__dict__["_cls"].keys())
print("")

SOLVER = "glpk"  # "ipopt" "appsi_highs" "cbc"
if SolverFactory(SOLVER).available():
    print("Solver " + SOLVER + " is available. :)")
else:
    print("Solver " + SOLVER + " is not available. :(")

# ---------------------------------------------
# Make m
# ---------------------------------------------

# create m
m = pyo.ConcreteModel()

# define sets
m.drivers = driver_hours.keys()
m.routes = route_hours.keys()

# decision variables
m.x = pyo.Var(eligibility, domain=pyo.Binary)

# constraint driver assignment : each driver can only be assigned to one route
m.driver_assignment = pyo.ConstraintList()
for d in m.drivers:
    m.driver_assignment.add(sum(m.x[r, d] for r in m.routes if (r, d) in eligibility) <= 1)

# constraing route assignment : each route must be assigned to one driver
m.route_assignment = pyo.ConstraintList()
for r in m.routes:
    m.route_assignment.add(sum(m.x[r, d] for d in m.drivers if (r, d) in eligibility) == 1)


# objective function
# m.obj = pyo.Objective(
#     expr=max(
#         sum(m.x[r, d] * route_hours[r] for r in m.routes if (r, d) in eligibility)
#         + driver_hours[d]
#         for d in m.drivers
#     )
#     - min(
#         sum(m.x[r, d] * route_hours[r] for r in m.routes if (r, d) in eligibility)
#         + driver_hours[d]
#         for d in m.drivers
#     ),
#     sense=pyo.minimize,
# )

# New auxiliary variables
m.max_hours = pyo.Var()
m.min_hours = pyo.Var()


# Constraints to ensure max_hours and min_hours behave correctly
def max_hours_rule(m, d):
    return (
        sum(m.x[r, d] * route_hours[r] for r in m.routes if (r, d) in eligibility)
        + driver_hours[d]
        <= m.max_hours
    )


m.max_hours_constr = pyo.Constraint(m.drivers, rule=max_hours_rule)


def min_hours_rule(m, d):
    return (
        sum(m.x[r, d] * route_hours[r] for r in m.routes if (r, d) in eligibility)
        + driver_hours[d]
        >= m.min_hours
    )


m.min_hours_constr = pyo.Constraint(m.drivers, rule=min_hours_rule)

# Modified objective
m.obj = pyo.Objective(expr=m.max_hours - m.min_hours, sense=pyo.minimize)


# ---------------------------------------------
# Solve m & Show Results
# ---------------------------------------------

opt = SolverFactory(SOLVER)
results = opt.solve(m)

if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    # with open("src/50.driver-assignment/solution.json", "w") as f:
    #     m.solutions.store_to(results)
    #     results.write(num=1, format="json", ostream=f)

    # Access solution information as needed
    print(50 * "-")
    print(f"min_hours {pyo.value(m.min_hours):=.2f}")
    print(f"max_hours {pyo.value(m.max_hours):=.2f}")
    print(f"Objective {pyo.value(m.obj):=.2f}")
    print(50 * "-")

    idx = 1
    for x in m.x:
        if pyo.value(m.x[x]) == 1:
            print(f"{str(idx).zfill(2)}) {x[0]} -> {x[1]}")
            idx += 1

    solution = pd.DataFrame(
        [(x[0] + x[1], x[0], x[1], pyo.value(m.x[x])) for x in m.x],
        columns=["Key", "Route", "Driver", "Assignment"],
    )

else:
    print("Solver did not find an optimal solution.")

solution.to_clipboard(index=False)
