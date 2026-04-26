# Operations Research Problems - 978-1-4471-5576-8 978-1-4471-5577-5
# 1.10 Portfolio of Investments

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

investments_data = {
    "Telefonita": {
        "Price_per_share": 100,
        "Annual_rate_of_return": 0.12,
        "Measure_of_risk_per_$": 0.10,
    },
    "Sankander": {
        "Price_per_share": 50,
        "Annual_rate_of_return": 0.08,
        "Measure_of_risk_per_$": 0.07,
    },
    "Ferrofial": {
        "Price_per_share": 80,
        "Annual_rate_of_return": 0.06,
        "Measure_of_risk_per_$": 0.05,
    },
    "Gamefa": {
        "Price_per_share": 40,
        "Annual_rate_of_return": 0.10,
        "Measure_of_risk_per_$": 0.08,
    },
}

data = pd.DataFrame(investments_data)

# create model
model = pyo.ConcreteModel()

# define sets
set_options_o = data.columns.to_list()

# define parameters
ANNUAL_RETURN = 0.09
MAX_EXPOSURE_PER_INVESTMENT = 0.5
TOTAL_BUDGET = 200_000
RETURNS = data.loc["Annual_rate_of_return", :].to_dict()
PRICES = data.loc["Price_per_share", :].to_dict()
RISKS = data.loc["Measure_of_risk_per_$", :].to_dict()

# define variables
model.shares = pyo.Var(set_options_o, domain=pyo.NonNegativeReals)

# define objective function
model.obj = pyo.Objective(
    expr=sum(model.shares[o] * PRICES[o] * RISKS[o] for o in set_options_o), sense=pyo.minimize
)

# define constraint : ANNUAL RETURN
model.annual_return = pyo.Constraint(
    expr=sum(RETURNS[o] * PRICES[o] * model.shares[o] for o in set_options_o)
    >= ANNUAL_RETURN * TOTAL_BUDGET
)

# define constraint : MAX EXPOSURE PER INVESTMENT
model.max_exposure = pyo.ConstraintList()
for o in set_options_o:
    model.max_exposure.add(
        expr=model.shares[o] * PRICES[o] <= MAX_EXPOSURE_PER_INVESTMENT * TOTAL_BUDGET
    )

# define constraint : TOTAL BUDGET
model.budget = pyo.Constraint(
    expr=sum(PRICES[o] * model.shares[o] for o in set_options_o) == TOTAL_BUDGET
)

model.pprint()

# solve model
opt = SolverFactory("glpk")
results = opt.solve(model)

if (
    results.solver.status == pyo.SolverStatus.ok
    and results.solver.termination_condition == pyo.TerminationCondition.optimal
):
    # Access and save the solution to a file (e.g., JSON)
    with open("src/14.portfolio-investment/solution.json", "w") as f:
        model.solutions.store_to(results)
        results.write(num=1, format="json", ostream=f)

    print(f"Total Objective : {pyo.value(model.obj):=.2f}")

    total_investment = sum([model.shares[o].value * PRICES[o] for o in set_options_o])
    print(f"Total Investment ${total_investment:,.0f}")

    for o in set_options_o:
        print(
            f"{o} * {model.shares[o].value:.1f} @ ${PRICES[o]:.0f} => ${model.shares[o].value * PRICES[o]:=,.0f}"
        )
