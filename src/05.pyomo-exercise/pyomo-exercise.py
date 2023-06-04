import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

SOLVER = "glpk"

model = pyo.ConcreteModel()

x = pyo.Var(bounds=(-float("inf"), 3))
y = pyo.Var(bounds=(0, float("inf")))

model.x = x
model.y = y

model.C1 = pyo.Constraint(expr=x + y <= 8)
model.C2 = pyo.Constraint(expr=8 * x + 3 * y >= -24)
model.C3 = pyo.Constraint(expr=-6 * x + 8 * y <= 48)
model.C4 = pyo.Constraint(expr=3 * x + 5 * y <= 15)

model.obj = pyo.Objective(expr=x + y, sense=minimize)

opt = SolverFactory(SOLVER)
opt.solve(model)

model.pprint()

x_value = pyo.value(x)
y_value = pyo.value(y)

print(f"{x_value=}")
print(f"{y_value=}")
