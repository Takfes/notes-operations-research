import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

SOLVER = "cplex_direct"  # "gplk"

model = pyo.ConcreteModel()

# x = pyo.Var(bounds=(0, 10))
x = pyo.Var(within=Integers,bounds=(0, 10))
y = pyo.Var(bounds=(0, 10))

model.x = x
model.y = y

# model.C1 = pyo.Constraint(expr=-x + 2 * y <= 8)
model.C1 = pyo.Constraint(expr=-x + 2 * y <= 7)
model.C2 = pyo.Constraint(expr=2 * x + y <= 14)
model.C3 = pyo.Constraint(expr=2 * x - y <= 10)

model.obj = pyo.Objective(expr=x + y, sense=maximize)

opt = SolverFactory(SOLVER)
opt.solve(model)

model.pprint()

x_value = pyo.value(x)
y_value = pyo.value(y)

print(f"{x_value=}")
print(f"{y_value=}")
