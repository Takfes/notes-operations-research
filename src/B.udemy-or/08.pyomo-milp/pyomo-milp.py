import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

SOLVER = "cplex_direct"

# ============================================
# instantiate model
# ============================================
model = pyo.ConcreteModel()

# ============================================
# define variables
# ============================================
nb_vars = 5
model.x = pyo.Var(range(nb_vars), within=Integers, bounds=(0,np.inf))
model.y = pyo.Var(bounds=(0,float('inf')))

# ============================================
# define constraints
# ============================================

# contraint 1
my_sum = sum([model.x[i] for i in range(nb_vars)]) + model.y
model.c1 = pyo.Constraint(expr = my_sum <= 20)
model.c1.pprint()

# contraint 2
model.c2 = pyo.ConstraintList()
for i in range(nb_vars):
    model.c2.add(expr= model.x[i] + model.y >=15)
model.c2.pprint()

# contraint 3
model.c3 = pyo.Constraint(expr = sum([model.x[i] * (i+1) for i in range(nb_vars)]) >=10 )
model.c3.pprint()

# constraint 4
model.c4 = pyo.Constraint(expr = model.x[4] + 2*model.y >=30)
model.c4.pprint()

# ============================================
# define objective
# ============================================
model.obj = pyo.Objective(expr=my_sum, sense=minimize)
model.obj.pprint()

# ============================================
# define solver and solve
# ============================================
opt = SolverFactory(SOLVER)
results = opt.solve(model)

sol_x, sol_y = [pyo.value(model.x[i]) for i in range(nb_vars)], pyo.value(model.y)

print(f'{sol_x=}')
print(f'{sol_y=}')
print(f'{pyo.value(model.obj)=}')